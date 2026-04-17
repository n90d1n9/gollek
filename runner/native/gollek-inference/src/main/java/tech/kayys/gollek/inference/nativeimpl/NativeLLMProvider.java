package tech.kayys.gollek.inference.nativeimpl;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.subscription.MultiEmitter;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.tokenizer.GGUFTokenizer;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.*;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Implementation of LLMProvider using the pure Java-native inference engine.
 */
@ApplicationScoped
public final class NativeLLMProvider implements StreamingProvider {

    private static final float DEFAULT_REPETITION_PENALTY = 1.1f;
    private static final int REPETITION_WINDOW = 64;

    private final Map<String, NativeInferenceEngine> engines = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    
    @Override
    public String id() {
        return "native";
    }

    @Override
    public String name() {
        return "Gollek Native Inference Engine (Pure Java)";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(id())
                .name(name())
                .version("0.1.0")
                .description("High-performance CPU inference via Vector API and FFM (Zero-Copy)")
                .vendor("Gollek")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .supportedFormats(java.util.Set.of(tech.kayys.gollek.spi.model.ModelFormat.GGUF))
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
        // Initialization handled lazily or via Bootstrap
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return modelId.endsWith(".gguf");
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return inferStream(request)
                .onItem().transform(StreamingInferenceChunk::getDelta)
                .filter(java.util.Objects::nonNull)
                .collect().in(StringBuilder::new, StringBuilder::append)
                .onItem().transform(sb -> {
                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .model(request.getModel())
                            .content(sb.toString())
                            .durationMs(0) // Metrics recorded by router
                            .build();
                });
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                executeInference(request, emitter);
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    private void executeInference(ProviderRequest request, MultiEmitter<? super StreamingInferenceChunk> emitter) throws Exception {
        String modelPath = request.getModel();
        NativeInferenceEngine engine = getOrLoadEngine(modelPath);
        GGUFTokenizer tokenizer = new GGUFTokenizer(engine.getModel());

        // 1. Prepare prompt
        StringBuilder prompt = new StringBuilder();
        for (Message msg : request.getMessages()) {
            prompt.append(msg.getContent()).append("\n");
        }

        long[] tokens = tokenizer.encode(prompt.toString(), EncodeOptions.builder()
                .addBos(tokenizer.shouldAddBos())
                .addEos(tokenizer.shouldAddEos())
                .build());
        
        System.out.println("Prompt: [" + prompt.toString().replace("\n", "\\n") + "]");
        System.out.print("Token IDs: ");
        for (long t : tokens) System.out.print(t + " ");
        System.out.println();
        
        try (NativeInferenceSession session = new NativeInferenceSession(engine, 4096)) {
            // Initial prefill
            for (int i = 0; i < tokens.length - 1; i++) {
                session.tick((int) tokens[i], executor);
            }

            int lastToken = (int) tokens[tokens.length - 1];
            int maxNewTokens = request.getMaxTokens();
            float temp = (float) request.getTemperature();
            float topP = (float) request.getTopP();

            // Token history for repetition penalty
            List<Integer> tokenHistory = new ArrayList<>();
            for (long t : tokens) {
                tokenHistory.add((int) t);
            }

            // Scope for logits allocation
            try (Arena localArena = Arena.ofShared()) {
                String arch = (String) engine.getModel().metadata().getOrDefault("general.architecture", "llama");
                Object vsObj = engine.getModel().metadata().get(arch + ".vocab_size");
                if (vsObj == null) vsObj = engine.getModel().metadata().get("llama.vocab_size");
                
                int vocabSize = ((Number) vsObj).intValue();
                MemorySegment logits = localArena.allocate((long) vocabSize * Float.BYTES, 64);

                for (int i = 0; i < maxNewTokens; i++) {
                    MemorySegment hidden = session.tick(lastToken, executor);
                    
                    // Logit Projection
                    LogitProjectionKernel.execute(
                        hidden, 
                        engine.getOutputWeight(), 
                        logits, 
                        engine.getHidden(), 
                        vocabSize
                    );

                    // Get recent tokens for repetition penalty (last N tokens)
                    List<Integer> recentTokens = tokenHistory.subList(
                        Math.max(0, tokenHistory.size() - REPETITION_WINDOW),
                        tokenHistory.size()
                    );

                    // Sampling with repetition penalty
                    int nextToken = SamplingKernel.sample(
                        logits, vocabSize, temp, 40, topP,
                        DEFAULT_REPETITION_PENALTY, recentTokens
                    );
                    
                    // Multi-EOS detection: stop on primary EOS or any additional EOS tokens
                    if (tokenizer.isEosToken(nextToken)) {
                        break;
                    }

                    String piece = tokenizer.decode(new long[]{nextToken}, tech.kayys.gollek.tokenizer.spi.DecodeOptions.builder().build());
                    emitter.emit(StreamingInferenceChunk.of(
                            request.getRequestId(),
                            i,
                            piece
                    ));

                    tokenHistory.add(nextToken);
                    lastToken = nextToken;
                }
            }
            emitter.complete();
        }
    }

    private NativeInferenceEngine getOrLoadEngine(String path) throws Exception {
        return engines.computeIfAbsent(path, p -> {
            try {
                GGUFReader reader = new GGUFReader(Path.of(p));
                GGUFParser parser = new GGUFParser();
                return new NativeInferenceEngine(parser.parse(reader.segment()));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy(id()));
    }

    @Override
    public void shutdown() {
        engines.values().forEach(NativeInferenceEngine::close);
        executor.shutdown();
    }
}
