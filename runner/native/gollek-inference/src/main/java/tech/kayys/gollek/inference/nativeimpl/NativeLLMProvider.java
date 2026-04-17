package tech.kayys.gollek.inference.nativeimpl;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.subscription.MultiEmitter;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.tokenizer.GGUFTokenizer;
import tech.kayys.gollek.spi.inference.*;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ApplicationScoped
public class NativeLLMProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(NativeLLMProvider.class);
    
    @Override
    public String id() {
        return "native";
    }

    @Override
    public String name() {
        return "Native LLM Provider";
    }

    @Override
    public ProviderMetadata metadata() {
        return null; // Placeholder
    }

    @Override
    public ProviderCapabilities capabilities() {
        return null; // Placeholder
    }

    @Override
    public void initialize(ProviderConfig config) {
        // Init logic
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return modelId != null && modelId.contains("native");
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("Native engine is ready"));
    }

    @Override
    public void shutdown() {
        executor.shutdown();
    }

    // Shared executor for parallel kernel operations
    private final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    
    // Simple cache for engine instances
    private final Map<String, NativeInferenceEngine> engineCache = new HashMap<>();

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
                            .build();
                });
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                executeInference(request, emitter);
            } catch (Exception e) {
                LOG.error("Inference failed", e);
                emitter.fail(e);
            }
        });
    }

    private synchronized NativeInferenceEngine getOrLoadEngine(String modelPath) throws Exception {
        if (!engineCache.containsKey(modelPath)) {
            GGUFModel model = GGUFLoader.loadModel(Path.of(modelPath));
            NativeInferenceEngine engine = new NativeInferenceEngine(model);
            engineCache.put(modelPath, engine);
        }
        return engineCache.get(modelPath);
    }

    private void executeInference(ProviderRequest request, MultiEmitter<? super StreamingInferenceChunk> emitter) throws Exception {
        String modelPath = request.getModel();
        long loadStartTime = System.nanoTime();
        NativeInferenceEngine engine = getOrLoadEngine(modelPath);
        long loadEndTime = System.nanoTime();
        
        GGUFTokenizer tokenizer = new GGUFTokenizer(engine.getModel());

        // 1. Prepare prompt with ChatTemplate
        String arch = (String) engine.getModel().metadata().getOrDefault("general.architecture", "llama");
        ChatTemplate template = ChatTemplate.forArchitecture(arch);
        String promptText = template.apply(request.getMessages());

        long[] tokens = tokenizer.encode(promptText, EncodeOptions.builder().build());
        
        LOG.debugf("Prompt Tokens: %d", tokens.length);
        
        try (NativeInferenceSession session = new NativeInferenceSession(engine, 4096)) {
            long prefillStartTime = System.nanoTime();
            
            // Prefill loop (process all but the last prompt token)
            for (int i = 0; i < tokens.length - 1; i++) {
                session.tick((int) tokens[i], executor);
            }

            long prefillEndTime = System.nanoTime();
            long firstTokenTime = 0;
            
            int lastToken = (int) tokens[tokens.length - 1];
            int maxNewTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 512;
            
            List<Integer> tokenHistory = new ArrayList<>();
            for (long t : tokens) tokenHistory.add((int) t);

            int generatedCount = 0;
            try (Arena localArena = Arena.ofShared()) {
                Object vsObj = engine.getModel().metadata().get(arch + ".vocab_size");
                if (vsObj == null) vsObj = engine.getModel().metadata().get("llama.vocab_size");
                int vocabSize = ((Number) vsObj).intValue();
                MemorySegment logits = localArena.allocate((long) vocabSize * Float.BYTES, 64);

                for (int i = 0; i < maxNewTokens; i++) {
                    MemorySegment hidden = session.tick(lastToken, executor);
                    
                    if (i == 0) firstTokenTime = System.nanoTime();

                    // Logit Projection
                    LogitProjectionKernel.execute(hidden, engine.getOutputWeight(), logits, engine.getHidden(), vocabSize);
                    
                    // Simple Greedy Sampling (temp=0.0f)
                    lastToken = SamplingKernel.sample(logits, vocabSize, 0.0f, 0, 0.0f, 1.0f, tokenHistory);
                    tokenHistory.add(lastToken);
                    generatedCount++;

                    if (tokenizer.isEosToken(lastToken)) break;

                    String delta = tokenizer.decode(new long[]{lastToken}, null);
                    emitter.emit(StreamingInferenceChunk.of(request.getRequestId(), i, delta));
                }
            }

            long generationEndTime = System.nanoTime();
            
            // 2. Wrap up metrics
            Map<String, Object> metrics = new HashMap<>();
            metrics.put("bench.load_ms", (loadEndTime - loadStartTime) / 1_000_000.0);
            metrics.put("bench.prefill_ms", (prefillEndTime - prefillStartTime) / 1_000_000.0);
            metrics.put("bench.decode_ms", (generationEndTime - prefillEndTime) / 1_000_000.0);
            metrics.put("bench.ttft_ms", (firstTokenTime - prefillStartTime) / 1_000_000.0);
            
            double prefillTps = (tokens.length - 1) / ((prefillEndTime - prefillStartTime) / 1_000_000_000.0);
            double genTps = generatedCount / ((generationEndTime - prefillEndTime) / 1_000_000_000.0);
            
            metrics.put("bench.prefill_tps", prefillTps);
            metrics.put("bench.generation_tps", genTps);
            metrics.put("tokens.input", tokens.length);
            metrics.put("tokens.output", generatedCount);

            emitter.emit(StreamingInferenceChunk.withMetadata(
                request.getRequestId(), 
                generatedCount, 
                "", 
                metrics
            ));
            
            LOG.infof("Inference Bench: Prefill %.2f t/s, Gen %.2f t/s, TTFT %.2f ms", prefillTps, genTps, (firstTokenTime - prefillStartTime) / 1_000_000.0);
            
            emitter.complete();
        }
    }
}
