package tech.kayys.gollek.safetensor.engine.sdk;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.prompt.PromptTemplateCompat;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.model.SystemInfo;
import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderRequests;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * GollekSdk implementation backed by the SafeTensor DirectInferenceEngine.
 *
 * <p>This bridges the high-level SDK API (used by NLP pipelines, Gollek.java facade,
 * and the GollekClient builder) with the actual SafeTensor inference pipeline.
 *
 * <p>When a user calls:
 * <pre>{@code
 * var gen = Gollek.pipeline("text-generation", "Qwen/Qwen2.5-0.5B-Instruct");
 * String result = gen.process("Tell me about Java");
 * }</pre>
 * The call chain is:
 * <ol>
 *   <li>TextGenerationPipeline → GollekSdk.createCompletion()</li>
 *   <li>SafetensorGollekSdk.createCompletion() → DirectInferenceEngine.generate()</li>
 * </ol>
 */
@ApplicationScoped
public class SafetensorGollekSdk implements GollekSdk {

    private static final Logger log = Logger.getLogger(SafetensorGollekSdk.class);
    private static final Pattern GEMMA4_THOUGHT_CHANNEL =
            Pattern.compile("^<\\|channel>thought\\n.*?<channel\\|>", Pattern.DOTALL);
    private static final Pattern GEMMA4_GENERIC_CHANNEL_OPEN =
            Pattern.compile("^<\\|channel>[^\\n]*\\n", Pattern.DOTALL);
    private static final String ALLOW_UNSAFE_GEMMA3_DIRECT_PROPERTY =
            "gollek.safetensor.allow_unsafe_gemma3_direct";

    private final DirectInferenceEngine engine;
    private final Path modelBasePath;
    private final ProviderRegistry providerRegistry;
    private String preferredProvider = "safetensor";

    /**
     * Creates a new SafetensorGollekSdk with the given inference engine.
     *
     * @param engine        the direct inference engine (may be CDI-managed or standalone)
     * @param modelBasePath base directory for model discovery
     */
    @Inject
    public SafetensorGollekSdk(
            DirectInferenceEngine engine,
            SafetensorProviderConfig config,
            ProviderRegistry providerRegistry) {
        this(engine, resolveModelBasePath(config), providerRegistry);
    }

    public SafetensorGollekSdk(DirectInferenceEngine engine, Path modelBasePath) {
        this(engine, modelBasePath, null);
    }

    public SafetensorGollekSdk(DirectInferenceEngine engine, Path modelBasePath, ProviderRegistry providerRegistry) {
        this.engine = engine;
        this.modelBasePath = modelBasePath;
        this.providerRegistry = providerRegistry;
        log.infof("SafetensorGollekSdk initialized with model base path [%s]", modelBasePath);
    }

    private static Path resolveModelBasePath(SafetensorProviderConfig config) {
        if (config != null && config.basePath() != null && !config.basePath().isBlank()) {
            return Path.of(config.basePath());
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "models");
    }

    // ==================== Core Inference ====================

    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            if (useProviderPath() && !shouldUseLegacyDirectPath(request)) {
                return requireProvider()
                        .infer(buildProviderRequest(request, false))
                        .await()
                        .atMost(request.getTimeout().orElse(Duration.ofMinutes(5)));
            }
            Path modelPath = resolveModelPath(request.getModel());
            GenerationConfig cfg = toGenerationConfig(request);
            String modelType = readModelType(modelPath);
            validateLegacyDirectModelSupport(modelType);
            String prompt = buildLegacyPrompt(request, modelPath);

            Uni<InferenceResponse> uni = engine.generate(prompt, modelPath, cfg);
            InferenceResponse response = uni.await().atMost(Duration.ofMinutes(5));
            return sanitizeLegacyResponse(response, modelType);
        } catch (Exception e) {
            throw new SdkException("SafeTensor inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public java.util.concurrent.CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                return createCompletion(request);
            } catch (SdkException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public Multi<StreamingInferenceChunk> streamCompletion(InferenceRequest request) {
        try {
            if (useProviderPath() && !shouldUseLegacyDirectPath(request)) {
                return requireStreamingProvider().inferStream(buildProviderRequest(request, true));
            }
            Path modelPath = resolveModelPath(request.getModel());
            GenerationConfig cfg = toGenerationConfig(request);
            String modelType = readModelType(modelPath);
            validateLegacyDirectModelSupport(modelType);
            String prompt = buildLegacyPrompt(request, modelPath);
            String requestId = request.getRequestId() != null
                    ? request.getRequestId()
                    : java.util.UUID.randomUUID().toString();

            AtomicInteger index = new AtomicInteger(0);
            AtomicBoolean leadingGemma4ChannelsPending = new AtomicBoolean(true);
            return engine.generateStream(prompt, modelPath, cfg)
                    .map(response -> {
                        String content = response.getContent();
                        if (content != null) {
                            content = sanitizeGemma4StreamingDelta(content, modelType, leadingGemma4ChannelsPending);
                        }
                        int idx = index.getAndIncrement();
                        boolean isFinal = response.getFinishReason() != null;
                        if (isFinal) {
                            return StreamingInferenceChunk.finalChunk(requestId, idx, content);
                        }
                        return StreamingInferenceChunk.textDelta(requestId, idx, content);
                    });
        } catch (Exception e) {
            return Multi.createFrom().failure(
                    new SdkException("Streaming failed: " + e.getMessage(), e));
        }
    }

    // ==================== Embeddings ====================

    @Override
    public EmbeddingResponse createEmbedding(EmbeddingRequest request) throws SdkException {
        throw new SdkException("Embedding extraction is not yet supported by the SafeTensor provider. " +
                "Use a cloud provider or ONNX runtime for embeddings.");
    }

    // ==================== Async Jobs ====================

    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        createCompletionAsync(request);
        return java.util.UUID.randomUUID().toString();
    }

    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        return new AsyncJobStatus(jobId, null, "UNKNOWN", null, null, null, null);
    }

    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        return getJobStatus(jobId);
    }

    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        return batchRequest.requests().stream()
                .map(req -> {
                    try { return createCompletion(req); }
                    catch (SdkException e) { throw new RuntimeException(e); }
                })
                .toList();
    }

    // ==================== Provider Operations ====================

    @Override
    public List<ProviderInfo> listAvailableProviders() throws SdkException {
        return List.of(ProviderInfo.builder()
                .id("safetensor")
                .name("SafeTensor Direct Inference")
                .description("Zero-copy SafeTensor model loading and inference via LibTorch FFM")
                .version("0.1.0")
                .build());
    }

    @Override
    public ProviderInfo getProviderInfo(String providerId) throws SdkException {
        return listAvailableProviders().getFirst();
    }

    @Override
    public void setPreferredProvider(String providerId) throws SdkException {
        this.preferredProvider = providerId;
    }

    @Override
    public Optional<String> getPreferredProvider() {
        return Optional.of(preferredProvider);
    }

    // ==================== Model Operations ====================

    @Override
    public List<ModelInfo> listModels() throws SdkException {
        return engine.listLoadedModels().stream()
                .map(m -> ModelInfo.builder()
                        .modelId(m.key())
                        .name(m.key())
                        .format("SAFETENSORS")
                        .build())
                .toList();
    }

    @Override
    public List<ModelInfo> listModels(int offset, int limit) throws SdkException {
        var all = listModels();
        return all.subList(Math.min(offset, all.size()), Math.min(offset + limit, all.size()));
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelId) throws SdkException {
        return listModels().stream()
                .filter(m -> m.getModelId().equals(modelId))
                .findFirst();
    }

    @Override
    public void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException {
        Path modelPath = resolveModelPath(modelSpec);
        if (modelPath == null || !Files.exists(modelPath)) {
            throw new SdkException("Model not found locally: " + modelSpec +
                    ". SafeTensor provider requires pre-downloaded models in " + modelBasePath);
        }
        log.infof("Model [%s] already available at [%s]", modelSpec, modelPath);
    }

    @Override
    public void deleteModel(String modelId) throws SdkException {
        Path modelPath = resolveModelPath(modelId);
        if (modelPath != null) {
            engine.unloadModel(modelPath);
        }
    }

    // ==================== System Operations ====================

    @Override
    public SystemInfo getSystemInfo() throws SdkException {
        return SystemInfo.builder()
                .cliVersion("0.1.0")
                .javaVersion(System.getProperty("java.version"))
                .osName(System.getProperty("os.name"))
                .osVersion(System.getProperty("os.version"))
                .osArch(System.getProperty("os.arch"))
                .build();
    }

    // ==================== Internal helpers ====================

    private Path resolveModelPath(String modelId) {
        if (modelId == null) return null;

        // 1. Absolute path
        Path direct = Path.of(modelId);
        if (Files.exists(direct)) return direct;

        // 2. Relative to model base path
        Path relative = modelBasePath.resolve(modelId);
        if (Files.exists(relative)) return relative;

        // 3. HuggingFace-style with separators normalized
        String normalized = modelId.replace("/", "--");
        Path hfStyle = modelBasePath.resolve(normalized);
        if (Files.exists(hfStyle)) return hfStyle;

        // 4. Check ~/.gollek/models
        Path gollekModels = Path.of(System.getProperty("user.home"), ".gollek", "models");
        Path gollekPath = gollekModels.resolve(normalized);
        if (Files.exists(gollekPath)) return gollekPath;

        return relative;
    }

    private GenerationConfig toGenerationConfig(InferenceRequest request) {
        Map<String, Object> params = request.getParameters();
        int maxTokens = params.containsKey("max_tokens")
                ? ((Number) params.get("max_tokens")).intValue() : 256;
        float temperature = params.containsKey("temperature")
                ? ((Number) params.get("temperature")).floatValue() : 0.7f;
        float topP = params.containsKey("top_p")
                ? ((Number) params.get("top_p")).floatValue() : 0.9f;
        int topK = params.containsKey("top_k")
                ? ((Number) params.get("top_k")).intValue() : 40;
        float minP = params.containsKey("min_p")
                ? ((Number) params.get("min_p")).floatValue() : 0.0f;
        float repetitionPenalty = params.containsKey("repeat_penalty")
                ? ((Number) params.get("repeat_penalty")).floatValue()
                : params.containsKey("repetition_penalty")
                ? ((Number) params.get("repetition_penalty")).floatValue()
                : 1.0f;
        float frequencyPenalty = params.containsKey("frequency_penalty")
                ? ((Number) params.get("frequency_penalty")).floatValue() : 0.0f;
        long seed = params.containsKey("seed")
                ? ((Number) params.get("seed")).longValue() : -1L;
        int maxKvCacheTokens = params.containsKey("max_kv_cache_tokens")
                ? ((Number) params.get("max_kv_cache_tokens")).intValue() : 2048;
        GenerationConfig.KvCacheQuantization kvCacheQuant = normalizeKvCacheQuantization(
                (String) params.getOrDefault("kv_cache_quant", "none"));

        return GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveSamplingStrategy(temperature, topK, topP))
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .minP(minP)
                .repetitionPenalty(repetitionPenalty)
                .frequencyPenalty(frequencyPenalty)
                .maxKvCacheTokens(maxKvCacheTokens)
                .kvCacheQuant(kvCacheQuant)
                .seed(seed)
                .build();
    }

    private String buildLegacyPrompt(InferenceRequest request, Path modelPath) {
        String rawPrompt = request.getPrompt();
        if (rawPrompt == null || rawPrompt.isBlank()) {
            return rawPrompt;
        }
        String modelType = readModelType(modelPath);
        boolean useRawPrompt = shouldUseRawLegacyPrompt(request, rawPrompt, modelType);
        boolean shouldFormat = shouldFormatLegacyPrompt(modelType);
        if (Boolean.getBoolean("gollek.verbose")) {
            int messageCount = request.getMessages() == null ? -1 : request.getMessages().size();
            System.out.printf(
                    "[DEBUG-PROMPT] modelType=%s shouldFormat=%s useRaw=%s messageCount=%d%n",
                    modelType, shouldFormat, useRawPrompt, messageCount);
            System.out.flush();
        }
        if (useRawPrompt || !shouldFormat) {
            return rawPrompt;
        }

        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return rawPrompt;
        }

        try {
            return PromptTemplateCompat.format(new ArrayList<>(messages), modelType);
        } catch (Exception e) {
            log.debugf("Falling back to raw prompt for legacy direct path: %s", e.getMessage());
            return rawPrompt;
        }
    }

    private boolean shouldUseRawLegacyPrompt(InferenceRequest request, String rawPrompt, String modelType) {
        if (request == null) {
            return true;
        }
        String normalizedModelType = modelType == null ? "" : modelType.trim().toLowerCase(Locale.ROOT);
        if (normalizedModelType.startsWith("gemma3") || normalizedModelType.startsWith("gemma4")) {
            return false;
        }
        List<Message> messages = request.getMessages();
        if (messages == null || messages.size() != 1) {
            return false;
        }
        Message only = messages.getFirst();
        if (only.getRole() != Message.Role.USER) {
            return false;
        }
        String messagePrompt = only.getContent();
        if (messagePrompt == null) {
            return false;
        }
        return rawPrompt.equals(messagePrompt);
    }

    private void validateLegacyDirectModelSupport(String modelType) {
        String normalized = modelType == null ? "" : modelType.trim().toLowerCase(Locale.ROOT);
        if (!normalized.startsWith("gemma3")) {
            return;
        }
        if (Boolean.getBoolean(ALLOW_UNSAFE_GEMMA3_DIRECT_PROPERTY)) {
            return;
        }
        throw new IllegalStateException(
                "Gemma3 direct safetensor path is temporarily disabled due incorrect generation quality. "
                        + "Use GGUF/LiteRT route, or override only for debugging with -D"
                        + ALLOW_UNSAFE_GEMMA3_DIRECT_PROPERTY + "=true");
    }

    private InferenceResponse sanitizeLegacyResponse(InferenceResponse response, String modelType) {
        if (response == null || response.getContent() == null) {
            return response;
        }
        String normalized = modelType == null ? "" : modelType.trim().toLowerCase(Locale.ROOT);
        if (!normalized.startsWith("gemma4")) {
            return response;
        }

        String content = response.getContent();
        content = GEMMA4_THOUGHT_CHANNEL.matcher(content).replaceFirst("");
        content = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(content).replaceFirst("");
        content = stripGemma4InlineMarkers(content);

        if (content.equals(response.getContent())) {
            return response;
        }
        return response.toBuilder().content(content).build();
    }

    /**
     * Strips Gemma 4 channel/control fragments from streamed deltas (non-streaming path uses
     * {@link #sanitizeLegacyResponse}).
     */
    private static String sanitizeGemma4StreamingDelta(
            String content, String modelType, AtomicBoolean leadingChannelsPending) {
        if (content == null || modelType == null) {
            return content;
        }
        String normalizedType = modelType.trim().toLowerCase(Locale.ROOT);
        if (!normalizedType.startsWith("gemma4")) {
            return content;
        }
        String text = content;
        if (leadingChannelsPending.get()) {
            text = GEMMA4_THOUGHT_CHANNEL.matcher(text).replaceFirst("");
            text = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(text).replaceFirst("");
            if (!text.isBlank()) {
                leadingChannelsPending.set(false);
            }
        }
        return stripGemma4InlineMarkers(text);
    }

    private static String stripGemma4InlineMarkers(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        String content = text.replace("<channel|>", "");
        content = content.replace("<turn|>", "");
        return content.replace("<|tool_response>", "");
    }

    private String readModelType(Path modelPath) {
        if (modelPath == null) {
            return "";
        }
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir == null) {
                return "";
            }
            ModelConfig config = ModelConfig.fromDirectory(configDir, new ObjectMapper());
            return config.modelType() != null ? config.modelType() : "";
        } catch (Exception e) {
            log.debugf("Unable to read model type for legacy direct prompt shaping: %s", e.getMessage());
            return "";
        }
    }

    private boolean shouldFormatLegacyPrompt(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String normalized = modelType.trim().toLowerCase(Locale.ROOT);
        return normalized.startsWith("gemma")
                || normalized.startsWith("llama")
                || normalized.startsWith("mistral")
                || normalized.startsWith("mixtral")
                || normalized.startsWith("phi")
                || normalized.startsWith("qwen");
    }

    private GenerationConfig.SamplingStrategy resolveSamplingStrategy(float temperature, int topK, float topP) {
        if (temperature < 1.0e-4f || topK == 1) {
            return GenerationConfig.SamplingStrategy.GREEDY;
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0f && topP < 1.0f;
        if (hasTopK && hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_K_TOP_P;
        }
        if (hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_P;
        }
        if (hasTopK) {
            return GenerationConfig.SamplingStrategy.TOP_K;
        }
        return GenerationConfig.SamplingStrategy.GREEDY;
    }

    private GenerationConfig.KvCacheQuantization normalizeKvCacheQuantization(String raw) {
        if (raw == null || raw.isBlank() || "none".equalsIgnoreCase(raw)) {
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        String normalized = raw.trim().toUpperCase(Locale.ROOT);
        if ("INT8".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT8;
        }
        if ("INT4".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        if ("TURBO".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        return GenerationConfig.KvCacheQuantization.NONE;
    }

    private boolean useProviderPath() {
        return providerRegistry != null && providerRegistry.hasProvider(preferredProvider);
    }

    private boolean shouldUseLegacyDirectPath(InferenceRequest request) {
        if (request == null) {
            return false;
        }
        String preferred = request.getPreferredProvider().orElse(preferredProvider);
        if (!"safetensor".equalsIgnoreCase(preferred)) {
            return false;
        }
        if (request.getSessionId().isPresent()) {
            return false;
        }
        if (request.getTools() != null && !request.getTools().isEmpty()) {
            return false;
        }
        if (request.getToolChoice() != null) {
            return false;
        }
        if (request.getAttachments() != null && !request.getAttachments().isEmpty()) {
            return false;
        }
        String prompt = request.getPrompt();
        return prompt != null && !prompt.isBlank();
    }

    private LLMProvider requireProvider() throws SdkException {
        if (providerRegistry == null) {
            throw new SdkException("SafeTensor provider registry is not available");
        }
        return providerRegistry.getProvider(preferredProvider)
                .orElseThrow(() -> new SdkException("SafeTensor provider is not available: " + preferredProvider));
    }

    private StreamingProvider requireStreamingProvider() throws SdkException {
        LLMProvider provider = requireProvider();
        if (!(provider instanceof StreamingProvider streamingProvider)) {
            throw new SdkException("SafeTensor provider does not support streaming: " + preferredProvider);
        }
        return streamingProvider;
    }

    private ProviderRequest buildProviderRequest(InferenceRequest request, boolean streaming) {
        Path resolvedModelPath = resolveModelPath(request.getModel());
        Map<String, Object> parameters = new java.util.HashMap<>(request.getParameters());
        if (resolvedModelPath != null) {
            parameters.put("model_path", resolvedModelPath.toString());
        }
        return ProviderRequests.fromInferenceRequest(
                request,
                request.getModel(),
                streaming,
                Duration.ofMinutes(5),
                preferredProvider,
                parameters,
                Map.of("request_id", request.getRequestId()));
    }
}
