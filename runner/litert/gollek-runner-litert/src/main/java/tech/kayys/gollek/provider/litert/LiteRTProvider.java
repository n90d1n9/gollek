package tech.kayys.gollek.provider.litert;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderMetrics;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.AdapterCapabilityProfile;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.observability.AdapterMetricTagResolver;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.pipeline.ModelPipeline;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRegistry;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRequest;

import java.time.Duration;
import java.time.Instant;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * LiteRT Provider for Wayang/Gollek.
 *
 * <p>Implements the LLMProvider SPI using LiteRT 2.0.
 * This class has been refactored to be POJO-first, allowing manual instantiation
 * while remaining compatible with CDI-based environments.
 */
@jakarta.enterprise.context.ApplicationScoped
@io.quarkus.arc.Unremovable
public class LiteRTProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(LiteRTProvider.class);
    private static final String PROVIDER_ID = "litert";
    private static final String PROVIDER_VERSION = "2.0.0"; // Upgraded to 2.0 based on LiteRT 2.0

    @jakarta.inject.Inject
    LiteRTProviderConfig config;
    
    @jakarta.inject.Inject
    AdapterMetricsRecorder adapterMetricsRecorder = new NoopAdapterMetricsRecorder();
    
    @jakarta.inject.Inject
    LiteRTSessionManager sessionManager;

    @jakarta.inject.Inject
    ModelPipelineRegistry pipelineRegistry;

    private volatile boolean initialized = false;

    /**
     * Default constructor for SPI/CDI discovery.
     */
    public LiteRTProvider() {
    }

    /**
     * Manual constructor for standalone/JBang use cases.
     */
    public LiteRTProvider(LiteRTProviderConfig config) {
        this.config = config;
        this.sessionManager = new LiteRTSessionManager(config);
    }

    // Setters for manual/Quarkus-like injection if needed outside CDI
    public void setConfig(LiteRTProviderConfig config) {
        this.config = config;
    }

    public void setAdapterMetricsRecorder(AdapterMetricsRecorder recorder) {
        this.adapterMetricsRecorder = recorder != null ? recorder : new NoopAdapterMetricsRecorder();
    }

    public void setSessionManager(LiteRTSessionManager sessionManager) {
        this.sessionManager = sessionManager;
    }

    @Override
    public boolean isEnabled() {
        return config == null || config.enabled();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "LiteRT 2.0 Provider (ARM64 Optimized)";
    }

    @Override
    public String version() {
        return PROVIDER_VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(name())
                .version(PROVIDER_VERSION)
                .description("Stable LiteRT 2.0 inference provider for macOS/Linux ARM64")
                .vendor("Kayys Tech")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        var features = new java.util.LinkedHashSet<>(Set.of("local_inference", "cpu_inference", "compiled_model_2.0"));
        features.addAll(AdapterCapabilityProfile.unsupportedWithMetrics().toFeatureFlags());
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(false)
                .multimodal(false)
                .toolCalling(false)
                .embeddings(false)
                .maxContextTokens(2048)
                .maxOutputTokens(512)
                .supportedFormats(Set.of(ModelFormat.LITERT))
                .supportedDevices(config != null ? LiteRTDeviceSupport.supportedDevices(config) : Set.of(DeviceType.CPU))
                .features(Set.copyOf(features))
                .build();
    }

    @Override
    public void initialize(ProviderConfig cfg) throws ProviderException.ProviderInitializationException {
        if (config == null) {
            // Attempt to resolve config from cfg if possible, or expect it to be set already
            LOG.warn("LiteRTProvider initialized without explicit config mapping.");
            config = defaultConfig();
        }
        
        if (sessionManager == null) {
            sessionManager = new LiteRTSessionManager(config);
        }

        if (sessionManager != null) {
            sessionManager.startEvictor();
        }
        
        initialized = true;
        LOG.infov("LiteRT 2.0 Provider initialized (GPU Auto-Metal: {0})", 
                config != null && LiteRTDeviceSupport.shouldAutoMetal(config));
    }

    private LiteRTProviderConfig defaultConfig() {
        return new LiteRTProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

            @Override
            public String modelBasePath() {
                return Path.of(System.getProperty("user.home"), ".gollek", "models", "litert").toString();
            }

            @Override
            public int threads() {
                return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
            }

            @Override
            public boolean gpuEnabled() {
                return false;
            }

            @Override
            public boolean autoMetalEnabled() {
                return true;
            }

            @Override
            public boolean npuEnabled() {
                return false;
            }

            @Override
            public String gpuBackend() {
                return "auto";
            }

            @Override
            public String npuType() {
                return "auto";
            }

            @Override
            public Duration defaultTimeout() {
                return Duration.ofSeconds(30);
            }

            @Override
            public LiteRTProviderConfig.SessionConfig session() {
                return new LiteRTProviderConfig.SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 2;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 8;
                    }
                };
            }
        };
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (AdapterMetricTagResolver.hasAdapterRequest(request)) {
            return false;
        }
        Path path = resolveModelPath(modelId, request);
        return path != null && Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            if (AdapterMetricTagResolver.hasAdapterRequest(request)) {
                throw new UnsupportedOperationException(
                        "LiteRT provider does not support adapters (adapter_unsupported).");
            }
            String tenantId = resolveTenantId(request);
            String adapterId = PROVIDER_ID;
            String adapterType = AdapterMetricTagResolver.resolveAdapterType(request);
            
            adapterMetricsRecorder.recordSuccess(
                    AdapterMetricSchema.builder()
                            .adapterId(adapterId)
                            .modelId(request.getModel())
                            .operation("request_receive")
                            .tags(java.util.Map.of("adapter_type", adapterType))
                            .build(),
                    0);

            Path modelPath = resolveModelPath(request.getModel(), request);
            ModelPipelineRequest pipelineRequest = modelPipelineRequest(request, modelPath);
            Optional<ModelPipeline> pipeline = selectFeaturePipeline(pipelineRequest);
            if (pipeline.isPresent()) {
                return pipeline.get().infer(pipelineRequest)
                        .await()
                        .atMost(request.getTimeout());
            }
            boolean gpuEnabled = config != null && LiteRTDeviceSupport.effectiveGpuEnabled(config);
            String gpuBackend = config != null ? LiteRTDeviceSupport.resolveGpuBackend(config) : "auto";
            
            LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(
                    config != null ? config.threads() : 4,
                    gpuEnabled,
                    config != null && config.npuEnabled(),
                    gpuBackend,
                    config != null ? config.npuType() : "auto");

            LiteRTSessionManager.SessionContext sessionContext = null;
            Instant sessionAcquireStart = Instant.now();
            try {
                if (sessionManager == null) {
                    throw new InferenceException("LiteRT Session manager not initialized");
                }
                
                sessionContext = sessionManager.getSession(tenantId, request.getModel(), modelPath, runnerConfig);
                adapterMetricsRecorder.recordSuccess(
                        AdapterMetricSchema.builder()
                                .adapterId(adapterId)
                                .modelId(request.getModel())
                                .operation("session_acquire")
                                .tags(java.util.Map.of("adapter_type", adapterType))
                                .build(),
                        Duration.between(sessionAcquireStart, Instant.now()).toMillis());

                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(request.isStreaming())
                        .build();

                InferenceResponse response = sessionContext.runner().infer(inferenceRequest);

                return InferenceResponse.builder()
                        .requestId(response.getRequestId())
                        .content(response.getContent())
                        .model(request.getModel())
                        .durationMs(response.getDurationMs())
                        .tokensUsed(response.getTokensUsed())
                        .metadata(response.getMetadata())
                        .metadata("provider", PROVIDER_ID)
                        .metadata("litert_version", "2.0")
                        .build();
            } catch (Exception e) {
                String userFacingMessage = userFacingLiteRtFailureMessage(e);
                if (userFacingMessage != null) {
                    LOG.error("LiteRT 2.0 Inference failed: " + userFacingMessage);
                    throw new InferenceException(userFacingMessage);
                }
                LOG.error("LiteRT 2.0 Inference failed: " + e.getMessage(), e);
                throw new InferenceException("Inference failed", e);
            } finally {
                if (sessionContext != null && sessionManager != null) {
                    sessionManager.releaseSession(tenantId, request.getModel(), sessionContext);
                }
            }
        });
    }

    private String resolveTenantId(ProviderRequest request) {
        Object tenantId = request.getMetadata().get("tenantId");
        if (tenantId instanceof String && !((String) tenantId).isBlank()) {
            return (String) tenantId;
        }
        if (request.getUserId().isPresent()) {
            return request.getUserId().get();
        }
        if (request.getApiKey().isPresent()) {
            return request.getApiKey().get();
        }
        return "community";
    }

    private Path resolveModelPath(String modelId, ProviderRequest request) {
        if (modelId == null) return null;
        
        LOG.debugf("Resolving model path for modelId: %s, request parameters: %s", modelId, (request != null ? request.getParameters() : "null"));

        // Check for explicit model_path parameter (e.g. from CLI --modelFile)
        if (request != null && request.getParameters().containsKey("model_path")) {
            Object pathObj = request.getParameters().get("model_path");
            LOG.debugf("Found model_path parameter: %s", pathObj);
            if (pathObj instanceof String s && !s.isBlank()) {
                try {
                    Path p = Paths.get(s);
                    if (Files.exists(p)) {
                        Path nativeGemmaLitertlm = findNativeGemmaLitertlm(p);
                        if (nativeGemmaLitertlm != null) {
                            LOG.debugf("Explicit model_path preferred native Gemma LiteRT-LM: %s", nativeGemmaLitertlm);
                            return nativeGemmaLitertlm;
                        }
                        Optional<Path> bestFile = LiteRTContainerParser.findBestModelFile(p);
                        LOG.debugf("Model path resolved to: %s", bestFile.orElse(p));
                        return bestFile.orElse(p);
                    } else {
                        LOG.debugf("Explicit model_path does not exist: %s", s);
                    }
                } catch (Exception e) {
                    LOG.debug("Failed to resolve explicit model_path: " + s);
                }
            }
        }

        try {
            Path targetPath = null;
            if (modelId.startsWith("file://")) {
                targetPath = Paths.get(modelId.substring("file://".length()));
            } else {
                Path asPath = Paths.get(modelId);
                if (asPath.isAbsolute() && Files.exists(asPath)) {
                    targetPath = asPath;
                } else if (config != null) {
                    Path basePath = Paths.get(config.modelBasePath());
                    Path modelDir = basePath.resolve(modelId);
                    
                    if (Files.exists(modelDir)) {
                        targetPath = modelDir;
                    } else {
                        Path legacyPath = basePath.resolve(modelId + ".litertlm");
                        targetPath = Files.exists(legacyPath) ? legacyPath : modelDir;
                    }
                }
            }

            if (targetPath != null && Files.exists(targetPath)) {
                Path nativeGemmaLitertlm = findNativeGemmaLitertlm(targetPath);
                if (nativeGemmaLitertlm != null) {
                    LOG.debugf("Preferred native Gemma LiteRT-LM artifact: %s", nativeGemmaLitertlm);
                    return nativeGemmaLitertlm;
                }
                java.util.Optional<Path> bestFile = LiteRTContainerParser.findBestModelFile(targetPath);
                return bestFile.orElse(targetPath);
            }
            return targetPath;
        } catch (Exception e) {
            LOG.warn("Error resolving model path for " + modelId, e);
            return config != null ? Paths.get(config.modelBasePath()).resolve(modelId) : Paths.get(modelId);
        }
    }

    private Path findNativeGemmaLitertlm(Path sourcePath) {
        try {
            Path searchDir = Files.isDirectory(sourcePath) ? sourcePath : sourcePath.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return null;
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".litertlm"))
                        .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).contains("gemma"))
                        .sorted((left, right) -> Integer.compare(
                                litertlmPreferenceScore(left),
                                litertlmPreferenceScore(right)))
                        .findFirst()
                        .orElse(null);
            }
        } catch (Exception e) {
            LOG.debugf("Failed to find native Gemma LiteRT-LM sibling for %s: %s", sourcePath, e.getMessage());
            return null;
        }
    }

    private int litertlmPreferenceScore(Path candidate) {
        String fileName = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (fileName.contains("qualcomm")) {
            score += 10;
        }
        if (!fileName.contains("gemma-4-e2b-it.litertlm")) {
            score += 1;
        }
        return score;
    }

    @Override
    public Uni<ProviderHealth> health() {
        if (!initialized) {
            return Uni.createFrom().item(ProviderHealth.healthy("LiteRT provider pending initialization"));
        }
        return Uni.createFrom().item(ProviderHealth.healthy("LiteRT 2.0 provider operational"));
    }

    @Override
    public void shutdown() {
        if (sessionManager != null) {
            sessionManager.shutdown();
        }
        initialized = false;
        LOG.info("LiteRT 2.0 Provider shutdown complete");
    }

    @Override
    public io.smallrye.mutiny.Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            ensureInitialized();
            if (AdapterMetricTagResolver.hasAdapterRequest(request)) {
                emitter.fail(new UnsupportedOperationException(
                        "LiteRT provider does not support adapters (adapter_unsupported)."));
                return;
            }

            String tenantId = resolveTenantId(request);
            Path modelPath = resolveModelPath(request.getModel(), request);
            ModelPipelineRequest pipelineRequest = modelPipelineRequest(request, modelPath);
            Optional<ModelPipeline> pipeline = selectFeaturePipeline(pipelineRequest);
            if (pipeline.isPresent()) {
                pipeline.get().inferStream(pipelineRequest).subscribe().with(
                        emitter::emit,
                        emitter::fail,
                        emitter::complete);
                return;
            }
            boolean gpuEnabled = config != null && LiteRTDeviceSupport.effectiveGpuEnabled(config);
            String gpuBackend = config != null ? LiteRTDeviceSupport.resolveGpuBackend(config) : "auto";

            LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(
                    config != null ? config.threads() : 4,
                    gpuEnabled,
                    config != null && config.npuEnabled(),
                    gpuBackend,
                    config != null ? config.npuType() : "auto");

            LiteRTSessionManager.SessionContext sessionContext = null;
            Instant started = Instant.now();
            AtomicInteger chunkIndex = new AtomicInteger(0);

            try {
                if (sessionManager == null) {
                    throw new InferenceException("LiteRT Session manager not initialized");
                }

                sessionContext = sessionManager.getSession(tenantId, request.getModel(), modelPath, runnerConfig);
                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(true)
                        .build();

                sessionContext.runner().stream(inferenceRequest, delta -> {
                    Map<String, Object> metadata = Map.of(
                            "provider", PROVIDER_ID,
                            "litert_version", "2.0");
                    emitter.emit(new StreamingInferenceChunk(
                            request.getRequestId(),
                            chunkIndex.getAndIncrement(),
                            ModalityType.TEXT,
                            delta,
                            null,
                            false,
                            null,
                            null,
                            Instant.now(),
                            metadata));
                });

                Map<String, Object> finalMetadata = new LinkedHashMap<>();
                finalMetadata.put("provider", PROVIDER_ID);
                finalMetadata.put("litert_version", "2.0");
                finalMetadata.put("duration_ms", Duration.between(started, Instant.now()).toMillis());
                finalMetadata.put("tokens.output", chunkIndex.get());
                emitter.emit(new StreamingInferenceChunk(
                        request.getRequestId(),
                        chunkIndex.getAndIncrement(),
                        ModalityType.TEXT,
                        null,
                        null,
                        true,
                        "stop",
                        null,
                        Instant.now(),
                        finalMetadata));
                emitter.complete();
            } catch (Exception e) {
                String userFacingMessage = userFacingLiteRtFailureMessage(e);
                if (userFacingMessage != null) {
                    emitter.fail(new InferenceException(userFacingMessage));
                } else {
                    LOG.error("LiteRT 2.0 Streaming inference failed: " + e.getMessage(), e);
                    emitter.fail(new InferenceException("Streaming inference failed", e));
                }
            } finally {
                if (sessionContext != null && sessionManager != null) {
                    sessionManager.releaseSession(tenantId, request.getModel(), sessionContext);
                }
            }
        });
    }

    private Optional<ModelPipeline> selectFeaturePipeline(ModelPipelineRequest request) {
        if (pipelineRegistry == null) {
            return Optional.empty();
        }
        return pipelineRegistry.select(request);
    }

    private ModelPipelineRequest modelPipelineRequest(ProviderRequest request, Path modelPath) {
        Map<String, Object> facts = new LinkedHashMap<>();
        put(facts, "provider", PROVIDER_ID);
        put(facts, "format", "litert");
        put(facts, "litert.model_path", modelPath == null ? null : modelPath.toAbsolutePath().normalize().toString());
        put(facts, "litert.model_name", modelPath == null || modelPath.getFileName() == null
                ? null
                : modelPath.getFileName().toString());
        put(facts, "litert.model_directory", modelPath == null ? null : Files.isDirectory(modelPath));
        put(facts, "litert.model_regular_file", modelPath == null ? null : Files.isRegularFile(modelPath));
        put(facts, "litert.model_size_bytes", regularFileSize(modelPath));
        put(facts, "litert.artifact_type", litertArtifactType(modelPath));
        put(facts, "litert.native_gemma_lm", modelPath != null
                && modelPath.getFileName() != null
                && modelPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".litertlm"));
        boolean gpuEnabled = config != null && LiteRTDeviceSupport.effectiveGpuEnabled(config);
        String gpuBackend = config != null ? LiteRTDeviceSupport.resolveGpuBackend(config) : "auto";

        return ModelPipelineRequest.builder(request)
                .providerId(PROVIDER_ID)
                .modelPath(modelPath)
                .facts(facts)
                .attribute("litert.threads", config != null ? config.threads() : 4)
                .attribute("litert.gpu_enabled", gpuEnabled)
                .attribute("litert.gpu_backend", gpuBackend)
                .build();
    }

    private static void put(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static Long regularFileSize(Path path) {
        try {
            return path != null && Files.isRegularFile(path) ? Files.size(path) : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private static String litertArtifactType(Path path) {
        if (path == null || path.getFileName() == null) {
            return null;
        }
        String fileName = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (fileName.endsWith(".litertlm")) {
            return "litertlm";
        }
        if (fileName.endsWith(".tflite")) {
            return "tflite";
        }
        if (fileName.endsWith(".task")) {
            return "task";
        }
        if (Files.isDirectory(path)) {
            return "directory";
        }
        return "unknown";
    }

    private String userFacingLiteRtFailureMessage(Throwable throwable) {
        String rootMessage = rootCauseMessage(throwable);
        if (rootMessage == null || rootMessage.isBlank()) {
            return null;
        }
        if (rootMessage.contains("strict LiteRT mode")) {
            return rootMessage;
        }
        if (rootMessage.contains("Gemma 4 LiteRT .task runner is disabled by default")
                || rootMessage.contains("Raw Gemma LiteRT-LM signatures are disabled by default")) {
            return rootMessage;
        }
        return null;
    }

    private String rootCauseMessage(Throwable throwable) {
        if (throwable == null) {
            return null;
        }
        Throwable cursor = throwable;
        while (cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        return cursor.getMessage();
    }

    private void ensureInitialized() {
        if (!initialized) {
            try {
                initialize(null);
            } catch (Exception e) {
                LOG.error("Lazy initialization failed", e);
            }
        }
    }
}
