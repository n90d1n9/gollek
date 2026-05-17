package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextGeneration;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackend;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelection;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelector;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendCapabilities;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionPreparationPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionManager;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionReuseDecision;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.prompt.PromptTemplateCompat;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionStateResolver;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Converts provider requests into backend-ready execution plans.
 *
 * <p>By isolating request shaping from provider glue, Gollek can evolve toward
 * backend-owned prompt/prefill/KV lifecycles instead of rebuilding that state on
 * every call site. This is aligned with the execution-model concerns discussed in
 * Shkolnikov (2026), arXiv:2603.04428.
 */
@ApplicationScoped
public class InferenceRequestPlanner {

    private static final Logger log = Logger.getLogger(InferenceRequestPlanner.class);
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String FORCE_CPU_GEMMA4_PROPERTY = "gollek.safetensor.force_cpu_gemma4";
    private static final String ALLOW_METAL_GEMMA4_PROPERTY = "gollek.safetensor.allow_metal_gemma4";
    private static final String VERBOSE_PROPERTY = "gollek.verbose";

    @Inject
    DirectInferenceEngine engine;

    @Inject
    SafetensorProviderConfig config;

    @Inject
    TextExecutionBackendSelector backendSelector;

    @Inject
    TextExecutionSessionManager sessionManager;

    @Inject
    TextExecutionPreparationPlanner preparationPlanner;

    @Inject
    ConversationExecutionStateResolver conversationStateResolver;

    public InferenceRequestPlanner() {
    }

    public InferenceRequestPlanner(DirectInferenceEngine engine, SafetensorProviderConfig config) {
        this.engine = engine;
        this.config = config;
    }

    public InferenceRequestPlanner(
            DirectInferenceEngine engine,
            SafetensorProviderConfig config,
            TextExecutionBackendSelector backendSelector) {
        this.engine = engine;
        this.config = config;
        this.backendSelector = backendSelector;
    }

    public InferenceRequestPlanner(
            DirectInferenceEngine engine,
            SafetensorProviderConfig config,
            TextExecutionBackendSelector backendSelector,
            TextExecutionSessionManager sessionManager) {
        this.engine = engine;
        this.config = config;
        this.backendSelector = backendSelector;
        this.sessionManager = sessionManager;
    }

    public InferenceRequestPlanner(
            DirectInferenceEngine engine,
            SafetensorProviderConfig config,
            TextExecutionBackendSelector backendSelector,
            TextExecutionSessionManager sessionManager,
            TextExecutionPreparationPlanner preparationPlanner) {
        this.engine = engine;
        this.config = config;
        this.backendSelector = backendSelector;
        this.sessionManager = sessionManager;
        this.preparationPlanner = preparationPlanner;
    }

    public InferenceRequestPlanner(
            DirectInferenceEngine engine,
            SafetensorProviderConfig config,
            TextExecutionBackendSelector backendSelector,
            TextExecutionSessionManager sessionManager,
            TextExecutionPreparationPlanner preparationPlanner,
            ConversationExecutionStateResolver conversationStateResolver) {
        this.engine = engine;
        this.config = config;
        this.backendSelector = backendSelector;
        this.sessionManager = sessionManager;
        this.preparationPlanner = preparationPlanner;
        this.conversationStateResolver = conversationStateResolver;
    }

    public PreparedInferenceRequest prepare(ProviderRequest request) {
        Path modelPath = resolveModelPathFromRequest(request);
        QuantizationEngine.QuantStrategy quantStrategy = parseQuantStrategy(request);
        TextExecutionBackendSelection backendSelection = selector().select();
        TextExecutionBackend textBackend = backendSelection.selectedBackend();
        PreparedTextModel preparedModel = textBackend.prepareModel(modelPath, null, quantStrategy);
        SafetensorEngine.LoadedModel loadedModel = preparedModel.loadedModel();

        String modelType = loadedModel != null && loadedModel.config() != null
                ? loadedModel.config().modelType()
                : "";
        String architecture = loadedModel != null && loadedModel.config() != null
                ? loadedModel.config().primaryArchitecture()
                : "";

        boolean audioModel = ("speecht5".equals(modelType) || "whisper".equals(modelType)
                || modelPath.toString().contains("speecht5")
                || (architecture != null && (architecture.toLowerCase().contains("speecht5")
                || architecture.toLowerCase().contains("whisper"))));
        boolean verbose = Boolean.getBoolean(VERBOSE_PROPERTY);

        configureForwardPlatform(modelType, architecture);

        if (verbose) {
            System.err.println("[DEBUG-ROUTING] modelPath: " + modelPath);
            System.err.println("[DEBUG-ROUTING] loadedModel null? " + (loadedModel == null));
            System.err.println("[DEBUG-ROUTING] config null? " + (loadedModel != null && loadedModel.config() == null));
            System.err.println("[DEBUG-ROUTING] modelType: '" + modelType + "'");
            System.err.println("[DEBUG-ROUTING] arch: '" + architecture + "'");
            System.err.println("[DEBUG-ROUTING] isAudioModel: " + audioModel);
        }

        List<tech.kayys.gollek.spi.Message> rawMessages = new ArrayList<>(request.getMessages());
        boolean hasSystem = rawMessages.stream()
                .anyMatch(message -> message.getRole() == tech.kayys.gollek.spi.Message.Role.SYSTEM);
        boolean injectDefaultSystem = !hasSystem
                && !requiresCpuForward(modelType, architecture)
                && !shouldSkipDefaultSystemPrompt(modelType, architecture);
        if (injectDefaultSystem) {
            String defaultSystem = modelType.toLowerCase().contains("qwen")
                    ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    : "You are a helpful assistant.";
            rawMessages.add(0, tech.kayys.gollek.spi.Message.system(defaultSystem));
        }

        @SuppressWarnings("unchecked")
        String prompt = PromptTemplateCompat.format((java.util.List) (Object) rawMessages, modelType);
        if (verbose) {
            System.err.println("FORMATTED PROMPT: " + prompt.replace("\n", "\\n"));
        }
        PreparedPrompt preparedPrompt = PreparedPrompt.of(prompt, injectDefaultSystem, modelType);

        GenerationConfig generationConfig = buildGenerationConfig(request, preparedModel.capabilities());
        String ttsPrompt = rawMessages.stream()
                .filter(message -> message.getRole() == tech.kayys.gollek.spi.Message.Role.USER)
                .map(tech.kayys.gollek.spi.Message::getContent)
                .reduce((first, second) -> second)
                .orElse(prompt);
        AudioConfig audioConfig = buildAudioConfig(request);
        ConversationExecutionState conversationExecutionState =
                conversationStateResolver().resolve(preparedModel, request.getSessionId().orElse(null));
        TextExecutionPreparationContext preparationContext =
                preparationPlanner().contextFor(preparedModel, conversationExecutionState);
        TextExecutionPreparationPlan preparationPlan =
                preparationPlanner().plan(preparedPrompt, preparationContext);
        PreparedTextGeneration preparedGeneration =
                textBackend.prepareGeneration(preparedPrompt, preparedModel, preparationPlan, generationConfig);
        TextExecutionSessionReuseDecision sessionReuseDecision = sessionManager().evaluate(preparedGeneration);

        return new PreparedInferenceRequest(
                modelPath,
                audioModel,
                ttsPrompt,
                audioConfig,
                loadedModel,
                preparedModel,
                preparedPrompt,
                conversationExecutionState,
                sessionReuseDecision,
                preparedGeneration);
    }

    public Path resolveModelPathFromRequest(ProviderRequest request) {
        java.util.Optional<String> fromParam = request.getParameter("model_path", String.class);
        if (fromParam.isPresent() && !fromParam.get().isBlank()) {
            log.debugf("InferenceRequestPlanner: using model_path from parameters: %s", fromParam.get());
            return resolveModelPath(fromParam.get());
        }

        Object fromMeta = request.getMetadata().get("model_path");
        if (fromMeta instanceof String s && !s.isBlank()) {
            log.debugf("InferenceRequestPlanner: using model_path from metadata: %s", s);
            return resolveModelPath(s);
        }

        log.debugf("InferenceRequestPlanner: no model_path override, resolving from model ID: %s", request.getModel());
        return resolveModelPath(request.getModel());
    }

    private GenerationConfig buildGenerationConfig(
            ProviderRequest request,
            TextExecutionBackendCapabilities capabilities) {
        int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 256;
        int topK = request.getTopK();
        float topP = (float) request.getTopP();
        float temperature = (float) request.getTemperature();
        float minP = request.getParameter("min_p", Number.class)
                .map(Number::floatValue)
                .orElse(0.0f);
        long seed = request.getParameter("seed", Number.class)
                .map(Number::longValue)
                .orElse(-1L);
        String kvQuantStr = request.getParameter("kv_cache_quant", String.class).orElse("none");
        GenerationConfig.KvCacheQuantization kvQuant = normalizeKvQuantization(kvQuantStr, capabilities);

        return GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveSamplingStrategy(temperature, topK, topP))
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .minP(minP)
                .repetitionPenalty((float) request.getRepeatPenalty())
                .kvCacheQuant(kvQuant)
                .seed(seed)
                .build();
    }

    private AudioConfig buildAudioConfig(ProviderRequest request) {
        String outputFormat = request.getParameter("output_format", String.class).orElse("wav");
        AudioConfig.Format format = AudioConfig.Format.WAV;
        try {
            format = AudioConfig.Format.valueOf(outputFormat.toUpperCase());
        } catch (Exception ignored) {
        }

        return AudioConfig.builder()
                .temperature((float) request.getTemperature())
                .format(format)
                .build();
    }

    private ConversationExecutionStateResolver conversationStateResolver() {
        return conversationStateResolver != null
                ? conversationStateResolver
                : new ConversationExecutionStateResolver();
    }

    private GenerationConfig.KvCacheQuantization normalizeKvQuantization(
            String raw,
            TextExecutionBackendCapabilities capabilities) {
        if (raw == null || raw.isBlank() || "none".equalsIgnoreCase(raw)) {
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        if (!capabilities.supportsKvCacheQuantization()) {
            log.warnf("KV cache quantization '%s' is not supported by backend '%s'; using NONE", raw, capabilities);
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        String normalized = raw.trim().toUpperCase();
        if ("INT8".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT8;
        }
        if ("INT4".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        if ("TURBO".equals(normalized)) {
            log.warnf("KV cache quantization '%s' falls back to packed INT4 in the direct safetensor runtime", raw);
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        return GenerationConfig.KvCacheQuantization.NONE;
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

    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            throw new ProviderException("Model ID is null or blank");
        }

        Path asPath = Path.of(modelId);
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        Path resolved = Path.of(config.basePath(), modelId);
        if (Files.exists(resolved)) {
            log.debugf("Resolved model '%s' to path: %s", modelId, resolved);
            return resolved;
        }

        log.warnf("Model path not found at %s, using as-is: %s", resolved, modelId);
        return asPath;
    }

    private QuantizationEngine.QuantStrategy parseQuantStrategy(ProviderRequest request) {
        String raw = request.getParameter("quantize_strategy", String.class).orElse("none");
        if (raw == null || raw.isBlank()) {
            return QuantizationEngine.QuantStrategy.NONE;
        }

        try {
            return QuantizationEngine.QuantStrategy.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            log.warnf("Unknown quantize_strategy '%s'; falling back to NONE", raw);
            return QuantizationEngine.QuantStrategy.NONE;
        }
    }

    private void configureForwardPlatform(String modelType, String architecture) {
        if (requiresCpuForward(modelType, architecture)) {
            System.setProperty(FORCE_CPU_FORWARD_PROPERTY, "true");
            System.err.println("⚠ Safetensor forward path: forcing CPU for " + architecture
                    + " because gollek.safetensor.force_cpu_gemma4=true");
            return;
        }
        if (isGemma4(modelType, architecture) && Boolean.getBoolean(ALLOW_METAL_GEMMA4_PROPERTY)) {
            System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
            System.err.println("⚠ Safetensor forward path: allowing Metal for Gemma4 experimental validation");
            return;
        }

        System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
    }

    private boolean requiresCpuForward(String modelType, String architecture) {
        if (!isGemma4(modelType, architecture)) {
            return false;
        }
        return Boolean.getBoolean(FORCE_CPU_GEMMA4_PROPERTY);
    }

    private boolean isGemma4(String modelType, String architecture) {
        String mt = modelType == null ? "" : modelType.toLowerCase();
        String arch = architecture == null ? "" : architecture.toLowerCase();
        return mt.startsWith("gemma4") || arch.contains("gemma4");
    }

    private boolean shouldSkipDefaultSystemPrompt(String modelType, String architecture) {
        String mt = modelType == null ? "" : modelType.toLowerCase();
        String arch = architecture == null ? "" : architecture.toLowerCase();
        return mt.startsWith("gemma4") || arch.contains("gemma4");
    }

    private TextExecutionBackendSelector selector() {
        return backendSelector != null
                ? backendSelector
                : new TextExecutionBackendSelector(engine);
    }

    private TextExecutionSessionManager sessionManager() {
        return sessionManager != null
                ? sessionManager
                : new TextExecutionSessionManager();
    }

    private TextExecutionPreparationPlanner preparationPlanner() {
        return preparationPlanner != null
                ? preparationPlanner
                : new TextExecutionPreparationPlanner();
    }
}
