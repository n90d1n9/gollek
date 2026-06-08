package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextGeneration;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackend;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelection;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelector;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionPreparationPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionManager;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionReuseDecision;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionStateResolver;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Path;

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

    @Inject
    InferenceRequestRuntimeConfigMapper runtimeConfigMapper;

    @Inject
    InferencePromptPlanner promptPlanner;

    @Inject
    InferenceModelPathResolver modelPathResolver;

    @Inject
    RequestQuantizationPlanner quantizationPlanner;

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
        QuantizationEngine.QuantStrategy quantStrategy = quantizationPlanner().resolve(request);
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
        ModelRuntimeTraits runtimeTraits = loadedModel != null
                ? loadedModel.runtimeTraits()
                : ModelRuntimeTraits.EMPTY;

        boolean audioModel = runtimeTraits.audioModel();
        boolean verbose = Boolean.getBoolean(VERBOSE_PROPERTY);

        DirectForwardPlatformPolicy.Decision forwardPlatformDecision =
                DirectForwardPlatformPolicy.apply(runtimeTraits, architecture);
        if (forwardPlatformDecision.forceCpuForward() || forwardPlatformDecision.experimentalMetalAllowed()) {
            System.err.println("⚠ Safetensor forward path: " + forwardPlatformDecision.reason());
        }

        if (verbose) {
            System.err.println("[DEBUG-ROUTING] modelPath: " + modelPath);
            System.err.println("[DEBUG-ROUTING] loadedModel null? " + (loadedModel == null));
            System.err.println("[DEBUG-ROUTING] config null? " + (loadedModel != null && loadedModel.config() == null));
            System.err.println("[DEBUG-ROUTING] modelType: '" + modelType + "'");
            System.err.println("[DEBUG-ROUTING] arch: '" + architecture + "'");
            System.err.println("[DEBUG-ROUTING] runtimeTraits: " + runtimeTraits);
            System.err.println("[DEBUG-ROUTING] isAudioModel: " + audioModel);
            System.err.println("[DEBUG-ROUTING] forwardPlatform: " + forwardPlatformDecision);
        }

        boolean requiresCpuForward = forwardPlatformDecision.forceCpuForward();
        InferencePromptPlan promptPlan =
                promptPlanner().plan(request.getMessages(), modelType, runtimeTraits, requiresCpuForward);
        PreparedPrompt preparedPrompt = promptPlan.preparedPrompt();
        if (verbose) {
            System.err.println("FORMATTED PROMPT: " + preparedPrompt.formattedPrompt().replace("\n", "\\n"));
        }

        GenerationConfig generationConfig = runtimeConfigMapper().generationConfig(request, preparedModel.capabilities());
        AudioConfig audioConfig = runtimeConfigMapper().audioConfig(request);
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
                promptPlan.ttsPrompt(),
                audioConfig,
                loadedModel,
                preparedModel,
                preparedPrompt,
                conversationExecutionState,
                sessionReuseDecision,
                preparedGeneration);
    }

    public Path resolveModelPathFromRequest(ProviderRequest request) {
        return modelPathResolver().resolve(request, config);
    }

    private ConversationExecutionStateResolver conversationStateResolver() {
        return conversationStateResolver != null
                ? conversationStateResolver
                : new ConversationExecutionStateResolver();
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

    private InferenceRequestRuntimeConfigMapper runtimeConfigMapper() {
        return runtimeConfigMapper != null
                ? runtimeConfigMapper
                : new InferenceRequestRuntimeConfigMapper();
    }

    private InferencePromptPlanner promptPlanner() {
        return promptPlanner != null
                ? promptPlanner
                : new InferencePromptPlanner();
    }

    private InferenceModelPathResolver modelPathResolver() {
        return modelPathResolver != null
                ? modelPathResolver
                : new InferenceModelPathResolver();
    }

    private RequestQuantizationPlanner quantizationPlanner() {
        return quantizationPlanner != null
                ? quantizationPlanner
                : new RequestQuantizationPlanner();
    }
}
