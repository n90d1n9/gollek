package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.arc.Arc;
import io.quarkus.runtime.Quarkus;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.sdk.model.ModelResolver;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.regex.Pattern;
import tech.kayys.gollek.cli.chat.ChatUIRenderer;
import tech.kayys.gollek.cli.runtime.CliMetalRuntime;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;

/**
 * Run inference command using GollekSdk.
 * Usage: gollek run --model <model> --prompt <prompt> [--provider
 * litert|llamacpp|gemini] [--stream]
 */
@Dependent
@Unremovable
@Command(name = "run", description = "Run inference using a specified model")
public class RunCommand implements Runnable {
    private static final Pattern GEMMA4_THOUGHT_CHANNEL =
            Pattern.compile("^<\\|channel>thought\\n.*?<channel\\|>", Pattern.DOTALL);
    private static final Pattern GEMMA4_GENERIC_CHANNEL_OPEN =
            Pattern.compile("^<\\|channel>[^\\n]*\\n", Pattern.DOTALL);
    private static final String DEFAULT_RUN_SYSTEM_PROMPT = "Answer directly and briefly.";
    private static final String QWEN_SAFETENSOR_RUN_SYSTEM_PROMPT = "You are a helpful assistant.";
    private static final String DISABLE_DEFAULT_RUN_SYSTEM_PROPERTY = "gollek.cli.disable_default_run_system";
    private Map<String, Object> forcedExecutionRouteMetadata = Map.of();

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    GollekSdk sdk;
    @Inject
    ProviderRegistry providerRegistry;
    @Inject
    PluginAvailabilityChecker pluginChecker;
    @Inject
    ChatUIRenderer uiRenderer;
    @Inject
    DirectInferenceEngine directInferenceEngine;

    @Option(names = { "-m", "--model" }, description = "Model ID for repository resolution (e.g., huggingface ID)")
    public String modelId;

    @Option(names = { "--modelFile" }, description = "Path to a local model file (.gguf, .tflite, .task, .litertlm)")
    public String modelFile;

    @Option(names = { "--modelDir" }, description = "Path to a local model directory (Safetensors)")
    public String modelDir;

    @Option(names = { "-p", "--prompt" }, description = "Input prompt", required = true)
    public String prompt;

    @Option(names = {
            "--provider" }, description = "Provider: native, litert, llamacpp, safetensor, libtorch(experimental), gemini, openai, anthropic, cerebras. Omit for auto-detection.", arity = "0..1", fallbackValue = "")
    String providerId;

    @Option(names = {
            "--import" }, description = "Import (move) the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean importModel;

    @Option(names = {
            "--copy" }, description = "Copy the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean copyModel;

    @Option(names = { "-s", "--stream" }, description = "Stream output", defaultValue = "true")
    boolean stream;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
    double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.9")
    double topP;

    @Option(names = { "--top-k" }, description = "Top-k sampling", defaultValue = "40")
    int topK;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty", defaultValue = "1.1")
    double repeatPenalty;

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "false")
    boolean jsonMode;

    @Option(names = {
            "--enable-json" }, description = "Emit OpenAI-compatible SSE JSON for streamed chunks", defaultValue = "false")
    boolean enableJsonSse;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "256")
    int maxTokens;

    @Option(names = { "--mirostat" }, description = "Mirostat mode (0, 1, 2)", defaultValue = "0")
    int mirostat;

    @Option(names = { "--grammar" }, description = "GBNF grammar string")
    String grammar;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = { "--session-id" }, description = "Persistent conversation session id for KV/prefix reuse across repeated run calls")
    String sessionId;

    @Option(names = { "--no-cache" }, description = "Bypass response cache")
    boolean noCache;

    @Option(names = { "--offline",
            "--local" }, description = "Force using existing models without checking for updates/downloads")
    boolean offline;

    @Option(names = {
            "--model-path" }, description = "Path to a custom model file (bypasses repository lookup, prefer --modelFile)")
    String modelPath;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Option(names = { "-b", "--branch" }, description = "HuggingFace branch/revision (e.g. main, fp16, onnx)")
    String branch;

    @Option(names = { "--force" }, description = "Force re-download and replace existing files", defaultValue = "false")
    boolean force;

    @Option(names = { "--format" }, description = "Preferred model format (e.g., safetensors, gguf, onnx)")
    String format;

    @Option(names = { "--direct" }, description = "Force use of native Safetensor direct engine", defaultValue = "false")
    boolean direct;

    @Option(names = { "--gguf-quant" }, description = "Specify quantization for GGUF conversion (e.g. Q4_K_M)")
    String ggufQuant;

    @Option(names = { "--gguf" }, description = "Force GGUF conversion and usage", defaultValue = "false")
    boolean forceGguf;

    @Option(names = { "--fallback" }, split = ",", description = "Comma-separated fallback model IDs or short IDs to try only if the primary model is incompatible")
    List<String> fallbackModelIds;

    @Option(names = { "--quantize" }, description = "Enable JIT quantization during inference (bnb, turbo, awq, gptq, autoround)")
    String quantizeStrategy;

    @Option(names = { "--quantize-bits" }, description = "Bit width for JIT quantization (default: 4)", defaultValue = "4")
    int quantizeBits;

    // Stable Diffusion specific parameters
    @Option(names = { "--seed" }, description = "Random seed for image generation (default: random)")
    Long seed;

    @Option(names = { "--output", "-o" }, description = "Output image file path (default: output.png)")
    String outputPath;

    @Option(names = { "--steps" }, description = "Number of denoising steps (default: 20)")
    Integer steps;

    @Option(names = { "--guidance-scale", "--cfg" }, description = "Classifier-free guidance scale (default: 7.5)")
    Float guidanceScale;

    @Option(names = { "--width" }, description = "Output image width in pixels (default: 512, must be multiple of 64)")
    Integer width;

    @Option(names = {
            "--height" }, description = "Output image height in pixels (default: 512, must be multiple of 64)")
    Integer height;

    @Option(names = { "--ext" }, description = "Explicit output file extension (mp3, wav, flac, png, jpg, mp4). Overrides extension in --output if provided.")
    String extension;

    @Option(names = { "--quantize-kv" }, description = "Enable KV cache quantization (none, int8, int4, turbo)", defaultValue = "none")
    String quantizeKv;

    @Option(names = { "--plugin" }, description = "Explicit plugin/engine to use (e.g. llamacpp, java, bnb)")
    public String pluginId;

    @Option(names = { "--engine", "--gguf-engine" }, description = "GGUF engine mode: auto, java, llamacpp, benchmark")
    String ggufEngine;

    @Option(names = { "--backend" }, description = "GGUF backend to use for local fast path (metal or cpu)")
    String ggufBackend;

    @Option(names = { "--java-native" }, description = "Use the Java-native GGUF loader/probe path")
    boolean javaNativeGguf;

    @Option(names = { "--llamacpp", "--llama-cpp" }, description = "Use the llama.cpp GGUF engine")
    boolean llamaCppGguf;

    @Option(names = { "--benchmark", "--bench" }, description = "Compare Java-native GGUF probe with llama.cpp fallback")
    boolean benchmarkGguf;

    @Override
    public void run() {
        try {
            if (parentCommand != null) {
                parentCommand.bootstrapInheritedEnvironment();
            }

            providerId = normalizeRequestedProvider(providerId);
            boolean providerExplicit = providerId != null && !providerId.isBlank();
            ensureBuiltinProviderRegistration();

            if (tryStandaloneGgufFastPath()) {
                return;
            }

            // Check plugin availability first
            if (!pluginChecker.hasProviders() && !pluginChecker.hasRunnerPlugins()) {
                System.err.println(pluginChecker.getNoPluginsError());
                System.exit(1);
                return;
            }

            // Auto-detect and display kernel platform
            KernelPlatform detectedPlatform;
            try {
                detectedPlatform = sdk.getPlatformInfo();
            } catch (Throwable t) {
                System.err.println("CRITICAL: Platform detection failed: " + t.getMessage());
                return;
            }
            if (CliMetalRuntime.isMetal(detectedPlatform)) {
                ensureMetalRuntimeInitialized();
            }

            if (parentCommand != null && parentCommand.verbose) {
                System.out.println(ChatUIRenderer.CYAN + "Platform: " + detectedPlatform.getDisplayName() + ChatUIRenderer.RESET);
                if (detectedPlatform.isCpu()) {
                    System.out.println(ChatUIRenderer.YELLOW + "⚠️  Running on CPU (GPU acceleration not available)" + ChatUIRenderer.RESET);
                } else {
                    boolean isMetalPlatform = CliMetalRuntime.isMetal(detectedPlatform);
                    if (isMetalPlatform && !isMetalNativeRuntimeActive()) {
                        System.out.println(ChatUIRenderer.YELLOW
                                + "⚠️  Metal selected but native runtime is not active (CPU fallback likely)"
                                + ChatUIRenderer.RESET);
                    } else {
                        System.out.println(ChatUIRenderer.GREEN + "✓ GPU acceleration enabled" + ChatUIRenderer.RESET);
                    }
                }
                printMetalRuntimeStatus();
                System.out.println();
            }

            boolean isMetalPlatform = CliMetalRuntime.isMetal(detectedPlatform);
            if (isMetalPlatform && !isMetalNativeRuntimeActive() && !allowCpuFallbackWhenMetalRequested()) {
                System.err.println("Error: Metal platform selected but native Metal runtime is not active.");
                System.err.println("Refusing CPU fallback for this run so performance behavior stays explicit.");
                System.err.println("Set GOLLEK_ALLOW_CPU_FALLBACK=true to override.");
                System.exit(1);
                return;
            }

            long startTime = System.currentTimeMillis();

            // Check if specific provider is requested but not available
            if (providerExplicit) {
                if (!pluginChecker.hasProvider(providerId)) {
                    System.err.println(pluginChecker.getProviderNotFoundError(providerId));
                    System.exit(1);
                    return;
                }
            }

            boolean customModelPathUsed = false;
            String finalLocalPath = null;
            uiRenderer.setJsonMode(enableJsonSse);
            uiRenderer.printBanner();

            if (isMcpProvider()) {
                System.out.println("MCP provider selected; skipping local model lookup.");
            } else if (modelFile != null && !modelFile.isBlank()) {
                // --- Resolve from --modelFile ---
                Path filePath = Paths.get(modelFile);
                if (!Files.exists(filePath)) {
                    System.err.println("Error: Model file not found: " + modelFile);
                    return;
                }

                if (importModel || copyModel) {
                    var res = sdk.importModel(filePath, importModel);
                    filePath = Paths.get(res.getLocalPath());
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + filePath.toAbsolutePath());
                }

                modelId = filePath.toAbsolutePath().toString();
                finalLocalPath = modelId;
                customModelPathUsed = true;

                if ("native".equals(providerId)) {
                    if (modelFile.endsWith(".litertlm") || modelFile.endsWith(".tflite") || modelFile.endsWith(".task")) {
                        providerId = "litert";
                    }
                }
                System.out.println("Model path: " + filePath.toAbsolutePath());

            } else if (modelDir != null && !modelDir.isBlank()) {
                // --- Resolve from --modelDir ---
                Path dirPath = Paths.get(modelDir);
                if (!Files.isDirectory(dirPath)) {
                    System.err.println("Error: Model directory not found: " + modelDir);
                    return;
                }

                if (importModel || copyModel) {
                    var res = sdk.importModel(dirPath, importModel);
                    dirPath = Paths.get(res.getLocalPath());
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + dirPath.toAbsolutePath());
                }

                modelId = dirPath.toAbsolutePath().toString();
                customModelPathUsed = true;
                if (!providerExplicit) {
                    providerId = "safetensor";
                }
                System.out.println("Model dir: " + dirPath.toAbsolutePath());

            } else if (modelPath != null && !modelPath.isEmpty()) {
                Path customModelPath = Paths.get(modelPath);
                if (!Files.exists(customModelPath)) {
                    System.err.println("Error: Model file not found: " + modelPath);
                    return;
                }
                System.out.println("Using model from: " + customModelPath.toAbsolutePath());
                modelId = customModelPath.toAbsolutePath().toString();
                customModelPathUsed = true;
            }

            if (!customModelPathUsed && modelId != null && !modelId.isBlank()) {
                try {
                    var indexed = LocalModelIndex.find(modelId);
                    if (indexed.isPresent()) {
                        var entry = indexed.get();
                        if (entry.path != null && !entry.path.isBlank() && Files.exists(Path.of(entry.path))) {
                            finalLocalPath = Path.of(entry.path).toAbsolutePath().toString();
                            modelId = finalLocalPath;
                            customModelPathUsed = true;
                            if (!providerExplicit) {
                                String inferred = inferProviderFromIndex(entry);
                                if (inferred != null && !inferred.isBlank()) {
                                    providerId = inferred;
                                }
                            }
                            System.out.println("Resolved local model index entry: " + finalLocalPath);
                        }
                    }
                } catch (Exception ignored) {
                    // Fallback to normal SDK resolution flow.
                }
            }

            if (!customModelPathUsed && (modelFile == null || modelFile.isBlank()) && (modelDir == null || modelDir.isBlank())
                    && (modelPath == null || modelPath.isEmpty())) {
                // Prepare model using SDK (this handles pulling, registration, and conversion)
                try {
                    String quant = ggufQuant != null ? ggufQuant : (ggufOutType != null ? ggufOutType : "Q4_K_M");
                    var resolution = sdk.ensureModelAvailable(modelId, (String) format, pluginId, forceGguf, quant,
                            fallbackModelIds == null ? List.of() : fallbackModelIds,
                            (Consumer<PullProgress>) progress -> {
                        if (progress.getTotal() > 0) {
                            System.out.printf("\r%s %s %3d%% (%d/%d MB)",
                                    ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET,
                                    progress.getProgressBar(20),
                                    progress.getPercentComplete(),
                                    progress.getCompleted() / 1024 / 1024,
                                    progress.getTotal() / 1024 / 1024);
                        } else {
                            System.out.print("\r" + ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET);
                        }
                    });
                    System.out.println();

                    modelId = resolution.getModelId();
                    if (resolution.getLocalPath() != null) {
                        finalLocalPath = resolution.getLocalPath();
                        String displayPath = finalLocalPath;
                        String userHome = System.getProperty("user.home");
                        if (userHome != null && displayPath.startsWith(userHome)) {
                            displayPath = "~" + displayPath.substring(userHome.length());
                        }
                        System.out.println("Model ready at: " + displayPath);
                        customModelPathUsed = true;
                    }

                    if (resolution.getNotice() != null && !resolution.getNotice().isBlank()) {
                        String notice = resolution.getNotice();
                        String compatibilityIssue = detectSafetensorCompatibilityIssue(providerId, finalLocalPath);
                        boolean staleGemma4Notice = notice.contains("Gemma4 multimodal text checkpoints")
                                && compatibilityIssue == null;
                        if (!staleGemma4Notice) {
                            System.out.println(notice);
                        }
                    }
                    
                    if (providerId == null || providerId.isBlank()) {
                        providerId = resolution.getProviderId();
                    }
                    if ((providerId == null || providerId.isBlank()) && resolution.getLocalPath() != null) {
                        providerId = sdk.autoSelectProvider(modelId, forceGguf, quant).orElse(null);
                        if (providerId != null) {
                            sdk.setPreferredProvider(providerId);
                        }
                    }

                } catch (Exception e) {
                    System.err.println("\nError: Failed to prepare model: " + e.getMessage());
                    return;
                }
            }

            tech.kayys.gollek.cli.util.QuantSuggestionDetector.suggestIfNeeded(
                    modelId, finalLocalPath, quantizeStrategy, false);

            ProviderLocalPathResolution providerLocalPathResolution =
                    resolveProviderSpecificLocalPath(providerId, modelId, finalLocalPath);
            if (!providerLocalPathResolution.ok()) {
                return;
            }
            finalLocalPath = providerLocalPathResolution.localPath();
            Gemma3RuntimeSelection gemma3Selection = maybeSelectGemma3AlternateRuntime(providerId, modelId, finalLocalPath);
            if (gemma3Selection != null) {
                providerId = gemma3Selection.provider();
                finalLocalPath = gemma3Selection.localPath();
            }
            if (!validateGemma3ExecutionRoute(providerId, modelId, finalLocalPath)) {
                return;
            }

            boolean libtorchSafetensorCliBridge =
                    "libtorch".equalsIgnoreCase(providerId)
                            && finalLocalPath != null
                            && !finalLocalPath.isBlank()
                            && (isSafetensorCheckpointDir(finalLocalPath) || isSafetensorWeightFile(finalLocalPath));
            String executionProviderId = libtorchSafetensorCliBridge ? "safetensor" : providerId;

            if (direct) {
                providerId = "safetensor";
                sdk.setPreferredProvider("safetensor");
                executionProviderId = "safetensor";
            } else if (executionProviderId != null && !executionProviderId.isEmpty()) {
                sdk.setPreferredProvider(executionProviderId);
            }

            String litertPreflightFailure = detectUnsupportedLiteRtPreflight(executionProviderId, finalLocalPath);
            if (litertPreflightFailure != null) {
                if (!enableJsonSse) {
                    uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                    printQuantizationInfo();
                }
                System.err.println(litertPreflightFailure);
                requestProcessExit();
                return;
            }
            
            if (executionProviderId != null && !ensureProviderHealthy(executionProviderId)) {
                if ("litert".equalsIgnoreCase(executionProviderId)
                        && finalLocalPath != null
                        && !finalLocalPath.isBlank()
                        && tryStandaloneLiteRtExecution(finalLocalPath, startTime)) {
                    return;
                }
                return;
            }
            
            printCompatibilityHintBeforeInference();

            String requestModelTarget = shouldUseLocalModelPath(executionProviderId, finalLocalPath)
                    ? finalLocalPath
                    : modelId;

            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(requestModelTarget)
                    .prompt(prompt)
                    .temperature(temperature)
                    .topP(topP)
                    .topK(topK)
                    .repeatPenalty(repeatPenalty)
                    .jsonMode(jsonMode)
                    .maxTokens(maxTokens)
                    .plugin(pluginId)
                    .streaming(stream);

            if (sessionId != null && !sessionId.isBlank()) {
                requestBuilder.sessionId(sessionId.trim());
            }

            if (customModelPathUsed && finalLocalPath != null) {
                requestBuilder.parameter("model_path", finalLocalPath);
            }

            if (mirostat > 0) {
                requestBuilder.mirostat(mirostat);
            }
            if (grammar != null && !grammar.isEmpty()) {
                requestBuilder.grammar(grammar);
            }

            if (quantizeStrategy != null && !quantizeStrategy.isBlank()) {
                requestBuilder.parameter("quantize_strategy", quantizeStrategy);
                requestBuilder.parameter("quantize_bits", quantizeBits);
            }

            if (quantizeKv != null && !quantizeKv.equalsIgnoreCase("none")) {
                requestBuilder.parameter("kv_cache_quant", quantizeKv);
            }

            if (seed != null) requestBuilder.parameter("seed", seed);
            if (steps != null) requestBuilder.parameter("steps", steps);
            if (format != null) requestBuilder.parameter("format", format);
            if (guidanceScale != null) requestBuilder.parameter("guidance_scale", guidanceScale);
            if (outputPath != null) requestBuilder.parameter("output_path", outputPath);
            if (width != null) requestBuilder.parameter("width", width);
            if (height != null) requestBuilder.parameter("height", height);

            if (extension != null && !extension.isBlank()) {
                requestBuilder.parameter("output_format", extension.toLowerCase());
            }

            String effectiveSystemPrompt = effectiveRunSystemPrompt(executionProviderId, finalLocalPath);
            if (effectiveSystemPrompt != null && !effectiveSystemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(effectiveSystemPrompt));
            }
            requestBuilder.message(Message.user(prompt));

            if (executionProviderId != null && !executionProviderId.isEmpty()) {
                requestBuilder.preferredProvider(executionProviderId);
            }

            if (libtorchSafetensorCliBridge) {
                requestBuilder.metadata("requested_provider", "libtorch");
                requestBuilder.metadata("effective_provider", "safetensor");
                requestBuilder.metadata("provider_bridge_mode", "cli_libtorch_to_safetensor");
                requestBuilder.metadata("provider_bridge_reason", "raw_safetensor_checkpoint");
            }

            requestBuilder.cacheBypass(noCache);

            InferenceRequest request = requestBuilder.build();

            if (!enableJsonSse) {
                uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                printQuantizationInfo();
                if (libtorchSafetensorCliBridge) {
                    forcedExecutionRouteMetadata = bridgeExecutionRouteMetadata();
                    printExecutionRouteInfo(forcedExecutionRouteMetadata);
                } else {
                    forcedExecutionRouteMetadata = Map.of();
                }
            }

            if (shouldUseDirectSafetensorRunPath(providerId, finalLocalPath)) {
                boolean useCliDirectSystemPrompt = (systemPrompt != null && !systemPrompt.isBlank())
                        || isQwenSafetensorModel(executionProviderId, finalLocalPath);
                String explicitDirectSystemPrompt = useCliDirectSystemPrompt
                        ? effectiveSystemPrompt
                        : null;
                InferenceResponse response = runDirectSafetensorCompletion(finalLocalPath, prompt,
                        explicitDirectSystemPrompt);
                printResponse(response, startTime);
                requestProcessExit();
            } else if (stream && shouldUseDirectLiteRtStreamPath(executionProviderId, finalLocalPath)) {
                streamDirectWithProvider("litert", request, startTime);
                requestProcessExit();
            } else if (stream) {
                java.io.ByteArrayOutputStream imageBuffer = new java.io.ByteArrayOutputStream();
                java.io.ByteArrayOutputStream audioBuffer = new java.io.ByteArrayOutputStream();
                CountDownLatch latch = new CountDownLatch(1);
                java.util.concurrent.atomic.AtomicReference<java.util.Map<String, Object>> metricsRef = new java.util.concurrent.atomic.AtomicReference<>();
                java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(0);
                java.util.concurrent.atomic.AtomicLong firstTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
                java.util.concurrent.atomic.AtomicLong lastTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
                long streamStartTime = System.currentTimeMillis();
                
                sdk.streamCompletion(request)
                        .subscribe().with(
                                chunk -> {
                                    if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                        metricsRef.set(chunk.metadata());
                                    }
                                    
                                    if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.IMAGE) {
                                        if (chunk.imageDeltaBase64() != null) {
                                            try {
                                                byte[] decoded = java.util.Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                                imageBuffer.write(decoded);
                                            } catch (Exception ignored) {}
                                        }
                                        return;
                                    }

                                    if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.AUDIO) {
                                        if (chunk.getDelta() != null) {
                                            try {
                                                byte[] decoded = java.util.Base64.getDecoder().decode(chunk.getDelta());
                                                audioBuffer.write(decoded);
                                            } catch (Exception ignored) {}
                                        }
                                        return;
                                    }

                                    String delta = chunk.getDelta();
                                    if (delta != null) {
                                        boolean progressDelta = delta.startsWith("[") && delta.contains("]");
                                        if (!progressDelta && !delta.isEmpty()) {
                                            long now = System.currentTimeMillis();
                                            firstTokenTime.compareAndSet(0, now);
                                            lastTokenTime.set(now);
                                        }
                                        if (progressDelta) {
                                            if (!enableJsonSse) {
                                                System.out.print("\r" + ChatUIRenderer.CYAN + delta + ChatUIRenderer.RESET + "  ");
                                                System.out.flush();
                                            }
                                        } else if (enableJsonSse) {
                                            printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                        } else {
                                            System.out.print(delta);
                                            System.out.flush();
                                        }
                                        tokenCount.incrementAndGet();
                                    }
                                },
                                error -> {
                                    uiRenderer.printError(error.getMessage(), false);
                                    printProviderHintFromError(error);
                                    latch.countDown();
                                },
                                () -> {
                                    long duration = observedStreamDurationMillis(
                                            streamStartTime, System.currentTimeMillis(), lastTokenTime);
                                    handleOutputs(imageBuffer, audioBuffer);
                                    double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                    Double ttftMs = ttftMillis(metricsRef.get(), streamStartTime, firstTokenTime);
                                    Map<String, Object> streamMetrics = observedStreamMetrics(
                                            metricsRef.get(), tokenCount.get(), duration, ttftMs);
                                    if (enableJsonSse) {
                                        printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                    } else {
                                        uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, ttftMs, false);
                                        uiRenderer.printBenchmarks(streamMetrics, false);
                                    }
                                    latch.countDown();
                                });
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                requestProcessExit();
            } else {
                InferenceResponse response = sdk.createCompletion(request);
                printResponse(response, startTime);
                requestProcessExit();
            }

        } catch (Throwable e) {
            System.err.println("\n[FATAL] RunCommand failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private void printMetalRuntimeStatus() {
        var status = CliMetalRuntime.status();
        if (status.errorMessage() != null) {
            System.out.println(ChatUIRenderer.YELLOW + "⚠️  Unable to verify Metal runtime status: " + status.errorMessage() + ChatUIRenderer.RESET);
        } else if (status.active()) {
            System.out.println(ChatUIRenderer.GREEN + "✓ Metal native backend active: " + status.deviceName() + ChatUIRenderer.RESET);
        } else {
            System.out.println(ChatUIRenderer.YELLOW + "⚠️  Metal native backend NOT active (device: " + status.deviceName() + "), using CPU fallback path" + ChatUIRenderer.RESET);
        }
    }

    private boolean isMetalNativeRuntimeActive() {
        return CliMetalRuntime.isNativeActive();
    }

    private void ensureMetalRuntimeInitialized() {
        CliMetalRuntime.initialize();
    }

    private boolean allowCpuFallbackWhenMetalRequested() {
        return CliMetalRuntime.allowCpuFallbackWhenMetalRequested();
    }

    private void printQuantizationInfo() {
        if (quantizeStrategy == null || quantizeStrategy.isBlank()) {
            return;
        }

        String effective = effectiveQuantizationStrategy(quantizeStrategy);
        String effectiveKv = effectiveKvQuantization(quantizeKv);
        System.out.println("Quantization: " + quantizeStrategy.toLowerCase()
                + (effective.equalsIgnoreCase(quantizeStrategy) ? "" : " -> " + effective)
                + " (" + quantizeBits + "-bit)");
        if (quantizeKv != null && !quantizeKv.equalsIgnoreCase("none")) {
            System.out.println("KV cache quantization: " + quantizeKv.toLowerCase()
                    + (effectiveKv.equalsIgnoreCase(quantizeKv) ? "" : " -> " + effectiveKv));
        } else {
            System.out.println("KV cache quantization: off");
        }
        System.out.println("--------------------------------------------------");
    }

    private boolean shouldUseLocalModelPath(String providerId, String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        if (providerId == null || providerId.isBlank()) {
            return false;
        }
        String normalized = providerId.trim().toLowerCase();
        return normalized.equals("safetensor")
                || normalized.equals("gguf")
                || normalized.equals("native")
                || normalized.equals("litert")
                || normalized.equals("onnx")
                || normalized.equals("libtorch");
    }

    private boolean shouldUseDirectSafetensorRunPath(String currentProvider, String localPath) {
        if (!"safetensor".equalsIgnoreCase(currentProvider)) {
            return false;
        }
        Path checkpointPath = resolveSafetensorCheckpointPath(localPath);
        if (checkpointPath == null || !isSafetensorCheckpointDir(checkpointPath.toString())) {
            return false;
        }
        String modelType = readModelType(checkpointPath);
        if (modelType != null && modelType.trim().toLowerCase(Locale.ROOT).startsWith("gemma4")) {
            // Gemma-4 currently regressed badly on the legacy direct one-shot
            // path; route through the provider-backed path instead.
            return false;
        }
        if (sessionId != null && !sessionId.isBlank()) {
            return false;
        }
        if (grammar != null && !grammar.isBlank()) {
            return false;
        }
        return prompt != null && !prompt.isBlank() && directInferenceEngine() != null;
    }

    private String effectiveRunSystemPrompt(String providerId, String localPath) {
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            return systemPrompt;
        }
        if (Boolean.getBoolean(DISABLE_DEFAULT_RUN_SYSTEM_PROPERTY)) {
            return null;
        }
        if (isQwenSafetensorModel(providerId, localPath)) {
            return QWEN_SAFETENSOR_RUN_SYSTEM_PROMPT;
        }
        return DEFAULT_RUN_SYSTEM_PROMPT;
    }

    private boolean isQwenSafetensorModel(String providerId, String localPath) {
        if (!"safetensor".equalsIgnoreCase(providerId) || localPath == null || localPath.isBlank()) {
            return false;
        }
        Path checkpointPath = resolveSafetensorCheckpointPath(localPath);
        if (checkpointPath == null) {
            return false;
        }
        String modelType = readModelType(checkpointPath);
        return modelType != null && modelType.trim().toLowerCase(Locale.ROOT).startsWith("qwen");
    }

    private boolean shouldUseDirectLiteRtStreamPath(String currentProvider, String localPath) {
        if (!"litert".equalsIgnoreCase(currentProvider)) {
            return false;
        }
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        if (!Boolean.getBoolean("gollek.cli.enable_direct_litert_stream")) {
            return false;
        }
        return looksLikeLiteRtArtifactOrDirectory(localPath);
    }

    private boolean looksLikeLiteRtArtifactOrDirectory(String localPath) {
        try {
            Path path = Path.of(localPath).toAbsolutePath().normalize();
            if (Files.isRegularFile(path)) {
                String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
                return isLiteRtFileName(name);
            }
            if (!Files.isDirectory(path)) {
                return false;
            }
            try (var stream = Files.list(path)) {
                return stream
                        .filter(Files::isRegularFile)
                        .map(candidate -> candidate.getFileName().toString().toLowerCase(Locale.ROOT))
                        .anyMatch(this::isLiteRtFileName);
            }
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isLiteRtFileName(String name) {
        return name != null
                && (name.endsWith(".litertlm") || name.endsWith(".task") || name.endsWith(".tflite"));
    }

    private InferenceResponse runDirectSafetensorCompletion(String localPath, String userPrompt,
            String effectiveSystemPrompt) throws Exception {
        Path modelPath = resolveSafetensorCheckpointPath(localPath);
        if (modelPath == null) {
            throw new IllegalArgumentException("Invalid safetensor model path: " + localPath);
        }
        String modelType = readModelType(modelPath);
        float effectiveRepeatPenalty = normalizeDirectRepeatPenalty(modelType, repeatPenalty);
        GenerationConfig.KvCacheQuantization kvQuant = GenerationConfig.KvCacheQuantization.NONE;
        if ("int8".equalsIgnoreCase(quantizeKv)) {
            kvQuant = GenerationConfig.KvCacheQuantization.INT8;
        } else if ("int4".equalsIgnoreCase(quantizeKv) || "turbo".equalsIgnoreCase(quantizeKv)) {
            kvQuant = GenerationConfig.KvCacheQuantization.INT4;
        }

        GenerationConfig config = GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveDirectSamplingStrategy())
                .temperature((float) temperature)
                .topK(topK)
                .topP((float) topP)
                .repetitionPenalty(effectiveRepeatPenalty)
                .kvCacheQuant(kvQuant)
                .seed(seed != null ? seed.longValue() : -1L)
                .build();
        String preparedPrompt = prepareDirectSafetensorPrompt(modelType, effectiveSystemPrompt, userPrompt);
        InferenceResponse response = directInferenceEngine().generate(preparedPrompt, modelPath, config)
                .await()
                .atMost(Duration.ofMinutes(5));
        return sanitizeDirectSafetensorResponse(response, modelType);
    }

    private Path resolveSafetensorCheckpointPath(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return null;
        }
        try {
            Path path = Path.of(localPath).toAbsolutePath().normalize();
            if (Files.isDirectory(path)) {
                return path;
            }
            if (!Files.isRegularFile(path)) {
                return null;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            if (!name.endsWith(".safetensors")) {
                return null;
            }
            Path parent = path.getParent();
            return parent != null && Files.isDirectory(parent) ? parent : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private float normalizeDirectRepeatPenalty(String modelType, double requestedRepeatPenalty) {
        if (modelType == null || modelType.isBlank()) {
            return (float) requestedRepeatPenalty;
        }
        String normalized = modelType.trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("gemma4") && Math.abs(requestedRepeatPenalty - 1.1d) < 1.0e-6) {
            return 1.0f;
        }
        return (float) requestedRepeatPenalty;
    }

    private GenerationConfig.SamplingStrategy resolveDirectSamplingStrategy() {
        if (temperature < 1.0e-4f || topK == 1) {
            return GenerationConfig.SamplingStrategy.GREEDY;
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0 && topP < 1.0;
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

    private DirectInferenceEngine directInferenceEngine() {
        if (directInferenceEngine != null) {
            return directInferenceEngine;
        }
        try {
            var instance = Arc.container().instance(DirectInferenceEngine.class);
            if (instance != null && instance.isAvailable()) {
                return instance.get();
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private boolean isSafetensorCheckpointDir(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.isDirectory(path)) {
                return false;
            }
            if (!Files.exists(path.resolve("config.json"))) {
                return false;
            }
            return Files.exists(path.resolve("model.safetensors"))
                    || Files.exists(path.resolve("model.safetensor"))
                    || Files.exists(path.resolve("model.safetensors.index.json"));
        } catch (Exception ignored) {
            return false;
        }
    }

    private String detectUnsupportedLiteRtPreflight(String provider, String localPath) {
        if (!"litert".equalsIgnoreCase(provider) || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (Boolean.getBoolean("gollek.litert.enable_experimental_gemma4_task_runner")
                || Boolean.getBoolean("gollek.litert.enable_experimental_raw_litertlm")) {
            return null;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.exists(path) || !looksLikeGemma4LiteRtArtifact(path)) {
                return null;
            }
            if (hasNativeGemmaLiteRtLm(path)) {
                return null;
            }
            return "Gemma 4 LiteRT runner is disabled by default because the current Java/Metal paths are slow "
                    + "and produce incorrect repeated-token output for this export. Set -D"
                    + "gollek.litert.enable_experimental_gemma4_task_runner=true only for diagnostics.";
        } catch (Exception ignored) {
            return null;
        }
    }

    private boolean looksLikeGemma4LiteRtArtifact(Path path) {
        String lower = path.toString().toLowerCase(Locale.ROOT);
        if (Files.isRegularFile(path)) {
            if (!lower.contains("gemma-4") && !lower.contains("gemma4")) {
                return false;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            return isLiteRtModelFileName(name);
        }
        if (!Files.isDirectory(path)) {
            return false;
        }
        try (var stream = Files.list(path)) {
            return stream.anyMatch(candidate -> {
                String name = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
                return (name.contains("gemma-4") || name.contains("gemma4"))
                        && isLiteRtModelFileName(name);
            });
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean hasNativeGemmaLiteRtLm(Path path) {
        try {
            if (Files.isRegularFile(path)) {
                String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
                return name.endsWith(".litertlm")
                        && (name.contains("gemma-4") || name.contains("gemma4"));
            }
            if (!Files.isDirectory(path)) {
                return false;
            }
            try (var stream = Files.list(path)) {
                return stream.anyMatch(candidate -> {
                    String name = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
                    return Files.isRegularFile(candidate)
                            && name.endsWith(".litertlm")
                            && (name.contains("gemma-4") || name.contains("gemma4"))
                            && !name.contains("qualcomm");
                });
            }
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean isLiteRtModelFileName(String name) {
        return name.endsWith(".litertlm")
                || name.endsWith(".task")
                || name.endsWith(".tflite")
                || name.endsWith(".tfl");
    }

    private String prepareDirectSafetensorPrompt(String modelType, String systemPrompt, String userPrompt) {
        if (userPrompt == null || userPrompt.isBlank()) {
            return userPrompt;
        }
        if (looksLikePreformattedPrompt(userPrompt)) {
            return userPrompt;
        }
        if (!shouldFormatDirectPrompt(modelType)) {
            return userPrompt;
        }
        try {
            if (systemPrompt != null && !systemPrompt.isBlank()) {
                return ChatTemplateFormatter.format(List.of(Message.system(systemPrompt), Message.user(userPrompt)),
                        modelType);
            }
            return ChatTemplateFormatter.format(List.of(Message.user(userPrompt)), modelType);
        } catch (Exception ignored) {
            return userPrompt;
        }
    }

    private boolean shouldFormatDirectPrompt(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String normalized = modelType.trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("gemma4")) {
            // The official Gemma-4 turn template is now supported without
            // double-BOS injection, but the one-shot local direct path still
            // produces worse first-token quality for this checkpoint than the
            // historical raw-prompt path. Keep raw prompt here until the
            // remaining Gemma-4 forward/prefill semantics are fixed.
            return false;
        }
        return normalized.startsWith("gemma")
                || normalized.startsWith("llama")
                || normalized.startsWith("mistral")
                || normalized.startsWith("mixtral")
                || normalized.startsWith("phi")
                || normalized.startsWith("qwen");
    }

    private boolean looksLikePreformattedPrompt(String prompt) {
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.stripLeading();
        if (trimmed.startsWith("<bos>")) {
            trimmed = trimmed.substring("<bos>".length()).stripLeading();
        }
        return trimmed.startsWith("<|turn>")
                || trimmed.startsWith("<start_of_turn>")
                || trimmed.startsWith("<|im_start|>")
                || trimmed.startsWith("<|begin_of_text|>")
                || trimmed.startsWith("[INST]");
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
        } catch (Exception ignored) {
            return "";
        }
    }

    private InferenceResponse sanitizeDirectSafetensorResponse(InferenceResponse response, String modelType) {
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
        content = content.replace("<channel|>", "");
        content = content.replace("<turn|>", "");
        content = content.replace("<|tool_response>", "");

        if (content.equals(response.getContent())) {
            return response;
        }
        return response.toBuilder().content(content).build();
    }

    private String detectSafetensorCompatibilityIssue(String providerId, String localPath) {
        if (!"safetensor".equalsIgnoreCase(providerId) || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (!isSafetensorCheckpointDir(localPath)) {
            return null;
        }

        try {
            Path configPath = Path.of(localPath).resolve("config.json");
            ModelConfig parsed = ModelConfig.load(configPath, new ObjectMapper());
            String modelType = parsed.modelType() == null ? "" : parsed.modelType().toLowerCase(Locale.ROOT);
            boolean resolvedGemma4Text = modelType.startsWith("gemma4")
                    && parsed.hiddenSize() > 0
                    && parsed.numHiddenLayers() > 0
                    && parsed.numAttentionHeads() > 0;
            if (resolvedGemma4Text) {
                return null;
            }
        } catch (Exception ignored) {
            // Fall back to raw config heuristics only when structured parsing cannot
            // reconcile the checkpoint into a supported text model shape.
        }

        try {
            Path configPath = Path.of(localPath).resolve("config.json");
            String config = Files.readString(configPath);
            boolean gemma4Conditional = config.contains("\"Gemma4ForConditionalGeneration\"")
                    || config.contains("\"model_type\": \"gemma4\"");
            boolean multimodalWrapper = config.contains("\"vision_config\"")
                    || config.contains("\"audio_config\"");
            boolean hasTextConfig = config.contains("\"text_config\"");

            if (gemma4Conditional && multimodalWrapper && !hasTextConfig) {
                return "this local safetensor runtime build does not reliably support Gemma4 multimodal text checkpoints like "
                        + Path.of(localPath).getFileName()
                        + ". It may run but produce incorrect output. Use a newer safetensor engine build or convert the model to a supported runtime first.";
            }
        } catch (Exception ignored) {
            return null;
        }

        return null;
    }

    private String detectNativeGgufCompatibilityIssue(String providerId, String modelId, String localPath) {
        if (!"native".equalsIgnoreCase(providerId) || !isGgufModelPath(localPath)) {
            return null;
        }

        String fingerprint = ((modelId == null ? "" : modelId) + " "
                + (localPath == null ? "" : Path.of(localPath).getFileName())).toLowerCase(java.util.Locale.ROOT);
        if (!fingerprint.contains("gemma4") && !fingerprint.contains("gemma-4")) {
            return null;
        }

        return "the local native GGUF runtime in this build does not reliably support Gemma 4 text checkpoints like "
                + Path.of(localPath).getFileName()
                + ". It can tokenize the prompt but still ignore Gemma 4 attention/KV/RoPE semantics and produce incorrect output. "
                + "Use a newer Gemma 4-capable runtime instead of the local GGUF/native fallback.";
    }

    private boolean isGgufModelPath(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            return Files.isRegularFile(path)
                    && path.getFileName().toString().toLowerCase(java.util.Locale.ROOT).endsWith(".gguf");
        } catch (Exception ignored) {
            return false;
        }
    }

    private String effectiveQuantizationStrategy(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "turbo" -> "bnb";
            case "gptq", "autoround" -> "int4";
            default -> normalized;
        };
    }

    private String effectiveKvQuantization(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "", "none" -> "off";
            case "int8" -> "int8";
            case "int4" -> "int4";
            case "turbo" -> "int4";
            default -> normalized;
        };
    }

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    boolean shouldTryStandaloneGgufFastPath() {
        if (prompt == null || prompt.isBlank()) {
            return false;
        }
        if (providerId != null && !providerId.isBlank() && !isGgufProviderAlias(providerId)) {
            return false;
        }
        return isGgufProviderAlias(providerId)
                || isGgufPluginAlias(pluginId)
                || hasExplicitGgufEngine()
                || forceGguf
                || isGgufFormat(format)
                || looksLikeGgufPath(modelFile)
                || looksLikeGgufPath(modelPath)
                || looksLikeGgufPath(modelId);
    }

    String[] buildStandaloneGgufFastRunArgs() {
        List<String> args = new java.util.ArrayList<>();
        args.add("run");
        if (modelFile != null && !modelFile.isBlank()) {
            args.add("--modelFile");
            args.add(modelFile);
        } else if (modelPath != null && !modelPath.isBlank()) {
            args.add("--model-path");
            args.add(modelPath);
        } else if (modelId != null && !modelId.isBlank()) {
            args.add("--model");
            args.add(modelId);
        }
        if (prompt != null) {
            args.add("--prompt");
            args.add(prompt);
        }
        args.add("--max-tokens");
        args.add(Integer.toString(maxTokens));
        args.add("--temperature");
        args.add(Double.toString(temperature));
        args.add("--top-k");
        args.add(Integer.toString(topK));
        args.add("--top-p");
        args.add(Double.toString(topP));

        String engine = requestedGgufEngine();
        if (engine != null && !engine.isBlank()) {
            args.add("--engine");
            args.add(engine);
        }
        String backend = requestedGgufBackend();
        if (backend != null && !backend.isBlank()) {
            args.add("--backend");
            args.add(backend);
        }
        if (providerId != null && !providerId.isBlank()) {
            args.add("--provider");
            args.add(providerId);
        } else {
            args.add("--provider");
            args.add("gguf");
        }
        return args.toArray(String[]::new);
    }

    private boolean tryStandaloneGgufFastPath() {
        if (!shouldTryStandaloneGgufFastPath()) {
            return false;
        }
        int status = GgufFastRun.run(buildStandaloneGgufFastRunArgs());
        if (GgufFastRun.isFallbackToFullCliStatus(status)) {
            return false;
        }
        if (status == 0) {
            if (GgufFastRun.hardExitAfterRun()) {
                GgufFastRun.hardExitProcess(0);
            }
            requestProcessExit();
            return true;
        }
        System.exit(status);
        return true;
    }

    private boolean hasExplicitGgufEngine() {
        return javaNativeGguf
                || llamaCppGguf
                || benchmarkGguf
                || (ggufEngine != null && !ggufEngine.isBlank());
    }

    private String requestedGgufEngine() {
        if (benchmarkGguf) {
            return "benchmark";
        }
        if (javaNativeGguf) {
            return "java";
        }
        if (llamaCppGguf) {
            return "llamacpp";
        }
        return ggufEngine == null ? null : ggufEngine.trim();
    }

    private String requestedGgufBackend() {
        if (ggufBackend != null && !ggufBackend.isBlank()) {
            return ggufBackend.trim();
        }
        if (parentCommand != null && parentCommand.isUseCpu()) {
            return "cpu";
        }
        if (parentCommand != null && parentCommand.platform() != null && !parentCommand.platform().isBlank()) {
            return parentCommand.platform().trim();
        }
        return null;
    }

    private boolean isGgufProviderAlias(String provider) {
        if (provider == null || provider.isBlank()) {
            return false;
        }
        String normalized = provider.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("gguf")
                || normalized.equals("llamacpp")
                || normalized.equals("llama.cpp")
                || normalized.equals("llama-cpp")
                || normalized.equals("java")
                || normalized.equals("java-native")
                || normalized.equals("jvm");
    }

    private boolean isGgufPluginAlias(String plugin) {
        if (plugin == null || plugin.isBlank()) {
            return false;
        }
        String normalized = plugin.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("gguf")
                || normalized.equals("gguf-runner")
                || normalized.equals("llamacpp")
                || normalized.equals("llama.cpp")
                || normalized.equals("llama-cpp");
    }

    private boolean isGgufFormat(String value) {
        return value != null && value.trim().equalsIgnoreCase("gguf");
    }

    private boolean looksLikeGgufPath(String value) {
        return value != null && value.trim().toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private String normalizeRequestedProvider(String rawProviderId) {
        if (rawProviderId == null) {
            return null;
        }
        String trimmed = rawProviderId.trim();
        if (trimmed.isEmpty()) {
            return trimmed;
        }
        return switch (trimmed.toLowerCase(Locale.ROOT)) {
            case "torch", "torchscript", "pytorch" -> "libtorch";
            default -> trimmed.toLowerCase(Locale.ROOT);
        };
    }

    private String inferProviderFromIndex(LocalModelIndex.Entry entry) {
        if (entry == null || entry.format == null) {
            return null;
        }
        return providerForFormat(entry.format);
    }

    private String providerForFormat(String format) {
        if (format == null) {
            return null;
        }
        String normalized = format.trim().toLowerCase(java.util.Locale.ROOT);
        return switch (normalized) {
            case "gguf" -> "gguf";
            case "safetensors", "safetensor" -> "safetensor";
            case "litert", "task", "tflite" -> "litert";
            case "onnx" -> "onnx";
            case "torchscript", "pytorch" -> "libtorch";
            default -> null;
        };
    }

    private record Gemma3RuntimeSelection(String provider, String localPath) {
    }

    private Gemma3RuntimeSelection maybeSelectGemma3AlternateRuntime(String currentProvider, String requestedModelId, String localPath) {
        if (currentProvider == null || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider)) {
            return null;
        }
        try {
            Path path = Path.of(localPath);
            Path modelDir = Files.isDirectory(path) ? path : path.getParent();
            if (modelDir == null || !Files.isDirectory(modelDir)) {
                return null;
            }
            String modelType = readModelType(modelDir);
            if (modelType == null || !modelType.trim().toLowerCase(Locale.ROOT).startsWith("gemma3")) {
                return null;
            }

            Optional<Path> litert = findPreferredAlternateArtifact(modelDir, requestedModelId, ".litertlm", ".tflite", ".task");
            if (litert.isPresent() && isProviderActive("litert")) {
                Path selected = litert.get().toAbsolutePath().normalize();
                System.out.println("Detected Gemma3 safetensor checkpoint; switching to LiteRT runtime artifact: " + selected);
                return new Gemma3RuntimeSelection("litert", selected.toString());
            }

            Optional<Path> gguf = findPreferredAlternateArtifact(modelDir, requestedModelId, ".gguf");
            if (gguf.isPresent() && isProviderActive("gguf")) {
                Path selected = gguf.get().toAbsolutePath().normalize();
                System.out.println("Detected Gemma3 safetensor checkpoint; switching to GGUF runtime artifact: " + selected);
                return new Gemma3RuntimeSelection("gguf", selected.toString());
            }

            if (litert.isPresent() || gguf.isPresent()) {
                System.out.println("Detected Gemma3 alternate runtime artifact, but provider is not currently available in this build."
                        + " Available providers are shown by: gollek extensions");
            }
        } catch (Exception ignored) {
            // Keep original route on any failure.
        }
        return null;
    }

    private Optional<Path> findPreferredAlternateArtifact(Path dir, String requestedModelId, String... suffixes) {
        if (dir == null || suffixes == null || suffixes.length == 0) {
            return Optional.empty();
        }
        String modelHint = normalizeArtifactHint(requestedModelId);
        try (var stream = Files.walk(dir, 2)) {
            Optional<Path> preferred = stream.filter(Files::isRegularFile)
                    .filter(p -> hasAnySuffix(p, suffixes))
                    .filter(p -> isCompatibleAlternateArtifact(p, modelHint))
                    .findFirst();
            if (preferred.isPresent()) {
                return preferred;
            }
            // Keep legacy fallback only when there is no model-specific hint.
            if (modelHint == null || modelHint.isBlank()) {
                return findFirstRegularFile(dir, suffixes);
            }
            return Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private boolean hasAnySuffix(Path path, String... suffixes) {
        if (path == null || suffixes == null || suffixes.length == 0) {
            return false;
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String suffix : suffixes) {
            if (suffix != null && name.endsWith(suffix.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private boolean isCompatibleAlternateArtifact(Path candidate, String modelHint) {
        String filename = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        if (filename.contains("tiny_garden")) {
            return false;
        }
        if (modelHint == null || modelHint.isBlank()) {
            return true;
        }
        String normalizedName = normalizeArtifactHint(filename);
        if (normalizedName.contains(modelHint)) {
            return true;
        }
        // functiongemma checkpoint must not auto-route to unrelated tiny artifacts.
        if (modelHint.contains("functiongemma")) {
            return normalizedName.contains("functiongemma");
        }
        return true;
    }

    private String normalizeArtifactHint(String value) {
        if (value == null) {
            return "";
        }
        return value.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "");
    }

    private Optional<Path> findFirstRegularFile(Path dir, String... suffixes) {
        if (dir == null || suffixes == null || suffixes.length == 0) {
            return Optional.empty();
        }
        try (var stream = Files.walk(dir, 2)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        for (String suffix : suffixes) {
                            if (suffix != null && name.endsWith(suffix.toLowerCase(Locale.ROOT))) {
                                return true;
                            }
                        }
                        return false;
                    })
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private boolean isProviderActive(String provider) {
        if (provider == null || provider.isBlank()) {
            return false;
        }
        try {
            List<ProviderInfo> providers = sdk.listAvailableProviders();
            if (providers == null || providers.isEmpty()) {
                return false;
            }
            String normalized = provider.trim().toLowerCase(Locale.ROOT);
            return providers.stream()
                    .map(ProviderInfo::id)
                    .filter(Objects::nonNull)
                    .map(id -> id.trim().toLowerCase(Locale.ROOT))
                    .anyMatch(normalized::equals);
        } catch (Exception ignored) {
            return false;
        }
    }

    private void ensureBuiltinProviderRegistration() {
        ensureProviderRegistered(
                "litert",
                "tech.kayys.gollek.provider.litert.LiteRTProvider");
    }

    private void ensureProviderRegistered(String providerId, String className) {
        try {
            if (providerRegistry.hasProvider(providerId)) {
                return;
            }
            Class<?> clazz = Class.forName(className);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            if (instance instanceof LLMProvider provider) {
                providerRegistry.register(provider);
            }
        } catch (Throwable t) {
            System.err.printf("Provider bootstrap failed for %s (%s): %s%n", providerId, className, t.getMessage());
        }
    }

    private String extractModelPath(ModelInfo modelInfo) {
        if (modelInfo == null || modelInfo.getMetadata() == null) {
            return null;
        }
        Object value = modelInfo.getMetadata().get("path");
        return value != null ? value.toString() : null;
    }

    private ProviderLocalPathResolution resolveProviderSpecificLocalPath(
            String currentProvider,
            String requestedModelId,
            String localPath) {
        if (localPath == null || localPath.isBlank() || currentProvider == null || currentProvider.isBlank()) {
            return new ProviderLocalPathResolution(true, localPath);
        }

        String normalizedProvider = normalizeRequestedProvider(currentProvider);
        if (!"libtorch".equals(normalizedProvider)) {
            return new ProviderLocalPathResolution(true, localPath);
        }

        try {
            Path originalPath = Path.of(localPath).toAbsolutePath().normalize();
            Optional<Path> directArtifact = resolveDirectLibtorchArtifact(originalPath);
            if (directArtifact.isPresent()) {
                Path resolved = directArtifact.get().toAbsolutePath().normalize();
                if (!resolved.equals(originalPath)) {
                    System.out.println("Provider 'libtorch' selected; using local TorchScript artifact: " + resolved);
                }
                return new ProviderLocalPathResolution(true, resolved.toString());
            }
        } catch (Exception ignored) {
            // Fall through to registry-based lookup and explicit compatibility hint.
        }

        Optional<ModelInfo> alternative = findLocalArtifactForProvider(requestedModelId, localPath, normalizedProvider);
        if (alternative.isPresent()) {
            String candidatePath = extractModelPath(alternative.get());
            if (candidatePath != null && !candidatePath.isBlank()) {
                try {
                    Optional<Path> resolved = resolveDirectLibtorchArtifact(Path.of(candidatePath));
                    if (resolved.isPresent()) {
                        Path artifactPath = resolved.get().toAbsolutePath().normalize();
                        System.out.println("Provider 'libtorch' selected; rerouting to registered TorchScript artifact: "
                                + artifactPath);
                        return new ProviderLocalPathResolution(true, artifactPath.toString());
                    }
                } catch (Exception ignored) {
                    // Keep the explicit error below if the candidate is not actually runnable.
                }
            }
        }

        if (isSafetensorCheckpointDir(localPath) || isSafetensorWeightFile(localPath)) {
            System.out.println("Provider 'libtorch' selected; using safetensor bridge mode for this checkpoint.");
            return new ProviderLocalPathResolution(true, localPath);
        }

        System.err.println("Error: Provider 'libtorch' currently needs a runnable TorchScript-style artifact "
                + "(.pt, .pts, .pth, .bin).");
        System.err.println("Resolved local path: " + localPath);
        return new ProviderLocalPathResolution(false, localPath);
    }

    private Optional<ModelInfo> findLocalArtifactForProvider(
            String requestedModelId,
            String localPath,
            String currentProvider) {
        if (!"libtorch".equalsIgnoreCase(currentProvider)) {
            return Optional.empty();
        }
        try {
            List<ModelInfo> models = sdk.listModels();
            if (models.isEmpty()) {
                return Optional.empty();
            }

            String canonicalModelId = models.stream()
                    .filter(model -> requestedModelId != null && (requestedModelId.equalsIgnoreCase(model.getModelId())
                            || requestedModelId.equalsIgnoreCase(model.getShortId())))
                    .map(ModelInfo::getModelId)
                    .findFirst()
                    .orElseGet(() -> models.stream()
                            .filter(model -> {
                                String candidatePath = extractModelPath(model);
                                return candidatePath != null && candidatePath.equals(localPath);
                            })
                            .map(ModelInfo::getModelId)
                            .findFirst()
                            .orElse(requestedModelId));

            return models.stream()
                    .filter(model -> canonicalModelId != null && canonicalModelId.equalsIgnoreCase(model.getModelId()))
                    .filter(model -> {
                        String candidatePath = extractModelPath(model);
                        if (candidatePath == null || candidatePath.equals(localPath)) {
                            return false;
                        }
                        try {
                            return resolveDirectLibtorchArtifact(Path.of(candidatePath)).isPresent();
                        } catch (Exception ignored) {
                            return false;
                        }
                    })
                    .sorted((left, right) -> Integer.compare(
                            libtorchArtifactPriority(extractModelPath(right), right.getFormat()),
                            libtorchArtifactPriority(extractModelPath(left), left.getFormat())))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Path> resolveDirectLibtorchArtifact(Path root) {
        if (root == null || !Files.exists(root)) {
            return Optional.empty();
        }
        if (Files.isRegularFile(root) && isLibtorchModelFile(root)) {
            return Optional.of(root);
        }
        if (!Files.isDirectory(root)) {
            return Optional.empty();
        }
        try (var paths = Files.walk(root, 3)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(this::isLibtorchModelFile)
                    .sorted((left, right) -> Integer.compare(libtorchArtifactPriority(right, null),
                            libtorchArtifactPriority(left, null)))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private boolean isLibtorchModelFile(Path path) {
        if (path == null) {
            return false;
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".pt")
                || name.endsWith(".pts")
                || name.endsWith(".pth")
                || name.endsWith(".bin");
    }

    private int libtorchArtifactPriority(Path path, String format) {
        if (path == null) {
            return 0;
        }
        return libtorchArtifactPriority(path.toString(), format);
    }

    private int libtorchArtifactPriority(String path, String format) {
        int score = 0;
        String normalizedPath = path == null ? "" : path.toLowerCase(Locale.ROOT);
        String normalizedFormat = format == null ? "" : format.toLowerCase(Locale.ROOT);
        if (normalizedFormat.equals("torchscript")) {
            score += 50;
        } else if (normalizedFormat.equals("pytorch")) {
            score += 20;
        }
        if (normalizedPath.contains("/libtorchscript/")) {
            score += 40;
        }
        if (normalizedPath.endsWith(".pt")) {
            score += 30;
        } else if (normalizedPath.endsWith(".pts")) {
            score += 25;
        } else if (normalizedPath.endsWith(".pth")) {
            score += 15;
        } else if (normalizedPath.endsWith(".bin")) {
            score += 10;
        }
        return score;
    }

    private boolean isSafetensorWeightFile(String localPath) {
        if (localPath == null || localPath.isBlank()) {
            return false;
        }
        try {
            Path path = Path.of(localPath);
            if (!Files.isRegularFile(path)) {
                return false;
            }
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            return name.endsWith(".safetensors") || name.endsWith(".safetensor");
        } catch (Exception ignored) {
            return false;
        }
    }

    private void printResponse(InferenceResponse response, long startTime) {
        Map<String, Object> metadata = effectiveExecutionRouteMetadata(response.getMetadata());
        printExecutionRouteInfo(metadata);
        printQuantCacheInfo(metadata);
        printKvCacheQuantizationInfo(metadata);
        System.out.println();
        System.out.println(ChatUIRenderer.GREEN + response.getContent() + ChatUIRenderer.RESET);
        if (response.getSessionId() != null && !response.getSessionId().isBlank()) {
            System.out.println(ChatUIRenderer.DIM + "Session: " + response.getSessionId() + ChatUIRenderer.RESET);
        }
        
        if (metadata.containsKey("audio")) {
            saveAudio(metadata.get("audio").toString());
        } else if (metadata.containsKey("image")) {
            saveImage(metadata.get("image").toString());
        }

        double seconds = Math.max(response.getDurationMs() / 1000.0, 0.001);
        double tps = response.getTokensUsed() / seconds;
        uiRenderer.printStats(response.getTokensUsed(), response.getDurationMs() / 1000.0, tps,
                ttftMillis(metadata), false);
        uiRenderer.printBenchmarks(metadata, false);
    }

    private static Double ttftMillis(Map<String, Object> metadata) {
        return metadataDouble(metadata, "bench.ttft_ms");
    }

    private static Double ttftMillis(
            Map<String, Object> metadata,
            long streamStartTimeMs,
            java.util.concurrent.atomic.AtomicLong firstTokenTimeMs) {
        Double metadataTtft = ttftMillis(metadata);
        if (metadataTtft != null) {
            return metadataTtft;
        }
        long first = firstTokenTimeMs != null ? firstTokenTimeMs.get() : 0L;
        if (first <= 0L || first < streamStartTimeMs) {
            return null;
        }
        return (double) (first - streamStartTimeMs);
    }

    private static Map<String, Object> observedStreamMetrics(
            Map<String, Object> metadata,
            int outputTokens,
            long durationMs,
            Double ttftMs) {
        Map<String, Object> metrics = metadata == null || metadata.isEmpty()
                ? new LinkedHashMap<>()
                : new LinkedHashMap<>(metadata);
        if (outputTokens > 0) {
            metrics.putIfAbsent("tokens.output", outputTokens);
            double durationSeconds = Math.max(durationMs / 1000.0, 0.001);
            metrics.putIfAbsent("bench.generation_tps", outputTokens / durationSeconds);
            if (ttftMs != null) {
                metrics.putIfAbsent("bench.ttft_ms", ttftMs);
                if (outputTokens > 1 && durationMs > ttftMs) {
                    metrics.putIfAbsent("bench.tpot_ms", (durationMs - ttftMs) / (outputTokens - 1));
                }
            }
        } else if (ttftMs != null) {
            metrics.putIfAbsent("bench.ttft_ms", ttftMs);
        }
        return metrics;
    }

    private static long observedStreamDurationMillis(
            long streamStartTimeMs,
            long completionTimeMs,
            java.util.concurrent.atomic.AtomicLong lastTokenTimeMs) {
        long last = lastTokenTimeMs != null ? lastTokenTimeMs.get() : 0L;
        if (last >= streamStartTimeMs) {
            return Math.max(1L, last - streamStartTimeMs);
        }
        return Math.max(1L, completionTimeMs - streamStartTimeMs);
    }

    private static Double metadataDouble(Map<String, Object> metadata, String key) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        Object value = metadata.get(key);
        return value instanceof Number number ? number.doubleValue() : null;
    }

    private void printExecutionRouteInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        Object requested = metadata.get("requested_provider");
        Object effective = metadata.get("effective_provider");
        Object backend = metadata.get("execution_backend");
        Object bridgeMode = metadata.get("provider_bridge_mode");
        Object bridgeReason = metadata.get("provider_bridge_reason");

        if (requested == null && effective == null && backend == null && bridgeMode == null) {
            return;
        }

        StringBuilder line = new StringBuilder("Execution route: ");
        if (requested != null && effective != null
                && !String.valueOf(requested).equalsIgnoreCase(String.valueOf(effective))) {
            line.append(requested).append(" -> ").append(effective);
        } else if (effective != null) {
            line.append(effective);
        } else if (requested != null) {
            line.append(requested);
        } else {
            line.append(backend);
        }
        if (backend != null) {
            line.append(" (backend=").append(backend).append(")");
        }
        if (bridgeMode != null) {
            line.append(" [").append(bridgeMode);
            if (bridgeReason != null) {
                line.append(": ").append(bridgeReason);
            }
            line.append("]");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private Map<String, Object> effectiveExecutionRouteMetadata(Map<String, Object> responseMetadata) {
        if (responseMetadata == null || responseMetadata.isEmpty()) {
            return forcedExecutionRouteMetadata;
        }
        if (forcedExecutionRouteMetadata == null || forcedExecutionRouteMetadata.isEmpty()) {
            return responseMetadata;
        }
        if (responseMetadata.containsKey("requested_provider")
                || responseMetadata.containsKey("effective_provider")
                || responseMetadata.containsKey("provider_bridge_mode")) {
            return responseMetadata;
        }
        Map<String, Object> merged = new LinkedHashMap<>(forcedExecutionRouteMetadata);
        merged.putAll(responseMetadata);
        return merged;
    }

    private Map<String, Object> bridgeExecutionRouteMetadata() {
        Map<String, Object> route = new LinkedHashMap<>();
        route.put("requested_provider", "libtorch");
        route.put("effective_provider", "safetensor");
        route.put("provider_bridge_mode", "cli_libtorch_to_safetensor");
        route.put("provider_bridge_reason", "raw_safetensor_checkpoint");
        return route;
    }

    private void printQuantCacheInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("quant_cache_state");
        if (state == null) {
            return;
        }
        StringBuilder line = new StringBuilder("Quant cache: ").append(state);
        Object path = metadata.get("quant_cache_path");
        if (path != null) {
            line.append(" (").append(path).append(")");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private void printKvCacheQuantizationInfo(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("kv_cache_quantization");
        if (state == null) {
            return;
        }
        System.out.println("Effective KV cache quantization: " + state);
        System.out.println("--------------------------------------------------");
    }

    private void handleOutputs(java.io.ByteArrayOutputStream imageBuffer, java.io.ByteArrayOutputStream audioBuffer) {
        if (imageBuffer.size() > 0) saveBuffer(imageBuffer, "png", "Image");
        if (audioBuffer.size() > 0) saveBuffer(audioBuffer, "flac", "Audio");
    }

    private void saveBuffer(java.io.ByteArrayOutputStream buffer, String defaultExt, String label) {
        try {
            String ext = (extension != null && !extension.isEmpty()) ? extension : defaultExt;
            if (ext.startsWith(".")) ext = ext.substring(1);
            Path outPath = (outputPath != null && !outputPath.isEmpty()) ? Path.of(outputPath) : Path.of("output." + ext);
            if (extension != null) outPath = replaceExtension(outPath, ext);
            Files.write(outPath, buffer.toByteArray());
            System.out.println("\n" + ChatUIRenderer.GREEN + ChatUIRenderer.BOLD + "✓ " + label + " saved to: " + ChatUIRenderer.RESET + outPath.toAbsolutePath());
            if ("png".equals(ext)) autoOpenImage(outPath); else autoOpenAudio(outPath);
        } catch (Exception e) {
            System.err.println("\nFailed to save " + label.toLowerCase() + ": " + e.getMessage());
        }
    }

    private void saveAudio(String base64) {
        saveBufferFromBase64(base64, "flac", "Audio");
    }

    private void saveImage(String base64) {
        saveBufferFromBase64(base64, "png", "Image");
    }

    private void saveBufferFromBase64(String base64, String defaultExt, String label) {
        try {
            byte[] decoded = java.util.Base64.getDecoder().decode(base64);
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            bos.write(decoded);
            saveBuffer(bos, defaultExt, label);
        } catch (Exception ignored) {}
    }

    private Path replaceExtension(Path path, String newExt) {
        String filename = path.getFileName().toString();
        int lastDot = filename.lastIndexOf('.');
        if (lastDot == -1) return path.resolveSibling(filename + "." + newExt);
        return path.resolveSibling(filename.substring(0, lastDot) + "." + newExt);
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = String.format("{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                escapeJson(id), created, escapeJson(model != null ? model : ""), escapeJson(delta));
        System.out.println("data: " + payload);
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = String.format("{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                escapeJson(id), created, escapeJson(model != null ? model : ""));
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
    }

    private String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private InferenceResponse inferDirectWithProvider(String id, InferenceRequest request) {
        ProviderRequest providerRequest = buildDirectProviderRequest(request, false);
        try {
            Optional<LLMProvider> providerOpt = providerRegistry.getProvider(id);
            if (providerOpt.isEmpty()) throw new RuntimeException("Provider not available: " + id);
            return providerOpt.get().infer(providerRequest).await().atMost(Duration.ofSeconds(300));
        } catch (RuntimeException primary) {
            String providerModel = providerRequest.getModel();
            if (providerModel != null && (providerModel.toLowerCase().endsWith(".safetensors") || providerModel.toLowerCase().endsWith(".bin"))) {
                try {
                    System.out.println("Primary load failed; falling back to libtorch...");
                    return providerRegistry.getProvider("libtorch").get().infer(providerRequest).await().atMost(Duration.ofSeconds(300));
                } catch (Exception ignored) {}
            }
            throw primary;
        }
    }

    private void streamDirectWithProvider(String id, InferenceRequest request, long startTime) {
        ProviderRequest providerRequest = buildDirectProviderRequest(request, true);
        Optional<LLMProvider> providerOpt = providerRegistry.getProvider(id);
        if (providerOpt.isEmpty()) {
            throw new RuntimeException("Provider not available: " + id);
        }
        if (!(providerOpt.get() instanceof tech.kayys.gollek.spi.provider.StreamingProvider streamingProvider)) {
            throw new RuntimeException("Provider does not support direct streaming: " + id);
        }

        java.io.ByteArrayOutputStream imageBuffer = new java.io.ByteArrayOutputStream();
        java.io.ByteArrayOutputStream audioBuffer = new java.io.ByteArrayOutputStream();
        CountDownLatch latch = new CountDownLatch(1);
        java.util.concurrent.atomic.AtomicReference<java.util.Map<String, Object>> metricsRef = new java.util.concurrent.atomic.AtomicReference<>();
        java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicBoolean routePrinted = new java.util.concurrent.atomic.AtomicBoolean(false);
        java.util.concurrent.atomic.AtomicLong firstTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
        java.util.concurrent.atomic.AtomicLong lastTokenTime = new java.util.concurrent.atomic.AtomicLong(0);
        long streamStartTime = System.currentTimeMillis();

        streamingProvider.inferStream(providerRequest)
                .subscribe().with(
                        chunk -> {
                            if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                metricsRef.set(chunk.metadata());
                                if (!enableJsonSse && routePrinted.compareAndSet(false, true)) {
                                    printExecutionRouteInfo(chunk.metadata());
                                }
                            }

                            if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.IMAGE) {
                                if (chunk.imageDeltaBase64() != null) {
                                    try {
                                        byte[] decoded = java.util.Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                        imageBuffer.write(decoded);
                                    } catch (Exception ignored) {}
                                }
                                return;
                            }

                            if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.AUDIO) {
                                if (chunk.getDelta() != null) {
                                    try {
                                        byte[] decoded = java.util.Base64.getDecoder().decode(chunk.getDelta());
                                        audioBuffer.write(decoded);
                                    } catch (Exception ignored) {}
                                }
                                return;
                            }

                            String delta = chunk.getDelta();
                            if (delta != null) {
                                boolean progressDelta = delta.startsWith("[") && delta.contains("]");
                                if (!progressDelta && !delta.isEmpty()) {
                                    long now = System.currentTimeMillis();
                                    firstTokenTime.compareAndSet(0, now);
                                    lastTokenTime.set(now);
                                }
                                if (progressDelta) {
                                    if (!enableJsonSse) {
                                        System.out.print("\r" + ChatUIRenderer.CYAN + delta + ChatUIRenderer.RESET + "  ");
                                        System.out.flush();
                                    }
                                } else if (enableJsonSse) {
                                    printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                } else {
                                    System.out.print(delta);
                                    System.out.flush();
                                }
                                tokenCount.incrementAndGet();
                            }
                        },
                        error -> {
                            if (shouldIgnoreDirectProviderError(error, tokenCount.get())) {
                                long duration = observedStreamDurationMillis(
                                        streamStartTime, System.currentTimeMillis(), lastTokenTime);
                                handleOutputs(imageBuffer, audioBuffer);
                                double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                Double ttftMs = ttftMillis(metricsRef.get(), streamStartTime, firstTokenTime);
                                Map<String, Object> streamMetrics = observedStreamMetrics(
                                        metricsRef.get(), tokenCount.get(), duration, ttftMs);
                                if (!enableJsonSse) {
                                    System.err.println();
                                    System.err.println("Warning: ignored native provider shutdown bug after streaming output.");
                                    uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, ttftMs, false);
                                    uiRenderer.printBenchmarks(streamMetrics, false);
                                }
                                latch.countDown();
                                return;
                            }
                            uiRenderer.printError(error.getMessage(), false);
                            printProviderHintFromError(error);
                            latch.countDown();
                        },
                        () -> {
                            long duration = observedStreamDurationMillis(
                                    streamStartTime, System.currentTimeMillis(), lastTokenTime);
                            handleOutputs(imageBuffer, audioBuffer);
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            Double ttftMs = ttftMillis(metricsRef.get(), streamStartTime, firstTokenTime);
                            Map<String, Object> streamMetrics = observedStreamMetrics(
                                    metricsRef.get(), tokenCount.get(), duration, ttftMs);
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else {
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, ttftMs, false);
                                uiRenderer.printBenchmarks(streamMetrics, false);
                            }
                            latch.countDown();
                        });

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private ProviderRequest buildDirectProviderRequest(InferenceRequest request, boolean streaming) {
        Object modelPathParam = request.getParameters().get("model_path");
        String providerModel = (modelPathParam != null && !String.valueOf(modelPathParam).isBlank())
                ? String.valueOf(modelPathParam)
                : request.getModel();
        return ProviderRequest.builder()
                .model(providerModel)
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(streaming)
                .timeout(Duration.ofSeconds(120))
                .metadata("request_id", request.getRequestId())
                .metadata("tenantId", "community")
                .build();
    }

    private boolean shouldIgnoreDirectProviderError(Throwable error, int emittedTokens) {
        if (error == null || emittedTokens <= 0) {
            return false;
        }
        String message = error.getMessage();
        return message != null && message.contains("Attempted to close a non-closeable session");
    }

    private boolean ensureProviderHealthy(String provider) {
        try {
            ensureBuiltinProviderRegistration();
            if (providerRegistry != null) {
                Optional<LLMProvider> registered = providerRegistry.getProvider(provider);
                if (registered.isPresent()) {
                    try {
                        ProviderHealth health = registered.get().health().await().atMost(Duration.ofSeconds(5));
                        if (health != null && health.status() == ProviderHealth.Status.UNHEALTHY) {
                            System.err.printf("Provider '%s' is unhealthy.%n", provider);
                            return false;
                        }
                    } catch (Exception ignored) {
                        // Keep as available; some providers lazily initialize during first infer.
                    }
                    return true;
                }
            }
            Optional<ProviderInfo> info = sdk.listAvailableProviders().stream().filter(p -> provider.equalsIgnoreCase(p.id())).findFirst();
            if (info.isEmpty()) {
                System.err.printf("Required provider is not available: %s%n", provider);
                return false;
            }
            if (info.get().healthStatus() == ProviderHealth.Status.UNHEALTHY) {
                System.err.printf("Provider '%s' is unhealthy.%n", provider);
                return false;
            }
            return true;
        } catch (Exception e) { return false; }
    }

    private boolean validateGemma3ExecutionRoute(String provider, String requestedModelId, String localPath) {
        if (provider == null || localPath == null || localPath.isBlank()) {
            return true;
        }
        if (!"safetensor".equalsIgnoreCase(provider)) {
            return true;
        }
        try {
            Path path = Path.of(localPath);
            Path modelDir = Files.isDirectory(path) ? path : path.getParent();
            if (modelDir == null || !Files.isDirectory(modelDir)) {
                return true;
            }
            String modelType = readModelType(modelDir);
            if (modelType == null || !modelType.trim().toLowerCase(Locale.ROOT).startsWith("gemma3")) {
                return true;
            }
            String hint = normalizeArtifactHint(requestedModelId);
            if (hint.contains("functiongemma")) {
                // FunctionGemma is expected to run through safetensor in this build.
                return true;
            }
            Optional<Path> litert = findPreferredAlternateArtifact(modelDir, requestedModelId, ".litertlm", ".tflite", ".task");
            Optional<Path> gguf = findPreferredAlternateArtifact(modelDir, requestedModelId, ".gguf");
            if (litert.isPresent() || gguf.isPresent()) {
                // There is at least one compatible alternate artifact candidate; let normal flow continue.
                return true;
            }

            System.err.println("Error: Gemma3 safetensor direct path is disabled for quality/safety in this build.");
            System.err.println("No compatible local runtime artifact was found for this checkpoint.");
            if (hint.contains("functiongemma")) {
                System.err.println("Hint: FunctionGemma currently needs a compatible GGUF/LiteRT artifact generated for this build profile.");
            }
            System.err.println("Try:");
            System.err.println("  gollek pull gguf:google/functiongemma-270m-it");
            System.err.println("  gollek run --provider gguf --model google/functiongemma-270m-it --prompt \"who are you\"");
            System.err.println("If gguf provider is unavailable, rebuild with gguf runtime/plugin enabled.");
            return false;
        } catch (Exception ignored) {
            return true;
        }
    }

    private void printCompatibilityHintBeforeInference() {
        if (providerId == null || modelId == null) return;
        String provider = providerId.toLowerCase();
        String model = modelId.toLowerCase();
        if ("safetensor".equals(provider)) {
            if (model.contains("vlm") || model.contains("vision")) {
                System.err.println("Hint: 'safetensor' provider is text-oriented; VLM models may fail.");
            }
        } else if ("llamacpp".equals(provider) && !model.endsWith(".gguf")) {
            System.err.println("Hint: 'llamacpp' works best with GGUF models.");
        }
    }

    private void printProviderHintFromError(Throwable throwable) {
        String detail = throwable.getMessage() != null ? throwable.getMessage().toLowerCase() : "";
        if (detail.contains("429") || detail.contains("quota")) {
            System.err.println("Hint: Provider quota reached. Wait or switch provider.");
        }
    }

    private void autoOpenImage(Path imagePath) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac")) new ProcessBuilder("open", imagePath.toString()).start();
            else if (os.contains("linux")) new ProcessBuilder("xdg-open", imagePath.toString()).start();
        } catch (Exception ignored) {}
    }

    private void autoOpenAudio(Path audioPath) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac")) new ProcessBuilder("afplay", audioPath.toString()).start();
        } catch (Exception ignored) {}
    }

    private boolean tryStandaloneLiteRtExecution(String localPath, long startTime) {
        try {
            Class<?> clazz = Class.forName("tech.kayys.gollek.provider.litert.LiteRTProvider");
            Object instance = clazz.getDeclaredConstructor().newInstance();
            if (!(instance instanceof LLMProvider provider)) {
                return false;
            }

            var params = new LinkedHashMap<String, Object>();
            params.put("model_path", localPath);

            ProviderRequest providerRequest = ProviderRequest.builder()
                    .model(localPath)
                    .messages(List.of(Message.user(prompt)))
                    .parameters(params)
                    .streaming(false)
                    .timeout(Duration.ofSeconds(120))
                    .metadata("request_id", UUID.randomUUID().toString())
                    .metadata("tenantId", "community")
                    .build();

            InferenceResponse response = provider.infer(providerRequest).await().atMost(Duration.ofSeconds(300));
            printResponse(response, startTime);
            requestProcessExit();
            return true;
        } catch (Throwable t) {
            System.err.println("LiteRT standalone fallback failed: " + t.getMessage());
            return false;
        }
    }

    private void requestProcessExit() {
        try {
            Quarkus.asyncExit(0);
        } catch (Throwable ignored) {
        }
        System.exit(0);
    }

    private record ProviderLocalPathResolution(boolean ok, String localPath) {
    }
}
