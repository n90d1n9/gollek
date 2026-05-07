package tech.kayys.gollek.cli.commands;

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
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.exception.SdkException;
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
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import tech.kayys.gollek.cli.chat.ChatUIRenderer;
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

    @Override
    public void run() {
        try {
            if (parentCommand != null) {
                parentCommand.bootstrapInheritedEnvironment();
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

            if (parentCommand == null || !parentCommand.verbose) {
                System.out.println(ChatUIRenderer.CYAN + "Platform: " + detectedPlatform.getDisplayName() + ChatUIRenderer.RESET);
                if (detectedPlatform.isCpu()) {
                    System.out.println(ChatUIRenderer.YELLOW + "⚠️  Running on CPU (GPU acceleration not available)" + ChatUIRenderer.RESET);
                } else {
                    System.out.println(ChatUIRenderer.GREEN + "✓ GPU acceleration enabled" + ChatUIRenderer.RESET);
                }
                System.out.println();
            }

            long startTime = System.currentTimeMillis();

            // Check if specific provider is requested but not available
            if (providerId != null && !providerId.trim().isEmpty()) {
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
                providerId = "safetensor";
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
                        System.out.println(resolution.getNotice());
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

            if (direct) {
                providerId = "safetensor";
                sdk.setPreferredProvider("safetensor");
            } else if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }
            
            if (providerId != null && !ensureProviderHealthy(providerId)) {
                return;
            }
            
            printCompatibilityHintBeforeInference();

            String requestModelTarget = shouldUseLocalModelPath(providerId, finalLocalPath)
                    ? finalLocalPath
                    : modelId;

            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(requestModelTarget)
                    .temperature(temperature)
                    .topP(topP)
                    .topK(topK)
                    .repeatPenalty(repeatPenalty)
                    .jsonMode(jsonMode)
                    .maxTokens(maxTokens)
                    .plugin(pluginId)
                    .streaming(stream);

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

            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(systemPrompt));
            }
            requestBuilder.message(Message.user(prompt));

            if (providerId != null && !providerId.isEmpty()) {
                requestBuilder.preferredProvider(providerId);
            }

            requestBuilder.cacheBypass(noCache);

            InferenceRequest request = requestBuilder.build();

            if (!enableJsonSse) {
                uiRenderer.printModelInfo(modelId, providerId, format, null, false);
                printQuantizationInfo();
            }

            if (stream) {
                java.io.ByteArrayOutputStream imageBuffer = new java.io.ByteArrayOutputStream();
                java.io.ByteArrayOutputStream audioBuffer = new java.io.ByteArrayOutputStream();
                CountDownLatch latch = new CountDownLatch(1);
                java.util.concurrent.atomic.AtomicReference<java.util.Map<String, Object>> metricsRef = new java.util.concurrent.atomic.AtomicReference<>();
                java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(0);
                
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
                                        if (delta.startsWith("[") && delta.contains("]")) {
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
                                    long duration = System.currentTimeMillis() - startTime;
                                    handleOutputs(imageBuffer, audioBuffer);
                                    double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                    if (enableJsonSse) {
                                        printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                    } else {
                                        uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, false);
                                        uiRenderer.printBenchmarks(metricsRef.get(), false);
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

    private String detectSafetensorCompatibilityIssue(String providerId, String localPath) {
        if (!"safetensor".equalsIgnoreCase(providerId) || localPath == null || localPath.isBlank()) {
            return null;
        }
        if (!isSafetensorCheckpointDir(localPath)) {
            return null;
        }

        try {
            Path configPath = Path.of(localPath).resolve("config.json");
            String config = Files.readString(configPath);

            boolean gemma4Conditional = config.contains("\"Gemma4ForConditionalGeneration\"")
                    || config.contains("\"model_type\": \"gemma4\"");
            boolean multimodalWrapper = config.contains("\"vision_config\"")
                    || config.contains("\"audio_config\"");
            boolean perLayerInputs = config.contains("\"hidden_size_per_layer_input\"");
            boolean sharedKv = config.contains("\"num_kv_shared_layers\"");

            if (gemma4Conditional && (multimodalWrapper || perLayerInputs || sharedKv)) {
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
            case "int4", "turbo" -> "off";
            default -> normalized;
        };
    }

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    private String inferProviderFromIndex(LocalModelIndex.Entry entry) {
        if (entry == null || entry.format == null) {
            return null;
        }
        return providerForFormat(entry.format);
    }

    private Optional<ModelInfo> findSupportedLocalFallback(String requestedModelId, String localPath, String currentProvider) {
        if (!"safetensor".equalsIgnoreCase(currentProvider)) {
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
                        String format = model.getFormat();
                        return format != null && !format.equalsIgnoreCase("safetensors") && !format.equalsIgnoreCase("safetensor");
                    })
                    .filter(model -> {
                        String candidatePath = extractModelPath(model);
                        return candidatePath != null && !candidatePath.equals(localPath) && Files.exists(Path.of(candidatePath));
                    })
                    .sorted((left, right) -> Integer.compare(runtimePriority(right.getFormat()), runtimePriority(left.getFormat())))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private int runtimePriority(String format) {
        if (format == null) {
            return 0;
        }
        return switch (format.trim().toUpperCase(java.util.Locale.ROOT)) {
            case "GGUF" -> 30;
            case "ONNX" -> 20;
            case "LITERT", "TFLITE", "TASK" -> 10;
            default -> 0;
        };
    }

    private String providerForFormat(String format) {
        if (format == null) {
            return null;
        }
        String normalized = format.trim().toLowerCase(java.util.Locale.ROOT);
        return switch (normalized) {
            case "gguf" -> "native";
            case "safetensors", "safetensor" -> "safetensor";
            case "litert", "task", "tflite" -> "litert";
            case "onnx" -> "onnx";
            case "torchscript", "pytorch" -> "libtorch";
            default -> null;
        };
    }

    private String extractModelPath(ModelInfo modelInfo) {
        if (modelInfo == null || modelInfo.getMetadata() == null) {
            return null;
        }
        Object value = modelInfo.getMetadata().get("path");
        return value != null ? value.toString() : null;
    }

    private void printResponse(InferenceResponse response, long startTime) {
        Map<String, Object> metadata = response.getMetadata();
        printQuantCacheInfo(metadata);
        System.out.println();
        System.out.println(ChatUIRenderer.GREEN + response.getContent() + ChatUIRenderer.RESET);
        
        if (metadata.containsKey("audio")) {
            saveAudio(metadata.get("audio").toString());
        } else if (metadata.containsKey("image")) {
            saveImage(metadata.get("image").toString());
        }

        double seconds = Math.max(response.getDurationMs() / 1000.0, 0.001);
        double tps = response.getTokensUsed() / seconds;
        uiRenderer.printStats(response.getTokensUsed(), response.getDurationMs() / 1000.0, tps, false);
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

        streamingProvider.inferStream(providerRequest)
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
                                if (delta.startsWith("[") && delta.contains("]")) {
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
                                long duration = System.currentTimeMillis() - startTime;
                                handleOutputs(imageBuffer, audioBuffer);
                                double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                if (!enableJsonSse) {
                                    System.err.println();
                                    System.err.println("Warning: ignored native provider shutdown bug after streaming output.");
                                    uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, false);
                                    uiRenderer.printBenchmarks(metricsRef.get(), false);
                                }
                                latch.countDown();
                                return;
                            }
                            uiRenderer.printError(error.getMessage(), false);
                            printProviderHintFromError(error);
                            latch.countDown();
                        },
                        () -> {
                            long duration = System.currentTimeMillis() - startTime;
                            handleOutputs(imageBuffer, audioBuffer);
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else {
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, false);
                                uiRenderer.printBenchmarks(metricsRef.get(), false);
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

    private void requestProcessExit() {
        try {
            Quarkus.asyncExit(0);
        } catch (Throwable ignored) {
        }
        System.exit(0);
    }
}
