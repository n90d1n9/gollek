package tech.kayys.gollek.cli.commands;

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

            if (parentCommand != null) {
                parentCommand.applyRuntimeOverrides();
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
            } else {
                // Prepare model using SDK (this handles pulling, registration, and conversion)
                try {
                    String quant = ggufQuant != null ? ggufQuant : (ggufOutType != null ? ggufOutType : "Q4_K_M");
                    var resolution = sdk.ensureModelAvailable(modelId, (String) format, pluginId, forceGguf, quant, (Consumer<PullProgress>) progress -> {
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
                    
                    if (providerId == null || providerId.isBlank()) {
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

            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(modelId)
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
            boolean directProviderBypass = "safetensor".equalsIgnoreCase(providerId);

            if (!enableJsonSse) {
                uiRenderer.printModelInfo(modelId, providerId, format, null, false);
            }

            if (directProviderBypass) {
                InferenceResponse response = inferDirectWithProvider(providerId, request);
                printResponse(response, startTime);
                return;
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
            } else {
                InferenceResponse response = sdk.createCompletion(request);
                printResponse(response, startTime);
            }

        } catch (Throwable e) {
            System.err.println("\n[FATAL] RunCommand failed: " + e.getMessage());
            System.exit(1);
        }
    }

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    private void printResponse(InferenceResponse response, long startTime) {
        System.out.println();
        System.out.println(ChatUIRenderer.GREEN + response.getContent() + ChatUIRenderer.RESET);
        
        Map<String, Object> metadata = response.getMetadata();
        if (metadata.containsKey("audio")) {
            saveAudio(metadata.get("audio").toString());
        } else if (metadata.containsKey("image")) {
            saveImage(metadata.get("image").toString());
        }

        double seconds = Math.max(response.getDurationMs() / 1000.0, 0.001);
        double tps = response.getTokensUsed() / seconds;
        uiRenderer.printStats(response.getTokensUsed(), response.getDurationMs() / 1000.0, tps, false);
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
        Object modelPathParam = request.getParameters().get("model_path");
        String providerModel = (modelPathParam != null && !String.valueOf(modelPathParam).isBlank()) ? String.valueOf(modelPathParam) : request.getModel();
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model(providerModel).messages(request.getMessages()).parameters(request.getParameters())
                .streaming(false).timeout(Duration.ofSeconds(120)).metadata("request_id", request.getRequestId())
                .metadata("tenantId", "community").build();
        try {
            Optional<LLMProvider> providerOpt = providerRegistry.getProvider(id);
            if (providerOpt.isEmpty()) throw new RuntimeException("Provider not available: " + id);
            return providerOpt.get().infer(providerRequest).await().atMost(Duration.ofSeconds(300));
        } catch (RuntimeException primary) {
            if (providerModel != null && (providerModel.toLowerCase().endsWith(".safetensors") || providerModel.toLowerCase().endsWith(".bin"))) {
                try {
                    System.out.println("Primary load failed; falling back to libtorch...");
                    return providerRegistry.getProvider("libtorch").get().infer(providerRequest).await().atMost(Duration.ofSeconds(300));
                } catch (Exception ignored) {}
            }
            throw primary;
        }
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
}
