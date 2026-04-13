package tech.kayys.gollek.onnx.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;
import tech.kayys.gollek.spi.model.ModalityType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Stable Diffusion ONNX Runner for Gollek.
 * 
 * Orchestrates CLIP, UNet, and VAE components via individual ONNX Runtime sessions.
 */
@ApplicationScoped
public class StableDiffusionOnnxRunner extends AbstractGollekRunner {

    private static final Logger LOG = Logger.getLogger(StableDiffusionOnnxRunner.class);
    public static final String RUNNER_NAME = "sd-onnx";

    @ConfigProperty(name = "gollek.runners.onnx.library-path", defaultValue = "/usr/lib/libonnxruntime.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.onnx.execution-provider", defaultValue = "auto")
    String executionProvider;

    private OnnxRuntimeBinding ort;
    private MemorySegment ortEnv = MemorySegment.NULL;
    private MemorySegment memInfo = MemorySegment.NULL;

    // Sessions for different pipeline components
    private MemorySegment textEncoderSession = MemorySegment.NULL;
    private MemorySegment unetSession = MemorySegment.NULL;
    private MemorySegment vaeDecoderSession = MemorySegment.NULL;

    private String resolvedEp = "cpu";
    private ModelManifest manifest;

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "onnxruntime";
    }

    @Override
    public DeviceType deviceType() {
        return executionProvider.toLowerCase().contains("coreml") || executionProvider.equals("auto") ? DeviceType.METAL : DeviceType.CPU;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.0.0", List.of(ModelFormat.ONNX), List.of(DeviceType.CPU, DeviceType.METAL), Map.of());
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportedDataTypes(new String[]{"fp32", "fp16"})
                .build();
    }

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config) throws RunnerInitializationException {
        this.manifest = modelManifest;
        
        // Resolve ORT library
        Path libPath = resolveLibraryPath(libraryPath);
        OnnxRuntimeBinding.initialize(libPath);
        ort = OnnxRuntimeBinding.getInstance();

        if (!ort.isNativeAvailable()) {
            throw new RunnerInitializationException(ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "Native ONNX Runtime not available");
        }

        ortEnv = ort.createEnv("gollek-sd");
        memInfo = ort.createCpuMemoryInfo();

        MemorySegment opts = ort.createSessionOptions();
        resolvedEp = resolveAndAttachEp(opts);

        // Resolve component paths
        Path baseDir = resolveBaseDir(modelManifest);
        
        try {
            textEncoderSession = ort.createSession(ortEnv, baseDir.resolve("text_encoder/model.onnx").toString(), opts);
            unetSession = ort.createSession(ortEnv, baseDir.resolve("unet/model.onnx").toString(), opts);
            vaeDecoderSession = ort.createSession(ortEnv, baseDir.resolve("vae_decoder/model.onnx").toString(), opts);
        } catch (Exception e) {
            throw new RunnerInitializationException(ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "Failed to load SD components: " + e.getMessage());
        } finally {
            ort.releaseSessionOptions(opts);
        }

        this.initialized = true;
        LOG.infof("[SD-ONNX] Initialized pipeline at %s (EP: %s)", baseDir, resolvedEp);
    }

    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        throw new UnsupportedOperationException("Use stream() for Stable Diffusion to get progress updates and image chunks.");
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                String reqId = request.getRequestId();
                String prompt = request.getPrompt().orElse("");
                int steps = request.getParameter("steps", Integer.class).orElse(20);
                float guidance = request.getParameter("guidance_scale", Float.class).orElse(7.5f);

                LOG.infof("[SD-ONNX] Generating image for prompt: '%s' (steps=%d, guidance=%.1f)", prompt, steps, guidance);

                try (Arena arena = Arena.ofShared()) {
                    // 1. Text Encoding
                    emitter.emit(StreamingInferenceChunk.textDelta(reqId, 0, "Encoding prompt..."));
                    MemorySegment textEmbeds = encodePrompt(arena, prompt);
                    MemorySegment nullEmbeds = encodePrompt(arena, "");

                    // 2. Latent Initialization (Random noise [1, 4, 64, 64])
                    emitter.emit(StreamingInferenceChunk.textDelta(reqId, 1, "Initializing latents..."));
                    MemorySegment latents = createNoiseLatents(arena);

                    // 3. Diffusion Loop
                    for (int i = 0; i < steps; i++) {
                        float progress = (float) i / steps;
                        emitter.emit(StreamingInferenceChunk.textDelta(reqId, i + 2, String.format("Denoising step %d/%d...", i + 1, steps)));
                        
                        // Prediction and step logic (Simplified)
                        latents = denoiseStep(arena, latents, textEmbeds, nullEmbeds, i, steps, guidance);
                        
                        // Progress metadata could be added here
                    }

                    // 4. VAE Decoding
                    emitter.emit(StreamingInferenceChunk.textDelta(reqId, steps + 2, "Decoding image..."));
                    byte[] pngData = decodeToPng(arena, latents);

                    // 5. Finalize
                    String base64 = Base64.getEncoder().encodeToString(pngData);
                    emitter.emit(StreamingInferenceChunk.imageChunk(reqId, steps + 3, base64, true));
                    emitter.complete();
                }

            } catch (Exception e) {
                LOG.error("[SD-ONNX] Generation failed", e);
                emitter.fail(e);
            }
        });
    }

    // ── Internal Pipeline Logic ──────────────────────────────────────────────

    private MemorySegment encodePrompt(Arena arena, String prompt) {
        // In a real implementation, we'd tokenize and run textEncoderSession
        // For now, returning a dummy tensor of shape [1, 77, 768]
        float[] dummy = new float[1 * 77 * 768];
        MemorySegment data = arena.allocate(dummy.length * 4L, 4);
        return ort.createTensorWithData(memInfo, data, new long[]{1, 77, 768}, OnnxRuntimeBinding.ONNX_TENSOR_FLOAT);
    }

    private MemorySegment createNoiseLatents(Arena arena) {
        float[] noise = new float[1 * 4 * 64 * 64];
        Random rnd = new Random();
        for (int i = 0; i < noise.length; i++) noise[i] = (float) rnd.nextGaussian();
        MemorySegment data = arena.allocate(noise.length * 4L, 4);
        for (int i = 0; i < noise.length; i++) data.setAtIndex(ValueLayout.JAVA_FLOAT, i, noise[i]);
        return ort.createTensorWithData(memInfo, data, new long[]{1, 4, 64, 64}, OnnxRuntimeBinding.ONNX_TENSOR_FLOAT);
    }

    private MemorySegment denoiseStep(Arena arena, MemorySegment latents, MemorySegment textEmbeds, MemorySegment nullEmbeds, int step, int totalSteps, float guidance) {
        // Placeholder for UNet forward and scheduler step
        // In a real implementation: run unetSession twice (cond/uncond) and apply DDIM/Euler step
        return latents; 
    }

    private byte[] decodeToPng(Arena arena, MemorySegment latents) {
        // Placeholder for VAE decode and PNG conversion
        // Returning a 1x1 transparent pixel for now to satisfy the flow
        return Base64.getDecoder().decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==");
    }

    private Path resolveBaseDir(ModelManifest manifest) {
        return manifest.artifacts().values().stream()
                .findFirst()
                .map(loc -> {
                    String uri = loc.uri();
                    if (uri.startsWith("file:")) return Path.of(java.net.URI.create(uri));
                    return Path.of(uri);
                })
                .map(p -> Files.isDirectory(p) ? p : p.getParent())
                .orElseThrow();
    }

    private String resolveAndAttachEp(MemorySegment opts) {
        // Reuse same logic as OnnxRuntimeRunner or implement specifically
        if (executionProvider.equals("auto") || executionProvider.contains("coreml")) {
            if (ort.appendCoreMlProvider(opts, 0)) return "CoreMLExecutionProvider";
        }
        return "CPUExecutionProvider";
    }

    private Path resolveLibraryPath(String configuredPath) {
        // Reuse logic from OnnxRuntimeRunner or centralize it
        Path configured = Path.of(configuredPath);
        if (java.nio.file.Files.exists(configured)) return configured;
        return Path.of(System.getProperty("user.home"), ".gollek", "libs", "onnxruntime", "libonnxruntime.dylib");
    }

    @Override
    public void close() {
        if (ort != null) {
            if (!textEncoderSession.equals(MemorySegment.NULL)) ort.releaseSession(textEncoderSession);
            if (!unetSession.equals(MemorySegment.NULL)) ort.releaseSession(unetSession);
            if (!vaeDecoderSession.equals(MemorySegment.NULL)) ort.releaseSession(vaeDecoderSession);
            if (!ortEnv.equals(MemorySegment.NULL)) ort.releaseEnv(ortEnv);
            if (!memInfo.equals(MemorySegment.NULL)) ort.releaseMemoryInfo(memInfo);
        }
        this.initialized = false;
    }
}
