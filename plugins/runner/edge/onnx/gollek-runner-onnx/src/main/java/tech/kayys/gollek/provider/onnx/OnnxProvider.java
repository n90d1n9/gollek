package tech.kayys.gollek.provider.onnx;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.onnx.runner.OnnxRuntimeRunner;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.provider.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;

/**
 * ONNX Runtime Provider for Gollek.
 * Provides a bridge between the Inference Engine and OnnxRuntimeRunner.
 */
@ApplicationScoped
@io.quarkus.arc.Unremovable
public class OnnxProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(OnnxProvider.class);
    private static final String PROVIDER_ID = "onnx";
    private static final String PROVIDER_VERSION = "1.0.0";

    @Inject
    OnnxProviderConfig config;

    @Inject
    OnnxRuntimeRunner runner;

    @Inject
    tech.kayys.gollek.spi.registry.LocalModelRegistry localModelRegistry;

    private volatile boolean initialized = false;

    @Override
    public boolean isEnabled() {
        return config != null && config.enabled();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "ONNX Runtime Provider";
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
                .description("ONNX Runtime inference provider for cross-platform hardware acceleration")
                .vendor("Kayys Tech")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .supportedFormats(Set.of(ModelFormat.ONNX))
                .supportedDevices(Set.of(DeviceType.CPU, DeviceType.METAL, DeviceType.CUDA))
                .build();
    }

    @Override
    public void initialize(ProviderConfig cfg) throws ProviderException.ProviderInitializationException {
        initialized = true;
        LOG.info("ONNX Provider initialized");
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        Path path = resolveModelPath(modelId, request);
        return path != null && Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            try {
                Path modelPath = resolveModelPath(request.getModel(), request);
                if (modelPath == null || !Files.exists(modelPath)) {
                    throw new RuntimeException("Model path not found: " + request.getModel());
                }

                ensureInitialized(request, modelPath);

                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(request.isStreaming())
                        .build();

                return runner.infer(inferenceRequest);
            } catch (Exception e) {
                LOG.error("ONNX Inference failed: " + e.getMessage(), e);
                throw new RuntimeException("Inference failed", e);
            }
        });
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                System.err.println("[OnnxProvider] inferStream for model: " + request.getModel());
                Path modelPath = resolveModelPath(request.getModel(), request);
                System.err.println("[OnnxProvider] Resolved model path: " + modelPath);
                
                if (modelPath == null || !Files.exists(modelPath)) {
                    System.err.println("[OnnxProvider] Model path NOT FOUND: " + modelPath);
                    emitter.fail(new RuntimeException("Model path not found: " + request.getModel()));
                    return;
                }

                ensureInitialized(request, modelPath);
                System.err.println("[OnnxProvider] Runner initialized successfully");

                var activeRunner = isStableDiffusion(modelPath) ? sdRunner : runner;
                activeRunner.stream(InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(true)
                        .build()).subscribe().with(
                            emitter::emit,
                            emitter::fail,
                            emitter::complete
                        );

            } catch (Exception e) {
                System.err.println("[OnnxProvider] inferStream failed: " + e.getMessage());
                e.printStackTrace(System.err);
                emitter.fail(e);
            }
        });
    }

    @Inject
    StableDiffusionOnnxRunner sdRunner;

    private void ensureInitialized(ProviderRequest request, Path modelPath) throws Exception {
        boolean isSd = isStableDiffusion(modelPath);
        var activeRunner = isSd ? sdRunner : runner;

        if (!activeRunner.health()) {
            System.err.println("[OnnxProvider] Initializing runner " + activeRunner.name() + "...");
            ModelManifest manifest = ModelManifest.builder()
                    .modelId(request.getModel())
                    .name(request.getModel())
                    .version("latest")
                    .path(modelPath.toAbsolutePath().toString())
                    .apiKey("none")
                    .requestId(request.getRequestId())
                    .artifacts(Map.of(ModelFormat.ONNX, new ArtifactLocation(modelPath.toUri().toString(), null, null, null)))
                    .build();
            
            RunnerConfiguration runnerCfg = RunnerConfiguration.builder()
                    .putParameter("intra_op_threads", config.threads())
                    .build();
            activeRunner.initialize(manifest, runnerCfg);
        }
    }

    private boolean isStableDiffusion(Path modelPath) {
        if (modelPath == null) return false;
        Path dir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
        return Files.isDirectory(dir.resolve("unet")) && Files.isDirectory(dir.resolve("vae_decoder"));
    }

    private Path resolveModelPath(String modelId, ProviderRequest request) {
        if (modelId == null) return null;

        Path targetPath = null;
        
        // Try metadata first (provided by ModelRouterService)
        if (request != null && request.getMetadata() != null) {
            Object pathObj = request.getMetadata().get("model_path");
            if (pathObj instanceof String pt && !pt.isBlank()) {
                Path p = Paths.get(pt);
                if (Files.exists(p)) {
                    targetPath = p;
                }
            }
        }
        
        // Try registry next
        if (targetPath == null && localModelRegistry != null) {
            java.util.Optional<tech.kayys.gollek.spi.registry.ModelEntry> entry = localModelRegistry.resolve(modelId);
            if (entry.isPresent() && entry.get().physicalPath() != null) {
                Path blobDir = entry.get().physicalPath();
                if (Files.exists(blobDir)) {
                    targetPath = blobDir;
                }
            }
        }

        // Same logic as LiteRTProvider for now
        if (targetPath == null) {
            try {
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
                        }
                    }
                }
            } catch (Exception e) {
                targetPath = Paths.get(modelId);
            }
        }

        if (targetPath != null && Files.isDirectory(targetPath)) {
            // For ONNX pipelines (like SD), the "model" might be a directory or a specific file
            // We'll look for .onnx files
            try (var stream = Files.walk(targetPath, 3)) {
                return stream.filter(p -> p.toString().endsWith(".onnx"))
                        .findFirst()
                        .orElse(targetPath);
            } catch (Exception e) {
                return targetPath;
            }
        }
        return targetPath;
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("ONNX provider operational"));
    }

    @Override
    public void shutdown() {
        runner.close();
        initialized = false;
        LOG.info("ONNX Provider shutdown complete");
    }
}
