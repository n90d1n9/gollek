package tech.kayys.gollek.ml.hub;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.core.model.ModelFormat;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.spi.model.ModelManifest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Unified model hub entry point for pretrained ML models.
 */
public final class ModelHub {

    private static final Logger LOG = LoggerFactory.getLogger(ModelHub.class);

    private ModelHub() {
    }

    public static void loadInto(NNModule model, String modelId) throws IOException {
        loadInto(model, modelId, HubConfig.DEFAULT);
    }

    public static void loadInto(NNModule model, String modelId, HubConfig config) throws IOException {
        Map<String, GradTensor> weights = loadWeights(modelId, config);
        applyWeights(model, weights);
    }

    public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
        return loadWeights(modelId, HubConfig.DEFAULT);
    }

    public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) throws IOException {
        String scheme = detectScheme(modelId);
        ModelRepository repo = ModelHubFactory.getRepository(scheme, config);

        ModelManifest manifest = repo.findById(modelId, "sdk-hub")
                .await().atMost(java.time.Duration.ofMinutes(10));

        if (manifest == null) {
            throw new IOException("Could not resolve model: " + modelId);
        }

        Path path = Path.of(manifest.path());
        if (scheme.equalsIgnoreCase("hf") || scheme.equalsIgnoreCase("kaggle")) {
            if (manifest.artifacts().containsKey(ModelFormat.GGUF)) {
                path = Path.of(java.net.URI.create(manifest.artifacts().get(ModelFormat.GGUF).uri()));
            } else if (manifest.artifacts().containsKey(ModelFormat.SAFETENSORS)) {
                path = Path.of(java.net.URI.create(manifest.artifacts().get(ModelFormat.SAFETENSORS).uri()));
            } else if (manifest.artifacts().containsKey(ModelFormat.LITERT)) {
                path = Path.of(java.net.URI.create(manifest.artifacts().get(ModelFormat.LITERT).uri()));
            }
        }

        return Files.isDirectory(path) ? loadFromDirectory(path) : loadFromFile(path);
    }

    public static void applyWeights(NNModule module, Map<String, GradTensor> weights) {
        Map<String, Parameter> params = module.namedParameters();
        int loaded = 0;

        for (var entry : weights.entrySet()) {
            Parameter param = params.get(entry.getKey());
            if (param == null) {
                for (var parameterEntry : params.entrySet()) {
                    if (entry.getKey().endsWith(parameterEntry.getKey())) {
                        param = parameterEntry.getValue();
                        break;
                    }
                }
            }
            if (param != null) {
                float[] paramData = param.data().data();
                float[] weightData = entry.getValue().data();
                int len = Math.min(paramData.length, weightData.length);
                System.arraycopy(weightData, 0, paramData, 0, len);
                loaded++;
            }
        }

        LOG.info("Loaded {}/{} weight tensors into module", loaded, weights.size());
    }

    private static String detectScheme(String modelId) {
        if (modelId.startsWith("hf:") || modelId.startsWith("huggingface:")) {
            return "hf";
        }
        if (modelId.startsWith("kaggle:")) {
            return "kaggle";
        }
        if (modelId.startsWith("onnx:") || modelId.startsWith("local:")) {
            return "local";
        }
        if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")
                || modelId.startsWith("~/")) {
            return "local";
        }
        if (modelId.length() > 2 && modelId.charAt(1) == ':'
                && (modelId.charAt(2) == '\\' || modelId.charAt(2) == '/')) {
            return "local";
        }
        return "hf";
    }

    private static Map<String, GradTensor> loadFromDirectory(Path dir) throws IOException {
        Map<String, GradTensor> weights = new LinkedHashMap<>();
        try (var stream = Files.list(dir)) {
            stream.filter(path -> path.toString().endsWith(".safetensors")
                            || path.toString().endsWith(".bin")
                            || path.toString().endsWith(".tflite")
                            || path.toString().endsWith(".onnx"))
                    .sorted()
                    .forEach(file -> {
                        try {
                            weights.putAll(loadFromFile(file));
                        } catch (IOException e) {
                            LOG.error("Failed to load {}: {}", file, e.getMessage());
                        }
                    });
        }
        return weights;
    }

    private static Map<String, GradTensor> loadFromFile(Path file) throws IOException {
        String filename = file.getFileName().toString().toLowerCase();
        if (filename.endsWith(".safetensors") || filename.endsWith(".safetensor")) {
            return SafeTensorBridge.load(file);
        }
        if (filename.endsWith(".tflite") || filename.endsWith(".onnx")) {
            LOG.info("Detected runtime-managed model file: {}", file);
            return new LinkedHashMap<>();
        }
        return new LinkedHashMap<>();
    }

    public static void setCacheDir(Path dir) {
        // compatibility no-op, prefer HubConfig
    }
}
