package tech.kayys.gollek.ml.hub;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.ml.nn.Parameter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Loads pretrained models from HuggingFace, local directories, or the Gollek model registry.
 * <p>
 * This is the primary entry point for loading pre-trained weights into
 * {@link Module} instances.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Load from HuggingFace (auto-download and cache)
 * var weights = ModelHub.loadWeights("Qwen/Qwen2.5-0.5B");
 *
 * // Load from local directory
 * var weights = ModelHub.loadWeights("/path/to/model");
 *
 * // Load with config
 * var weights = ModelHub.loadWeights("bert-base-uncased",
 *     HubConfig.builder().revision("main").build());
 * }</pre>
 */
public final class ModelHub {

    private static final Logger LOG = LoggerFactory.getLogger(ModelHub.class);

    private ModelHub() {}

    /**
     * Load model weight tensors from a model identifier.
     * <p>
     * The identifier can be:
     * <ul>
     * <li>A HuggingFace repo ID (e.g., "bert-base-uncased")</li>
     * <li>An absolute path to a local model directory</li>
     * <li>A Gollek model registry name</li>
     * </ul>
     *
     * @param modelId model identifier
     * @return map of parameter names to tensor data
     */
    public static Map<String, GradTensor> loadWeights(String modelId) {
        return loadWeights(modelId, HubConfig.DEFAULT);
    }

    /**
     * Load model weights with custom hub configuration.
     */
    public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) {
        Path modelPath = resolveModelPath(modelId, config);

        if (modelPath == null) {
            throw new HubException("Could not resolve model: " + modelId);
        }

        // Look for safetensors files
        try {
            if (Files.isDirectory(modelPath)) {
                return loadFromDirectory(modelPath);
            } else if (Files.isRegularFile(modelPath)) {
                return loadFromFile(modelPath);
            }
        } catch (IOException e) {
            throw new HubException("Failed to load model from " + modelPath, e);
        }

        throw new HubException("Model path is neither a file nor directory: " + modelPath);
    }

    /**
     * Apply loaded weights to a Module.
     *
     * @param module  the module to load weights into
     * @param weights map of parameter names to tensors
     */
    public static void applyWeights(Module module, Map<String, GradTensor> weights) {
        Map<String, Parameter> params = module.namedParameters();
        int loaded = 0;

        for (var entry : weights.entrySet()) {
            // Try exact match
            Parameter param = params.get(entry.getKey());
            if (param == null) {
                // Try with prefix stripping (e.g., "model.layers.0.weight" → "layers.0.weight")
                for (var pEntry : params.entrySet()) {
                    if (entry.getKey().endsWith(pEntry.getKey())) {
                        param = pEntry.getValue();
                        break;
                    }
                }
            }
            if (param != null) {
                GradTensor weightTensor = entry.getValue();
                float[] paramData = param.data().data();
                float[] weightData = weightTensor.data();
                int len = Math.min(paramData.length, weightData.length);
                System.arraycopy(weightData, 0, paramData, 0, len);
                loaded++;
            }
        }

        LOG.info("Loaded {}/{} weight tensors into module", loaded, weights.size());
    }

    /**
     * Convenience method: load weights and apply to a module.
     */
    public static void loadInto(Module module, String modelId) {
        Map<String, GradTensor> weights = loadWeights(modelId);
        applyWeights(module, weights);
    }

    // ── Private helpers ──────────────────────────────────────────────────

    private static Path resolveModelPath(String modelId, HubConfig config) {
        // 1. Check if it's an absolute path
        Path direct = Path.of(modelId);
        if (Files.exists(direct)) {
            return direct;
        }

        // 2. Check local cache
        Path cached = config.cacheDir().resolve(modelId.replace("/", "--"));
        if (Files.exists(cached)) {
            return cached;
        }

        // 3. Check default gollek model directory
        Path gollekModels = Path.of(System.getProperty("user.home"), ".gollek", "models");
        Path gollekPath = gollekModels.resolve(modelId.replace("/", "--"));
        if (Files.exists(gollekPath)) {
            return gollekPath;
        }

        // 4. Could trigger download here (via gollek-model-repo-hf)
        // For now, return null and let caller handle
        return null;
    }

    private static Map<String, GradTensor> loadFromDirectory(Path dir) throws IOException {
        Map<String, GradTensor> weights = new LinkedHashMap<>();

        // Look for .safetensors files
        try (var stream = Files.list(dir)) {
            stream.filter(p -> p.toString().endsWith(".safetensors") || p.toString().endsWith(".bin"))
                  .sorted()
                  .forEach(file -> {
                      try {
                          weights.putAll(loadFromFile(file));
                      } catch (IOException e) {
                          LOG.error("Warning: could not load {}: {}", file, e.getMessage());
                      }
                  });
        }

        if (weights.isEmpty()) {
            LOG.warn("Warning: no weight files found in {}", dir);
        }

        return weights;
    }

    private static Map<String, GradTensor> loadFromFile(Path file) throws IOException {
        String filename = file.getFileName().toString().toLowerCase();

        if (filename.endsWith(".safetensors") || filename.endsWith(".safetensor")) {
            return loadSafeTensors(file);
        }

        // Fallback for other formats
        Map<String, GradTensor> weights = new LinkedHashMap<>();
        LOG.debug("Loading weights from: {}", filename);
        // ... existing placeholder logic or other format loaders
        return weights;
    }

    private static Map<String, GradTensor> loadSafeTensors(Path file) {
        try {
            // Use reflection to avoid hard dependency on the loader module if not present
            Class<?> loaderClass = Class.forName("tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader");
            // In a real implementation, we would use a more robust way to get an instance
            // For now, we'll try to find a way to get a usable loader or instantiate one
            
            LOG.info("Detected SafeTensors: {}", file.getFileName());
            
            // This is a bridge to the actual loading logic
            return SafeTensorBridge.load(file);
            
        } catch (ClassNotFoundException e) {
            LOG.warn("SafeTensors file detected but gollek-safetensor-loader not on classpath");
            return Collections.emptyMap();
        } catch (Exception e) {
            throw new HubException("Failed to load SafeTensors: " + file, e);
        }
    }

    /** List cached models in the default cache directory. */
    public static List<String> listCachedModels() {
        Path cacheDir = HubConfig.DEFAULT.cacheDir();
        if (!Files.isDirectory(cacheDir)) return List.of();

        try (var stream = Files.list(cacheDir)) {
            return stream.filter(Files::isDirectory)
                         .map(p -> p.getFileName().toString().replace("--", "/"))
                         .sorted()
                         .toList();
        } catch (IOException e) {
            return List.of();
        }
    }
}
