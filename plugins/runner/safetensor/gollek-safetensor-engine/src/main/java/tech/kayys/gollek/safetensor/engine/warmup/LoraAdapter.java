/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * LoraAdapter.java
 * ─────────────────
 * Loads and manages LoRA (Low-Rank Adaptation) adapter weights from SafeTensors files.
 *
 * LoRA adapters contain delta weights in the form of low-rank decomposition matrices
 * (A and B) that can be applied to base model weights at inference time:
 *
 *   W' = W + α * (B × A)
 *
 * where:
 *   - W is the base model weight
 *   - A ∈ R^(r×d) is the down-projection matrix (rank r, hidden dim d)
 *   - B ∈ R^(k×r) is the up-projection matrix (output dim k)
 *   - α is a scaling factor (typically α = r / rank)
 *
 * Adapter directory structure:
 *   <adapter-dir>/
 *     adapter_model.safetensors    — LoRA weights (lora_A, lora_B matrices)
 *     adapter_config.json          — adapter metadata (rank, target modules, alpha)
 *
 * Thread safety
 * ═════════════
 * LoadedAdapter instances are immutable after creation and can be safely shared
 * across threads. The underlying tensor weights are read-only during inference.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * LoRA adapter loader and manager.
 *
 * <p>
 * Inject via CDI and call {@link #load(Path)} to load an adapter from disk.
 * Loaded adapters are cached to avoid redundant disk I/O.
 *
 * <pre>{@code
 * @Inject LoraAdapter loraAdapter;
 *
 * LoadedAdapter adapter = loraAdapter.load(Path.of("/models/adapters/sql-lora"));
 * // adapter.weights() contains lora_A and lora_B matrices per module
 * // adapter.config() contains rank, alpha, target_modules, etc.
 * }</pre>
 */
@ApplicationScoped
public class LoraAdapter {

    private static final Logger log = Logger.getLogger(LoraAdapter.class);

    /** Default adapter filename within an adapter directory. */
    private static final String DEFAULT_ADAPTER_FILE = "adapter_model.safetensors";

    /** Default config filename for adapter metadata. */
    private static final String DEFAULT_CONFIG_FILE = "adapter_config.json";

    /** Alternative adapter filename (PEFT convention). */
    private static final String PEFT_ADAPTER_FILE = "adapter.safetensors";

    @Inject
    SafetensorLoaderFacade safetensorLoader;

    @Inject
    ObjectMapper objectMapper;

    /** Cache of loaded adapters by their absolute path. */
    private final Map<Path, LoadedAdapter> adapterCache = new ConcurrentHashMap<>();

    // ─────────────────────────────────────────────────────────────────────────
    // LoadedAdapter
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A loaded LoRA adapter with weights and configuration.
     *
     * @param adapterPath     the directory or file path this adapter was loaded from
     * @param weights         map of tensor name → LibTorch TorchTensor (lora_A, lora_B matrices)
     * @param config          adapter configuration (rank, alpha, target modules, etc.)
     * @param loadedAt        timestamp when this adapter was loaded (epoch millis)
     */
    public static record LoadedAdapter(
            Path adapterPath,
            Map<String, TorchTensor> weights,
            AdapterConfig config,
            long loadedAt
    ) implements AutoCloseable {

        /**
         * Get the LoRA A matrix for a specific base module.
         *
         * @param baseModuleName the base module name (e.g., "model.layers.0.self_attn.q_proj")
         * @return the lora_A tensor, or empty if not present
         */
        public Optional<TorchTensor> getLoraA(String baseModuleName) {
            TorchTensor tensor = weights.get(baseModuleName + ".lora_A.weight");
            if (tensor == null) {
                tensor = weights.get(baseModuleName + ".lora_A");
            }
            return Optional.ofNullable(tensor);
        }

        /**
         * Get the LoRA B matrix for a specific base module.
         *
         * @param baseModuleName the base module name (e.g., "model.layers.0.self_attn.q_proj")
         * @return the lora_B tensor, or empty if not present
         */
        public Optional<TorchTensor> getLoraB(String baseModuleName) {
            TorchTensor tensor = weights.get(baseModuleName + ".lora_B.weight");
            if (tensor == null) {
                tensor = weights.get(baseModuleName + ".lora_B");
            }
            return Optional.ofNullable(tensor);
        }

        /**
         * Get both LoRA matrices (A, B) for a specific base module.
         *
         * @param baseModuleName the base module name
         * @return an optional containing both tensors if present, empty otherwise
         */
        public Optional<LoraPair> getLoraPair(String baseModuleName) {
            Optional<TorchTensor> loraA = getLoraA(baseModuleName);
            Optional<TorchTensor> loraB = getLoraB(baseModuleName);
            if (loraA.isPresent() && loraB.isPresent()) {
                return Optional.of(new LoraPair(loraA.get(), loraB.get()));
            }
            return Optional.empty();
        }

        /**
         * Check if this adapter contains weights for a specific module.
         *
         * @param baseModuleName the base module name to check
         * @return true if both lora_A and lora_B are present
         */
        public boolean hasModule(String baseModuleName) {
            return getLoraPair(baseModuleName).isPresent();
        }

        /**
         * Get all base module names that have LoRA weights in this adapter.
         *
         * @return set of module names
         */
        public Set<String> getModuleNames() {
            Set<String> modules = new HashSet<>();
            for (String key : weights.keySet()) {
                if (key.endsWith(".lora_A.weight") || key.endsWith(".lora_A")) {
                    String baseName = key
                            .replace(".lora_A.weight", "")
                            .replace(".lora_A", "");
                    modules.add(baseName);
                }
            }
            return modules;
        }

        /**
         * Get the adapter rank (dimension of the low-rank decomposition).
         *
         * @return the rank value from config
         */
        public int rank() {
            return config.rank();
        }

        /**
         * Get the alpha scaling factor.
         *
         * @return the alpha value
         */
        public float alpha() {
            return config.alpha();
        }

        /**
         * Get the effective scaling factor (alpha / rank).
         *
         * @return the scaling factor to apply during LoRA computation
         */
        public float scalingFactor() {
            return config.alpha() / config.rank();
        }

        @Override
        public void close() {
            for (TorchTensor tensor : weights.values()) {
                try {
                    tensor.close();
                } catch (Exception e) {
                    log.warnf(e, "LoraAdapter: failed to close tensor");
                }
            }
        }
    }

    /**
     * A pair of LoRA matrices (A, B) for a single module.
     *
     * @param a the down-projection matrix (r × d)
     * @param b the up-projection matrix (k × r)
     */
    public record LoraPair(TorchTensor a, TorchTensor b) {
    }

    /**
     * LoRA adapter configuration parsed from adapter_config.json.
     *
     * @param rank             the LoRA rank (dimension of low-rank decomposition)
     * @param alpha            the alpha scaling factor
     * @param dropout          dropout rate applied during training (typically 0.0 for inference)
     * @param bias             whether bias terms are included ("none", "all", "lora_only")
     * @param targetModules    list of module names/types to apply LoRA to
     * @param taskType         the task type (e.g., "CAUSAL_LM", "SEQ_2_SEQ_LM")
     * @param inferenceMode    whether adapter was exported for inference-only
     * @param baseModelName    name or path of the base model this adapter was trained on
     * @param properties       additional custom properties from config
     */
    public static record AdapterConfig(
            int rank,
            float alpha,
            float dropout,
            String bias,
            List<String> targetModules,
            String taskType,
            boolean inferenceMode,
            String baseModelName,
            Map<String, Object> properties
    ) {
        /**
         * Compute the effective scaling factor: alpha / rank.
         * This is the factor applied during LoRA forward pass.
         */
        public float scalingFactor() {
            return alpha / rank;
        }

        /**
         * Check if a module name matches any target module pattern.
         *
         * @param moduleName the module name to check
         * @return true if this module should receive LoRA weights
         */
        public boolean matchesTarget(String moduleName) {
            if (targetModules == null || targetModules.isEmpty()) {
                return true; // Apply to all if no targets specified
            }
            for (String target : targetModules) {
                if (moduleName.contains(target)) {
                    return true;
                }
            }
            return false;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load a LoRA adapter from the specified path.
     *
     * <p>
     * The path can be:
     * <ul>
     *   <li>A directory containing adapter_model.safetensors and adapter_config.json</li>
     *   <li>A directory containing adapter.safetensors (PEFT convention)</li>
     *   <li>A direct path to a .safetensors file</li>
     * </ul>
     *
     * @param adapterPath the path to the adapter
     * @return a loaded adapter with weights and configuration
     * @throws IllegalArgumentException if the path is invalid or missing required files
     */
    public LoadedAdapter load(Path adapterPath) {
        Objects.requireNonNull(adapterPath, "adapterPath must not be null");
        Path normalizedPath = adapterPath.toAbsolutePath().normalize();

        // Check cache first
        LoadedAdapter cached = adapterCache.get(normalizedPath);
        if (cached != null) {
            log.debugf("LoraAdapter: returning cached adapter from %s", normalizedPath);
            return cached;
        }

        log.infof("LoraAdapter: loading adapter from %s", normalizedPath);

        try {
            // Resolve the actual adapter file
            Path adapterFile = resolveAdapterFile(normalizedPath);
            if (!Files.isRegularFile(adapterFile)) {
                throw new IllegalArgumentException(
                        "Adapter file not found at " + adapterFile);
            }

            // Load weights
            Map<String, TorchTensor> weights = loadAdapterWeights(adapterFile);
            if (weights.isEmpty()) {
                throw new IllegalArgumentException(
                        "No LoRA weights found in " + adapterFile);
            }

            // Load configuration
            AdapterConfig config = loadAdapterConfig(normalizedPath);

            // Validate rank consistency
            validateRankConsistency(weights, config.rank());

            // Create and cache
            LoadedAdapter adapter = new LoadedAdapter(
                    normalizedPath,
                    weights,
                    config,
                    System.currentTimeMillis()
            );

            adapterCache.put(normalizedPath, adapter);

            log.infof("LoraAdapter: loaded adapter from %s — %d tensors, rank=%d, alpha=%.2f",
                    normalizedPath.getFileName(), weights.size(), config.rank(), config.alpha());

            return adapter;

        } catch (IOException e) {
            throw new RuntimeException("Failed to load LoRA adapter from " + normalizedPath, e);
        }
    }

    /**
     * Check if an adapter is already loaded in the cache.
     *
     * @param adapterPath the adapter path to check
     * @return true if the adapter is cached
     */
    public boolean isLoaded(Path adapterPath) {
        return adapterCache.containsKey(adapterPath.toAbsolutePath().normalize());
    }

    /**
     * Get a cached adapter without reloading.
     *
     * @param adapterPath the adapter path
     * @return the cached adapter, or empty if not loaded
     */
    public Optional<LoadedAdapter> getCached(Path adapterPath) {
        return Optional.ofNullable(adapterCache.get(adapterPath.toAbsolutePath().normalize()));
    }

    /**
     * Unload an adapter from the cache and release its tensors.
     *
     * @param adapterPath the adapter path to unload
     */
    public void unload(Path adapterPath) {
        Path normalizedPath = adapterPath.toAbsolutePath().normalize();
        LoadedAdapter adapter = adapterCache.remove(normalizedPath);
        if (adapter != null) {
            adapter.close();
            log.infof("LoraAdapter: unloaded adapter from %s", normalizedPath);
        }
    }

    /**
     * Clear all cached adapters and release their tensors.
     */
    public void clearCache() {
        for (LoadedAdapter adapter : adapterCache.values()) {
            adapter.close();
        }
        adapterCache.clear();
        log.info("LoraAdapter: cleared all cached adapters");
    }

    /**
     * Get all currently cached adapter paths.
     *
     * @return set of adapter paths
     */
    public Set<Path> getCachedAdapterPaths() {
        return Set.copyOf(adapterCache.keySet());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — weight loading
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Resolve the actual adapter file from a directory or file path.
     */
    private Path resolveAdapterFile(Path adapterPath) throws IOException {
        if (Files.isRegularFile(adapterPath)) {
            String name = adapterPath.getFileName().toString().toLowerCase();
            if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
                return adapterPath;
            }
            throw new IllegalArgumentException(
                    "File must be a .safetensors file: " + adapterPath);
        }

        if (Files.isDirectory(adapterPath)) {
            // Try standard naming convention
            Path standard = adapterPath.resolve(DEFAULT_ADAPTER_FILE);
            if (Files.isRegularFile(standard)) {
                return standard;
            }

            // Try PEFT naming convention
            Path peft = adapterPath.resolve(PEFT_ADAPTER_FILE);
            if (Files.isRegularFile(peft)) {
                return peft;
            }

            // Search for any .safetensors file in the directory
            try (var stream = Files.list(adapterPath)) {
                Optional<Path> found = stream
                        .filter(p -> {
                            String name = p.getFileName().toString().toLowerCase();
                            return name.endsWith(".safetensors") && !name.contains("index");
                        })
                        .findFirst();
                if (found.isPresent()) {
                    return found.get();
                }
            }

            throw new IllegalArgumentException(
                    "No adapter file found in " + adapterPath +
                            " (expected " + DEFAULT_ADAPTER_FILE + " or " + PEFT_ADAPTER_FILE + ")");
        }

        throw new IllegalArgumentException("Invalid adapter path: " + adapterPath);
    }

    /**
     * Load all LoRA weight tensors from a SafeTensors file.
     */
    private Map<String, TorchTensor> loadAdapterWeights(Path adapterFile) throws IOException {
        Map<String, TorchTensor> weights = new HashMap<>();

        try (SafetensorShardSession session = safetensorLoader.open(adapterFile)) {
            Set<String> tensorNames = session.tensorNames();
            log.debugf("LoraAdapter: found %d tensors in %s", tensorNames.size(), adapterFile.getFileName());

            for (String name : tensorNames) {
                // Only load LoRA matrices (filter out any non-LoRA tensors)
                if (isLoraTensor(name)) {
                    SafetensorTensor st = session.tensor(name);
                    TorchTensor tensor = bridgeToLibTorch(st);
                    weights.put(name, tensor);
                }
            }
        } catch (tech.kayys.gollek.spi.exception.ProviderException e) {
            throw new IOException("Failed to load adapter weights from " + adapterFile, e);
        }

        return weights;
    }

    /**
     * Check if a tensor name represents a LoRA weight matrix.
     */
    private boolean isLoraTensor(String name) {
        return name.endsWith(".lora_A.weight")
                || name.endsWith(".lora_A")
                || name.endsWith(".lora_B.weight")
                || name.endsWith(".lora_B");
    }

    /**
     * Bridge a SafeTensor to a LibTorch TorchTensor.
     * Uses the SafetensorWeightBridge if available, otherwise creates a copy.
     */
    private TorchTensor bridgeToLibTorch(SafetensorTensor st) {
        // Try to use the weight bridge for zero-copy if available
        // For now, create a tensor from the float array
        // TODO: Integrate with SafetensorWeightBridge for zero-copy
        float[] data = st.toFloatArray();
        long[] shape = st.shape();
        return TorchTensor.fromFloatArray(data, shape);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — config loading
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load and parse adapter_config.json.
     */
    private AdapterConfig loadAdapterConfig(Path adapterDir) throws IOException {
        Path configFile = adapterDir.resolve(DEFAULT_CONFIG_FILE);

        if (!Files.isRegularFile(configFile)) {
            log.warnf("LoraAdapter: no adapter_config.json found at %s, using defaults", configFile);
            return createDefaultConfig();
        }

        log.debugf("LoraAdapter: loading adapter config from %s", configFile);

        try {
            JsonNode root = objectMapper.readTree(configFile.toFile());

            // Extract PEFT LoRA config
            int rank = extractInt(root, "r").orElse(extractInt(root, "rank").orElse(16));
            float alpha = extractFloat(root, "lora_alpha").orElse((float) rank);
            float dropout = extractFloat(root, "lora_dropout").orElse(0.0f);
            String bias = extractString(root, "bias").orElse("none");
            String taskType = extractString(root, "task_type").orElse(null);
            boolean inferenceMode = extractBoolean(root, "inference_mode").orElse(true);
            String baseModelName = extractString(root, "base_model_name_or_path").orElse(null);

            // Extract target modules
            List<String> targetModules = extractStringArray(root, "target_modules");

            // Extract any additional properties
            Map<String, Object> properties = extractAdditionalProperties(root);

            return new AdapterConfig(
                    rank,
                    alpha,
                    dropout,
                    bias,
                    targetModules,
                    taskType,
                    inferenceMode,
                    baseModelName,
                    properties
            );

        } catch (IOException e) {
            log.warnf(e, "LoraAdapter: failed to parse adapter_config.json, using defaults");
            return createDefaultConfig();
        }
    }

    /**
     * Create a default adapter config when no config file is present.
     */
    private AdapterConfig createDefaultConfig() {
        return new AdapterConfig(
                16,  // default rank
                16.0f,  // default alpha (scaling = 1.0)
                0.0f,  // no dropout
                "none",  // no bias
                List.of(),  // apply to all modules
                null,
                true,
                null,
                Map.of()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — validation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Validate that all LoRA matrices have consistent rank dimensions.
     */
    private void validateRankConsistency(Map<String, TorchTensor> weights, int expectedRank) {
        for (Map.Entry<String, TorchTensor> entry : weights.entrySet()) {
            String name = entry.getKey();
            TorchTensor tensor = entry.getValue();
            long[] shape = tensor.shape();

            // LoRA A matrix: (rank, hidden_dim) — rank is first dimension
            // LoRA B matrix: (output_dim, rank) — rank is second dimension
            if (name.endsWith(".lora_A.weight") || name.endsWith(".lora_A")) {
                if (shape.length != 2) {
                    throw new IllegalArgumentException(
                            "LoRA A tensor '" + name + "' must be 2D, got shape " + Arrays.toString(shape));
                }
                if (shape[0] != expectedRank) {
                    log.warnf("LoraAdapter: LoRA A tensor '%s' has rank %d, expected %d",
                            name, shape[0], expectedRank);
                }
            } else if (name.endsWith(".lora_B.weight") || name.endsWith(".lora_B")) {
                if (shape.length != 2) {
                    throw new IllegalArgumentException(
                            "LoRA B tensor '" + name + "' must be 2D, got shape " + Arrays.toString(shape));
                }
                if (shape[1] != expectedRank) {
                    log.warnf("LoraAdapter: LoRA B tensor '%s' has rank %d, expected %d",
                            name, shape[1], expectedRank);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — JSON helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static Optional<Integer> extractInt(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isInt()) {
            return Optional.of(value.asInt());
        }
        return Optional.empty();
    }

    private static Optional<Float> extractFloat(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isNumber()) {
            return Optional.of((float) value.asDouble());
        }
        return Optional.empty();
    }

    private static Optional<Boolean> extractBoolean(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isBoolean()) {
            return Optional.of(value.asBoolean());
        }
        return Optional.empty();
    }

    private static Optional<String> extractString(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isTextual()) {
            return Optional.of(value.asText());
        }
        return Optional.empty();
    }

    private static List<String> extractStringArray(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value != null && value.isArray()) {
            List<String> result = new ArrayList<>();
            for (JsonNode element : value) {
                if (element.isTextual()) {
                    result.add(element.asText());
                }
            }
            return result;
        }
        return List.of();
    }

    private static Map<String, Object> extractAdditionalProperties(JsonNode root) {
        Map<String, Object> properties = new HashMap<>();
        Set<String> knownFields = Set.of(
                "r", "rank", "lora_alpha", "lora_dropout", "bias",
                "target_modules", "task_type", "inference_mode",
                "base_model_name_or_path", "peft_type");

        for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            if (!knownFields.contains(key)) {
                properties.put(key, toJsonValue(entry.getValue()));
            }
        }
        return properties;
    }

    private static Object toJsonValue(JsonNode node) {
        if (node.isTextual()) return node.asText();
        if (node.isInt()) return node.asInt();
        if (node.isLong()) return node.asLong();
        if (node.isFloatingPointNumber()) return (float) node.asDouble();
        if (node.isDouble()) return node.asDouble();
        if (node.isBoolean()) return node.asBoolean();
        if (node.isArray()) {
            List<Object> list = new ArrayList<>();
            for (JsonNode element : node) {
                list.add(toJsonValue(element));
            }
            return list;
        }
        if (node.isObject()) {
            Map<String, Object> map = new HashMap<>();
            for (Iterator<Map.Entry<String, JsonNode>> it = node.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                map.put(entry.getKey(), toJsonValue(entry.getValue()));
            }
            return map;
        }
        return null;
    }
}
