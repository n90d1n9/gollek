package tech.kayys.gollek.quantizer.gptq;

import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads GPTQ-quantized models from .safetensors files using the
 * safetensor-loader module's infrastructure (SafetensorHeaderParser, FFM API).
 *
 * <p>Handles both single-file and multi-shard models:
 * <pre>
 * model.safetensors                          → single-file
 * model-00001-of-00003.safetensors           → sharded
 * </pre>
 *
 * <p>The loader:
 * <ol>
 *   <li>Discovers all .safetensors files in a directory</li>
 *   <li>Parses headers via {@link SafetensorHeaderParser} to build a tensor→shard index</li>
 *   <li>Auto-detects GPTQ config from tensor naming patterns</li>
 *   <li>Loads quantized tensors into off-heap FFM memory</li>
 *   <li>Groups tensors into {@link QuantizedLayer} instances</li>
 * </ol>
 *
 * <p><b>GPTQ Tensor Naming (AutoGPTQ convention):</b>
 * <pre>
 * model.layers.N.self_attn.q_proj.qweight
 * model.layers.N.self_attn.q_proj.qzeros
 * model.layers.N.self_attn.q_proj.scales
 * model.layers.N.self_attn.q_proj.g_idx   (optional)
 * model.layers.N.self_attn.q_proj.bias    (optional)
 * </pre>
 */
public class GPTQLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GPTQLoader.class);

    // Pattern to extract layer prefix from tensor name
    private static final Pattern LAYER_PREFIX_PATTERN = Pattern.compile("^(.+?)\\.(qweight|qzeros|scales|g_idx|bias)$");

    private final Path modelDir;
    private final GPTQConfig config;
    private final MemoryAllocator allocator;

    /** Standalone safetensor file loader (no CDI required). */
    private final GPTQSafetensorFileLoader fileLoader;

    /** Map from tensor name → shard wrapper that holds it */
    private final Map<String, GPTQSafetensorShard> tensorToShard = new LinkedHashMap<>();

    /** Loaded layers, keyed by layer prefix */
    private final Map<String, QuantizedLayer> layers = new LinkedHashMap<>();

    /** All open shard wrappers (kept alive for off-heap segment lifetimes) */
    private final List<GPTQSafetensorShard> openShards = new ArrayList<>();

    /** Combined metadata from all shards */
    private Map<String, String> modelMetadata = new HashMap<>();

    public GPTQLoader(Path modelDir, GPTQConfig config) {
        this.modelDir = modelDir;
        this.config = config;
        this.allocator = new MemoryAllocator();
        this.fileLoader = new GPTQSafetensorFileLoader();
    }

    // ── Load Pipeline ─────────────────────────────────────────────────────────

    /**
     * Loads the complete model. Steps:
     * 1. Discover shards
     * 2. Parse all headers
     * 3. Build tensor index
     * 4. Load and group quantized tensors into layers
     *
     * @return this loader (fluent API)
     */
    public GPTQLoader load() throws IOException {
        log.info("Loading GPTQ model from: {}", modelDir);

        List<Path> shards = discoverShards();
        log.info("Discovered {} shard(s)", shards.size());

        parseAllHeaders(shards);
        log.info("Indexed {} tensors across all shards", tensorToShard.size());

        Set<String> layerPrefixes = discoverLayerPrefixes();
        log.info("Discovered {} quantized layers", layerPrefixes.size());

        loadLayers(layerPrefixes);
        log.info("Loaded {} layers ({:.2f} MB off-heap)",
                layers.size(), allocator.getTotalAllocated() / (1024.0 * 1024));

        return this;
    }

    // ── Shard Discovery ───────────────────────────────────────────────────────

    /**
     * Discovers all .safetensors files in the model directory.
     * Handles both single-file and sharded (HuggingFace) layouts.
     */
    private List<Path> discoverShards() throws IOException {
        var dir = modelDir.toFile();
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IOException("Model directory not found: " + modelDir);
        }

        var files = dir.listFiles(f -> f.isFile() && f.getName().endsWith(".safetensors"));

        if (files == null || files.length == 0) {
            throw new IOException("No .safetensors files found in: " + modelDir);
        }

        return Arrays.stream(files)
                .map(java.io.File::toPath)
                .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                .toList();
    }

    // ── Header Parsing ────────────────────────────────────────────────────────

    /**
     * Parses all shard headers and builds the tensor-to-shard index.
     * Shards are kept open because their MemorySegments back the tensor data.
     */
    private void parseAllHeaders(List<Path> shards) throws IOException {
        for (Path shard : shards) {
            log.info("Parsing shard: {}", shard.getFileName());

            // Use the standalone file loader which wraps SafetensorHeaderParser + FFM
            GPTQSafetensorShard gptqShard = fileLoader.loadShard(shard);
            openShards.add(gptqShard);

            SafetensorHeader header = gptqShard.getHeader();
            modelMetadata.putAll(header.fileMetadata());

            for (String tensorName : header.tensors().keySet()) {
                tensorToShard.put(tensorName, gptqShard);
            }

            log.debug("Shard '{}': {} tensors", shard.getFileName(), header.tensorCount());
        }
    }

    // ── Layer Discovery ───────────────────────────────────────────────────────

    /**
     * Identifies unique layer prefixes from GPTQ tensor names.
     * e.g., "model.layers.0.self_attn.q_proj" from
     * "model.layers.0.self_attn.q_proj.qweight"
     */
    private Set<String> discoverLayerPrefixes() {
        Set<String> prefixes = new LinkedHashSet<>();

        for (String tensorName : tensorToShard.keySet()) {
            Matcher m = LAYER_PREFIX_PATTERN.matcher(tensorName);
            if (m.matches()) {
                prefixes.add(m.group(1));
            }
        }

        return prefixes;
    }

    // ── Layer Loading ─────────────────────────────────────────────────────────

    /**
     * Loads all discovered layers, copying tensor data into off-heap memory.
     */
    private void loadLayers(Set<String> layerPrefixes) {
        for (String prefix : layerPrefixes) {
            try {
                QuantizedLayer layer = loadLayer(prefix);
                layers.put(prefix, layer);
            } catch (Exception e) {
                log.warn("Failed to load layer '{}': {}", prefix, e.getMessage());
            }
        }
    }

    /**
     * Loads a single quantized layer by reading its component tensors.
     *
     * Required tensors: qweight, qzeros, scales
     * Optional tensors: g_idx, bias
     */
    private QuantizedLayer loadLayer(String prefix) {
        log.debug("Loading layer: {}", prefix);

        QuantizedLayer layer = new QuantizedLayer(prefix, config);

        // ── Load required tensors ─────────────────────────────────────────────
        String qweightName = prefix + ".qweight";
        String qzerosName = prefix + ".qzeros";
        String scalesName = prefix + ".scales";

        if (!tensorToShard.containsKey(qweightName)) {
            throw new IllegalStateException("Missing qweight for layer: " + prefix);
        }

        // qweight → INT32 off-heap segment
        MemorySegment qweightSeg = loadTensorOffHeap(qweightName);
        layer.setQweight(qweightSeg);

        // qzeros → INT32 off-heap segment
        if (tensorToShard.containsKey(qzerosName)) {
            layer.setQzeros(loadTensorOffHeap(qzerosName));
        }

        // scales → FP16 off-heap segment
        if (tensorToShard.containsKey(scalesName)) {
            layer.setScales(loadTensorOffHeap(scalesName));
        }

        // ── Load optional tensors ─────────────────────────────────────────────

        // g_idx (act-order group index)
        String gIdxName = prefix + ".g_idx";
        if (tensorToShard.containsKey(gIdxName)) {
            layer.setGIdx(loadTensorOffHeap(gIdxName));
            log.debug("Layer '{}' has g_idx (act-order)", prefix);
        }

        // bias
        String biasName = prefix + ".bias";
        if (tensorToShard.containsKey(biasName)) {
            layer.setBias(loadTensorOffHeap(biasName));
        }

        // ── Infer dimensions from tensor shapes ───────────────────────────────
        inferDimensions(layer, prefix);

        log.debug("Loaded: {}", layer);
        return layer;
    }

    /**
     * Copies a tensor from mmap into an aligned off-heap buffer.
     * This ensures SIMD-alignment requirements are met for Vector API ops.
     */
    private MemorySegment loadTensorOffHeap(String tensorName) {
        GPTQSafetensorShard shard = tensorToShard.get(tensorName);
        SafetensorTensorInfo info = shard.getHeader().tensor(tensorName);

        // Zero-copy slice from mmap (still aligned to file offsets)
        MemorySegment mmapSlice = shard.getTensorSegment(tensorName);

        // Copy to SIMD-aligned off-heap buffer for Vector API
        MemorySegment aligned = allocator.copyFrom(mmapSlice, tensorName);

        log.trace("Loaded tensor '{}': {} bytes, dtype={}", tensorName,
                info.byteLength(), info.dtype());

        return aligned;
    }

    /**
     * Infers inFeatures and outFeatures from qweight tensor shape.
     *
     * qweight shape (AutoGPTQ): [outFeatures/packFactor, inFeatures]
     */
    private void inferDimensions(QuantizedLayer layer, String prefix) {
        GPTQSafetensorShard shard = tensorToShard.get(prefix + ".qweight");
        if (shard == null)
            return;

        SafetensorTensorInfo qwInfo = shard.getHeader().tensor(prefix + ".qweight");
        if (qwInfo == null || qwInfo.shape() == null || qwInfo.rank() < 2)
            return;

        long[] shape = qwInfo.shape();
        long packedRows = shape[0]; // outFeatures / packFactor
        long inFeatures = shape[1];

        layer.setInFeatures((int) inFeatures);
        layer.setOutFeatures((int) (packedRows * config.elementsPerInt32()));
        layer.setQweightShape(new long[] { packedRows, inFeatures });

        // Infer scales shape
        GPTQSafetensorShard sp = tensorToShard.get(prefix + ".scales");
        if (sp != null) {
            SafetensorTensorInfo sInfo = sp.getHeader().tensor(prefix + ".scales");
            if (sInfo != null && sInfo.shape() != null) {
                layer.setScalesShape(sInfo.shape());
            }
        }
    }

    // ── Auto-Config Detection ─────────────────────────────────────────────────

    /**
     * Attempts to auto-detect GPTQ config from model metadata or tensor shapes.
     * Reads "bits", "group_size", "desc_act" from the safetensor __metadata__.
     */
    public static GPTQConfig autoDetectConfig(Path modelDir) throws IOException {
        var files = modelDir.toFile().listFiles(f -> f.isFile() && f.getName().endsWith(".safetensors"));

        if (files == null || files.length == 0) {
            log.warn("No safetensors found, using default GPTQ config");
            return GPTQConfig.gptq4bit();
        }

        // Use the standalone file loader for header-only parsing
        try (GPTQSafetensorFileLoader loader = new GPTQSafetensorFileLoader()) {
            SafetensorHeader header = loader.loadHeaderOnly(files[0].toPath());
            Map<String, String> meta = header.fileMetadata();
            log.info("Model metadata: {}", meta);

            int bits = parseInt(meta.getOrDefault("bits", "4"));
            int groupSize = parseInt(meta.getOrDefault("group_size", "128"));
            boolean act = "true".equalsIgnoreCase(meta.getOrDefault("desc_act", "false"));
            boolean sym = "true".equalsIgnoreCase(meta.getOrDefault("sym", "false"));

            GPTQConfig cfg = GPTQConfig.builder()
                    .bits(bits)
                    .groupSize(groupSize)
                    .actOrder(act)
                    .symmetric(sym)
                    .exllamaV2(false)
                    .dequantDtype("float32")
                    .build();
            log.info("Auto-detected config: {}", cfg);
            return cfg;
        }
    }

    private static int parseInt(String s) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public Map<String, QuantizedLayer> getLayers() {
        return Collections.unmodifiableMap(layers);
    }

    public QuantizedLayer getLayer(String name) {
        return layers.get(name);
    }

    public List<String> getLayerNames() {
        return new ArrayList<>(layers.keySet());
    }

    public GPTQConfig getConfig() {
        return config;
    }

    public Map<String, String> getModelMetadata() {
        return Collections.unmodifiableMap(modelMetadata);
    }

    public MemoryAllocator getAllocator() {
        return allocator;
    }

    public int getLayerCount() {
        return layers.size();
    }

    /** Returns total off-heap memory used for all tensor data */
    public long getTotalOffHeapBytes() {
        return allocator.getTotalAllocated();
    }

    /**
     * Prints a detailed summary of loaded layers.
     */
    public void printSummary() {
        System.out.println("=== GPTQ Model Summary ===");
        System.out.printf("Config: %s%n", config);
        System.out.printf("Layers: %d%n", layers.size());
        System.out.printf("Off-heap memory: %.2f MB%n",
                allocator.getTotalAllocated() / (1024.0 * 1024));
        System.out.println();
        System.out.printf("%-70s %6s %6s %6s %5s%n",
                "Layer", "InF", "OutF", "Groups", "Bias");
        System.out.println("-".repeat(100));
        layers.forEach((name, layer) -> System.out.printf("%-70s %6d %6d %6d %5s%n",
                name, layer.getInFeatures(), layer.getOutFeatures(),
                layer.numGroups(), layer.hasBias() ? "yes" : "no"));
    }

    @Override
    public void close() {
        log.info("Closing GPTQLoader: releasing {} shard(s) and off-heap memory", openShards.size());
        for (GPTQSafetensorShard s : openShards) {
            try {
                s.close();
            } catch (Exception e) {
                log.warn("Shard close error", e);
            }
        }
        try {
            fileLoader.close();
        } catch (Exception e) {
            log.warn("File loader close error", e);
        }
        allocator.close();
    }
}
