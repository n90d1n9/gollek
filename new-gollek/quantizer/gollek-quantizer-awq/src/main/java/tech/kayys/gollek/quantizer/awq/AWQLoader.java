package tech.kayys.gollek.quantizer.awq;

import tech.kayys.gollek.quantizer.awq.AWQConfig;
import tech.kayys.gollek.quantizer.awq.AWQLayer;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import tech.kayys.gollek.safetensor.loader.SafetensorHeader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads AWQ-quantized models from .safetensors files using the
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
 *   <li>Auto-detects AWQ config from tensor naming patterns</li>
 *   <li>Loads quantized tensors into off-heap FFM memory</li>
 *   <li>Groups tensors into {@link AWQLayer} instances</li>
 * </ol>
 *
 * <p><b>AWQ Tensor Naming (AutoAWQ convention):</b>
 * <pre>
 * model.layers.N.self_attn.q_proj.qweight
 * model.layers.N.self_attn.q_proj.qzeros
 * model.layers.N.self_attn.q_proj.scales
 * model.layers.N.self_attn.q_proj.bias   (optional)
 * </pre>
 *
 * <p><b>AWQ vs GPTQ Layout Difference:</b>
 * AWQ qweight shape: [inF/pack, outF] — groups along INPUT dimension
 * GPTQ qweight shape: [outF/pack, inF] — groups along OUTPUT dimension
 */
public class AWQLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AWQLoader.class);

    /**
     * Matches the suffix of AWQ tensor names:
     * group(1) = layer prefix, group(2) = tensor type
     */
    private static final Pattern AWQ_TENSOR_PATTERN = Pattern.compile("^(.+?)\\.(qweight|qzeros|scales|bias)$");

    /** AWQ tensor suffixes (qweight must come before qzeros/scales during load) */
    private static final List<String> AWQ_SUFFIXES = List.of(".qweight", ".scales", ".qzeros", ".bias");

    private final Path modelDir;
    private final AWQConfig config;
    private final MemoryAllocator allocator;

    /** Standalone safetensor file loader (no CDI required). */
    private final AWQSafetensorFileLoader fileLoader;

    /** Map from tensor name → shard wrapper that holds it */
    private final Map<String, AWQSafetensorShard> tensorToShard = new LinkedHashMap<>();

    /** layer prefix → AWQLayer */
    private final Map<String, AWQLayer> layers = new LinkedHashMap<>();

    /** All open shard wrappers (kept alive for off-heap segment lifetimes) */
    private final List<AWQSafetensorShard> openShards = new ArrayList<>();

    /** Aggregated __metadata__ from all shards */
    private Map<String, String> modelMetadata = new HashMap<>();

    public AWQLoader(Path modelDir, AWQConfig config) {
        this.modelDir = modelDir;
        this.config = config;
        this.allocator = new MemoryAllocator();
        this.fileLoader = new AWQSafetensorFileLoader();
    }

    // ── Load Pipeline ─────────────────────────────────────────────────────────

    /**
     * Full load pipeline: discover → parse headers → load tensors → build layers.
     */
    public AWQLoader load() throws IOException {
        log.info("Loading AWQ model from: {}", modelDir);
        long t0 = System.currentTimeMillis();

        List<Path> shards = discoverShards();
        log.info("Found {} shard(s)", shards.size());

        parseHeaders(shards);
        log.info("Indexed {} tensors", tensorToShard.size());

        Set<String> prefixes = discoverLayerPrefixes();
        log.info("Found {} AWQ layers", prefixes.size());

        for (String prefix : prefixes) {
            try {
                AWQLayer layer = loadLayer(prefix);
                layers.put(prefix, layer);
            } catch (Exception e) {
                log.warn("Skipping layer '{}': {}", prefix, e.getMessage());
            }
        }

        log.info("Loaded {} layers in {} ms, off-heap = {:.2f} MB",
                layers.size(),
                System.currentTimeMillis() - t0,
                allocator.getTotalAllocated() / 1_048_576.0);

        return this;
    }

    // ── Shard Discovery ───────────────────────────────────────────────────────

    private List<Path> discoverShards() throws IOException {
        var files = modelDir.toFile().listFiles(
                f -> f.isFile() && f.getName().endsWith(".safetensors"));

        if (files == null || files.length == 0)
            throw new IOException("No .safetensors files in: " + modelDir);

        return Arrays.stream(files)
                .map(java.io.File::toPath)
                .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                .toList();
    }

    // ── Header Parsing ────────────────────────────────────────────────────────

    private void parseHeaders(List<Path> shards) throws IOException {
        for (Path shard : shards) {
            log.info("Parsing shard: {}", shard.getFileName());

            // Use the standalone file loader which wraps SafetensorHeaderParser + FFM
            AWQSafetensorShard awqShard = fileLoader.loadShard(shard);
            openShards.add(awqShard);

            SafetensorHeader header = awqShard.getHeader();
            modelMetadata.putAll(header.fileMetadata());

            for (String tensorName : header.tensors().keySet()) {
                tensorToShard.put(tensorName, awqShard);
            }

            log.debug("Shard '{}': {} tensors", shard.getFileName(),
                    header.tensorCount());
        }
    }

    // ── Layer Prefix Discovery ────────────────────────────────────────────────

    /**
     * Scans tensor names for AWQ layer prefixes.
     * A prefix is AWQ if it has at least "qweight" and "scales".
     */
    private Set<String> discoverLayerPrefixes() {
        // Collect all prefixes and count how many AWQ tensors they have
        Map<String, Integer> prefixHits = new LinkedHashMap<>();
        for (String name : tensorToShard.keySet()) {
            Matcher m = AWQ_TENSOR_PATTERN.matcher(name);
            if (m.matches()) {
                prefixHits.merge(m.group(1), 1, Integer::sum);
            }
        }

        // A layer needs at least qweight + scales (2 hits minimum)
        Set<String> result = new LinkedHashSet<>();
        for (var e : prefixHits.entrySet()) {
            if (e.getValue() >= 2
                    && tensorToShard.containsKey(e.getKey() + ".qweight")
                    && tensorToShard.containsKey(e.getKey() + ".scales")) {
                result.add(e.getKey());
            }
        }
        return result;
    }

    // ── Layer Loading ─────────────────────────────────────────────────────────

    /**
     * Loads a single AWQ layer's tensors into SIMD-aligned off-heap memory.
     */
    private AWQLayer loadLayer(String prefix) {
        AWQLayer layer = new AWQLayer(prefix, config);

        // Required: qweight
        layer.setQweight(loadTensor(prefix + ".qweight"));

        // Required: scales
        layer.setScales(loadTensor(prefix + ".scales"));

        // Optional: qzeros (absent in symmetric AWQ)
        if (tensorToShard.containsKey(prefix + ".qzeros")) {
            layer.setQzeros(loadTensor(prefix + ".qzeros"));
        }

        // Optional: bias
        if (tensorToShard.containsKey(prefix + ".bias")) {
            layer.setBias(loadTensor(prefix + ".bias"));
        }

        // Infer dimensions from tensor shapes
        inferDimensions(layer, prefix);

        log.debug("Loaded: {}", layer);
        return layer;
    }

    /**
     * Copies a tensor from memory-mapped file into a SIMD-aligned off-heap buffer.
     */
    private MemorySegment loadTensor(String name) {
        AWQSafetensorShard shard = tensorToShard.get(name);
        if (shard == null)
            throw new NoSuchElementException("Tensor not found: " + name);

        MemorySegment mmap = shard.getTensorSegment(name);
        return allocator.copyFrom(mmap, name); // SIMD-aligned copy
    }

    /**
     * Infers inFeatures and outFeatures from qweight shape.
     *
     * AWQ qweight shape: [inF / packFactor, outF]
     * rows = inF / pack → inF = rows × pack
     * cols = outF
     */
    private void inferDimensions(AWQLayer layer, String prefix) {
        AWQSafetensorShard shard = tensorToShard.get(prefix + ".qweight");
        if (shard == null)
            return;

        List<Long> shape = shard.getTensorShape(prefix + ".qweight");
        if (shape == null || shape.size() < 2)
            return;

        long packedRows = shape.get(0);
        long outFeatures = shape.get(1);
        long inFeatures = packedRows * config.packFactor();

        layer.setInFeatures((int) inFeatures);
        layer.setOutFeatures((int) outFeatures);
        layer.setQweightShape(new long[] { packedRows, outFeatures });

        // Scales shape [numGroups, outF]
        if (tensorToShard.containsKey(prefix + ".scales")) {
            List<Long> scalesShape = shard.getTensorShape(prefix + ".scales");
            if (scalesShape != null) {
                layer.setScalesShape(scalesShape.stream().mapToLong(Long::longValue).toArray());
            }
        }

        // Zeros shape [numGroups, outF/pack]
        if (tensorToShard.containsKey(prefix + ".qzeros")) {
            List<Long> zerosShape = shard.getTensorShape(prefix + ".qzeros");
            if (zerosShape != null) {
                layer.setQzerosShape(zerosShape.stream().mapToLong(Long::longValue).toArray());
            }
        }
    }

    // ── Auto-Detection ────────────────────────────────────────────────────────

    /**
     * Auto-detects AWQ configuration from model __metadata__.
     *
     * AutoAWQ writes:
     * "w_bit": "4", "q_group_size": "128",
     * "zero_point": "true", "version": "GEMM"
     */
    public static AWQConfig autoDetectConfig(Path modelDir) throws IOException {
        var files = modelDir.toFile().listFiles(
                f -> f.isFile() && f.getName().endsWith(".safetensors"));

        if (files == null || files.length == 0) {
            log.warn("No safetensors found, using default AWQ config");
            return AWQConfig.awq4bit();
        }

        try (AWQSafetensorFileLoader fileLoader = new AWQSafetensorFileLoader()) {
            SafetensorHeader header = fileLoader.loadHeaderOnly(files[0].toPath());
            Map<String, String> meta = header.fileMetadata();
            log.info("Model metadata: {}", meta);

            int bits = parseIntOr(meta.getOrDefault("w_bit",
                    meta.getOrDefault("bits", "4")), 4);
            int groupSize = parseIntOr(meta.getOrDefault("q_group_size",
                    meta.getOrDefault("group_size", "128")), 128);
            boolean zeros = !"false".equalsIgnoreCase(
                    meta.getOrDefault("zero_point", "true"));

            String versionStr = meta.getOrDefault("version", "GEMM").toUpperCase();
            AWQConfig.KernelFormat fmt = switch (versionStr) {
                case "GEMV" -> AWQConfig.KernelFormat.GEMV;
                case "MARLIN" -> AWQConfig.KernelFormat.MARLIN;
                default -> AWQConfig.KernelFormat.GEMM;
            };

            AWQConfig cfg = new AWQConfig(bits, groupSize, fmt, zeros, "float32",
                    false, 128, 2048, true, null);
            log.info("Auto-detected AWQ config: {}", cfg);
            return cfg;
        }
    }

    private static int parseIntOr(String s, int fallback) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public Map<String, AWQLayer> getLayers() {
        return Collections.unmodifiableMap(layers);
    }

    public AWQLayer getLayer(String name) {
        return layers.get(name);
    }

    public List<String> getLayerNames() {
        return new ArrayList<>(layers.keySet());
    }

    public AWQConfig getConfig() {
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

    public long getTotalOffHeapBytes() {
        return allocator.getTotalAllocated();
    }

    public void printSummary() {
        System.out.println("=== AWQ Model Summary ===");
        System.out.printf("Config: %s%n", config);
        System.out.printf("Layers: %d%n", layers.size());
        System.out.printf("Off-heap: %.2f MB%n",
                allocator.getTotalAllocated() / 1_048_576.0);
        System.out.println();
        System.out.printf("%-65s %6s %6s %6s %5s %5s%n",
                "Layer", "InF", "OutF", "Groups", "Zeros", "Bias");
        System.out.println("-".repeat(100));
        layers.forEach((name, layer) -> System.out.printf("%-65s %6d %6d %6d %5s %5s%n",
                name,
                layer.getInFeatures(),
                layer.getOutFeatures(),
                layer.numGroups(),
                layer.hasZeros() ? "yes" : "no",
                layer.hasBias() ? "yes" : "no"));
    }

    @Override
    public void close() {
        log.info("Closing AWQLoader: {} shards, {:.2f} MB off-heap",
                openShards.size(), allocator.getTotalAllocated() / 1_048_576.0);
        openShards.forEach(s -> {
            try {
                s.close();
            } catch (Exception e) {
                log.warn("Shard close: {}", e.getMessage());
            }
        });
        allocator.close();
    }
}
