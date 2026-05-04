package tech.kayys.gollek.quantizer.autoround;

import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import tech.kayys.gollek.quantizer.autoround.AutoRoundLayer;
import tech.kayys.gollek.quantizer.autoround.AutoRoundDequantizer;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads AutoRound-quantized models from .safetensors files using the FFM API.
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │ AutoRound Tensor Naming Conventions (two variants) │
 * ├──────────────────────────┬─────────────────────────────────────────────┤
 * │ AutoRound Native │ AutoRound GPTQ-Compat │
 * ├──────────────────────────┼─────────────────────────────────────────────┤
 * │ layer.weight (INT32) │ layer.qweight (INT32) │
 * │ layer.scale (FP32) │ layer.scales (FP16) │
 * │ layer.zp (INT32) │ layer.qzeros (INT32 packed) │
 * │ layer.bias (FP32) │ layer.bias (FP16 or FP32) │
 * └──────────────────────────┴─────────────────────────────────────────────┘
 *
 * AutoRound metadata keys (from __metadata__):
 * "quant_method": "autoround" or "auto_round"
 * "bits": "4"
 * "group_size": "128"
 * "sym": "False" / "True"
 * "backend": "autoround:exllamav2" | "autoround:marlin" | ...
 *
 * Format detection heuristic:
 * - If "scale" (FP32) tensor exists → AutoRound native
 * - If "scales" (FP16) + "qweight" tensor exists → GPTQ-compat
 * - "weight" dtype == INT32 → AutoRound native
 * - "qweight" dtype == INT32 → GPTQ-compat
 *
 * For GPTQ-compat exports, the loader normalises:
 * - FP16 scales → FP32 (transpose from GPTQ [numGroups, outF] to AR [outF,
 * numGroups])
 * - Packed INT32 qzeros → plain INT32 (unpacked, +1 bias removed)
 */
public class AutoRoundLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AutoRoundLoader.class);

    // ── Tensor name patterns for both naming conventions ──────────────────────

    /** Native: group(1) = prefix, group(2) = type (weight|scale|zp|bias) */
    private static final Pattern NATIVE_PATTERN = Pattern.compile("^(.+?)\\.(weight|scale|zp|bias)$");

    /**
     * GPTQ-compat: group(1) = prefix, group(2) = type (qweight|scales|qzeros|bias)
     */
    private static final Pattern GPTQ_COMPAT_PATTERN = Pattern.compile("^(.+?)\\.(qweight|scales|qzeros|bias)$");

    // ── State ─────────────────────────────────────────────────────────────────

    private final Path modelDir;
    private final AutoRoundConfig config;
    private final MemoryAllocator allocator;

    /** tensor name → shard parser (kept alive for MemorySegment lifetimes) */
    private final Map<String, SafetensorParser> tensorIndex = new LinkedHashMap<>();

    /** layer prefix → loaded layer */
    private final Map<String, AutoRoundLayer> layers = new LinkedHashMap<>();

    /** Open parsers (closed in AutoCloseable.close()) */
    private final List<SafetensorParser> openParsers = new ArrayList<>();

    /** Merged __metadata__ from all shards */
    private final Map<String, String> modelMetadata = new HashMap<>();

    /** Which naming convention was detected */
    private NamingConvention detectedConvention = NamingConvention.UNKNOWN;

    private enum NamingConvention {
        NATIVE, GPTQ_COMPAT, UNKNOWN
    }

    public AutoRoundLoader(Path modelDir, AutoRoundConfig config) {
        this.modelDir = modelDir;
        this.config = config;
        this.allocator = new MemoryAllocator();
    }

    // ── Load Pipeline ─────────────────────────────────────────────────────────

    /**
     * Full pipeline: shard discovery → header parsing → layer loading.
     */
    public AutoRoundLoader load() throws IOException {
        log.info("Loading AutoRound model from: {}", modelDir);
        long t0 = System.currentTimeMillis();

        List<Path> shards = discoverShards();
        log.info("Found {} shard(s)", shards.size());

        parseHeaders(shards);
        log.info("Indexed {} tensors", tensorIndex.size());

        detectedConvention = detectNamingConvention();
        log.info("Detected naming convention: {}", detectedConvention);

        Set<String> prefixes = discoverLayerPrefixes();
        log.info("Found {} AutoRound layers", prefixes.size());

        for (String prefix : prefixes) {
            try {
                AutoRoundLayer layer = loadLayer(prefix);
                layers.put(prefix, layer);
            } catch (Exception e) {
                log.warn("Skipping layer '{}': {}", prefix, e.getMessage());
            }
        }

        log.info("Loaded {} layers in {} ms, off-heap = {:.2f} MB",
                layers.size(), System.currentTimeMillis() - t0,
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
            SafetensorParser parser = new SafetensorParser(shard);
            openParsers.add(parser);
            modelMetadata.putAll(parser.getHeader().fileMetadata());
            parser.getHeader().tensors().keySet()
                    .forEach(name -> tensorIndex.put(name, parser));
            log.debug("Shard '{}': {} tensors", shard.getFileName(),
                    parser.getHeader().tensorCount());
        }
    }

    // ── Convention Detection ──────────────────────────────────────────────────

    private NamingConvention detectNamingConvention() {
        // Check metadata first (most reliable)
        String backend = modelMetadata.getOrDefault("backend", "");
        if (backend.startsWith("autoround:") || backend.equals("autoround")) {
            return NamingConvention.NATIVE;
        }

        // Count matching tensor names for each convention
        long nativeHits = tensorIndex.keySet().stream()
                .filter(n -> NATIVE_PATTERN.matcher(n).matches()
                        && n.endsWith(".weight") || n.endsWith(".scale"))
                .count();
        long gptqHits = tensorIndex.keySet().stream()
                .filter(n -> GPTQ_COMPAT_PATTERN.matcher(n).matches()
                        && n.endsWith(".qweight") || n.endsWith(".scales"))
                .count();

        // Check dtype of first "weight" or "qweight" tensor found
        for (var e : tensorIndex.entrySet()) {
            String name = e.getKey();
            SafetensorParser parser = e.getValue();
            SafetensorTensorInfo info = parser.getHeader().findTensor(name).orElse(null);
            if (info == null)
                continue;

            if (name.endsWith(".weight") && "I32".equals(info.dtype().jsonName())) {
                log.debug("Native AutoRound detected: .weight tensor is INT32");
                return NamingConvention.NATIVE;
            }
            if (name.endsWith(".qweight") && "I32".equals(info.dtype().jsonName())) {
                log.debug("GPTQ-compat AutoRound detected: .qweight tensor is INT32");
                return NamingConvention.GPTQ_COMPAT;
            }
        }

        return nativeHits >= gptqHits ? NamingConvention.NATIVE : NamingConvention.GPTQ_COMPAT;
    }

    // ── Layer Prefix Discovery ────────────────────────────────────────────────

    private Set<String> discoverLayerPrefixes() {
        Pattern pattern = (detectedConvention == NamingConvention.GPTQ_COMPAT)
                ? GPTQ_COMPAT_PATTERN
                : NATIVE_PATTERN;

        String weightSuffix = (detectedConvention == NamingConvention.GPTQ_COMPAT)
                ? ".qweight"
                : ".weight";
        String scaleSuffix = (detectedConvention == NamingConvention.GPTQ_COMPAT)
                ? ".scales"
                : ".scale";

        Set<String> prefixes = new LinkedHashSet<>();
        for (String name : tensorIndex.keySet()) {
            Matcher m = pattern.matcher(name);
            if (m.matches()) {
                String prefix = m.group(1);
                if (tensorIndex.containsKey(prefix + weightSuffix)
                        && tensorIndex.containsKey(prefix + scaleSuffix)) {
                    prefixes.add(prefix);
                }
            }
        }
        return prefixes;
    }

    // ── Layer Loading ─────────────────────────────────────────────────────────

    private AutoRoundLayer loadLayer(String prefix) {
        return (detectedConvention == NamingConvention.GPTQ_COMPAT)
                ? loadLayerGptqCompat(prefix)
                : loadLayerNative(prefix);
    }

    /**
     * Load AutoRound native format tensors:
     * .weight (INT32), .scale (FP32), .zp (INT32 plain, optional), .bias
     */
    private AutoRoundLayer loadLayerNative(String prefix) {
        AutoRoundLayer layer = new AutoRoundLayer(prefix, config);

        // Required: weight (INT32)
        layer.setWeight(copyTensorOffHeap(prefix + ".weight"));

        // Required: scale (FP32)
        layer.setScale(copyTensorOffHeap(prefix + ".scale"));

        // Optional: zp (plain INT32)
        if (tensorIndex.containsKey(prefix + ".zp")) {
            layer.setZp(copyTensorOffHeap(prefix + ".zp"));
        }

        // Optional: bias
        if (tensorIndex.containsKey(prefix + ".bias")) {
            layer.setBias(copyTensorOffHeap(prefix + ".bias"));
        }

        inferDimensionsNative(layer, prefix);
        log.debug("Loaded native: {}", layer);
        return layer;
    }

    /**
     * Load AutoRound GPTQ-compat format and normalise:
     * .qweight (INT32) → store as-is
     * .scales (FP16) → convert to FP32, transpose [g,j] → [j,g]
     * .qzeros (INT32 packed) → unpack to plain INT32 [j,g]
     * .bias (FP16/FP32) → store as-is (converted at dequant time)
     */
    private AutoRoundLayer loadLayerGptqCompat(String prefix) {
        AutoRoundLayer layer = new AutoRoundLayer(prefix, config);

        // Weight is identical packing to GPTQ — load as-is
        layer.setWeight(copyTensorOffHeap(prefix + ".qweight"));

        // Scales: FP16 [numGroups, outF] → FP32 [outF, numGroups]
        layer.setScale(loadAndTransposeScales(prefix + ".scales", layer));
        layer.setScaleWasFp16(true);

        // Zeros: packed INT32 → plain INT32 [outF, numGroups]
        if (tensorIndex.containsKey(prefix + ".qzeros")) {
            layer.setZp(loadAndUnpackZeros(prefix + ".qzeros", layer));
        }

        // Bias
        if (tensorIndex.containsKey(prefix + ".bias")) {
            layer.setBias(copyTensorOffHeap(prefix + ".bias"));
        }

        inferDimensionsGptqCompat(layer, prefix);
        log.debug("Loaded gptq-compat: {}", layer);
        return layer;
    }

    // ── GPTQ-compat normalisation helpers ─────────────────────────────────────

    /**
     * Reads FP16 scales [numGroups, outF] and transposes to FP32 [outF, numGroups].
     * The transposition makes index access scale[j * numGroups + g] consistent
     * with AutoRound native layout, so the dequantizer can use one code path.
     */
    private MemorySegment loadAndTransposeScales(String tensorName, AutoRoundLayer layer) {
        SafetensorParser parser = tensorIndex.get(tensorName);
        if (parser == null)
            throw new NoSuchElementException("Missing: " + tensorName);

        // Read raw FP16 array from mmap
        short[] fp16 = readFp16FromMmap(parser, tensorName);

        // Infer dimensions from shape
        SafetensorTensorInfo info = parser.getHeader().findTensor(tensorName).orElseThrow(
                () -> new NoSuchElementException("Missing tensor info: " + tensorName));
        long[] shape = info.shape();
        int numGroups = (int) shape[0];
        int outF = (int) shape[1];

        // Transpose and convert: GPTQ [g, j] → AutoRound [j, g]
        AutoRoundDequantizer dq = new AutoRoundDequantizer(config);
        float[] fp32 = dq.transposeFp16ScalesToFp32(fp16, outF, numGroups);

        // Store in off-heap FP32 segment
        return allocator.fromFloat32Array(fp32, tensorName + ":fp32-transposed");
    }

    /**
     * Reads packed INT32 qzeros [numGroups, outF/pack] and unpacks
     * to plain INT32 [outF, numGroups] — same shape as native zp tensor.
     */
    private MemorySegment loadAndUnpackZeros(String tensorName, AutoRoundLayer layer) {
        SafetensorParser parser = tensorIndex.get(tensorName);
        if (parser == null)
            return null;

        int[] packed = readInt32FromMmap(parser, tensorName);

        SafetensorTensorInfo info = parser.getHeader().findTensor(tensorName).orElse(null);
        if (info == null) return null;
        int numGroups = (int) info.shape()[0];
        int outF = (int) (info.shape()[1] * config.packFactor());

        AutoRoundDequantizer dq = new AutoRoundDequantizer(config);
        int[] plain = dq.unpackGptqCompatZeros(packed, outF, numGroups);

        return allocator.fromInt32Array(plain, tensorName + ":unpacked");
    }

    // ── Dimension Inference ───────────────────────────────────────────────────

    private void inferDimensionsNative(AutoRoundLayer layer, String prefix) {
        SafetensorParser parser = tensorIndex.get(prefix + ".weight");
        if (parser == null)
            return;

        SafetensorTensorInfo wi = parser.getHeader().findTensor(prefix + ".weight").orElse(null);
        if (wi == null || wi.shape() == null || wi.shape().length < 2)
            return;

        // Native weight shape: [outF/pack, inF]
        long packedRows = wi.shape()[0];
        long inF = wi.shape()[1];
        long outF = packedRows * config.packFactor();

        layer.setInFeatures((int) inF);
        layer.setOutFeatures((int) outF);
        layer.setWeightShape(new long[] { packedRows, inF });

        // Scale shape: [outF, inF/group]
        SafetensorParser sp = tensorIndex.get(prefix + ".scale");
        if (sp != null) {
            SafetensorTensorInfo si = sp.getHeader().findTensor(prefix + ".scale").orElse(null);
            if (si != null && si.shape() != null) {
                layer.setScaleShape(si.shape());
            }
        }
    }

    private void inferDimensionsGptqCompat(AutoRoundLayer layer, String prefix) {
        SafetensorParser parser = tensorIndex.get(prefix + ".qweight");
        if (parser == null)
            return;

        SafetensorTensorInfo wi = parser.getHeader().findTensor(prefix + ".qweight").orElse(null);
        if (wi == null || wi.shape() == null || wi.shape().length < 2)
            return;

        long packedRows = wi.shape()[0];
        long inF = wi.shape()[1];

        layer.setInFeatures((int) inF);
        layer.setOutFeatures((int) (packedRows * config.packFactor()));
        layer.setWeightShape(new long[] { packedRows, inF });
    }

    // ── FFM Read Helpers ──────────────────────────────────────────────────────

    private MemorySegment copyTensorOffHeap(String name) {
        SafetensorParser parser = tensorIndex.get(name);
        if (parser == null)
            throw new NoSuchElementException("Tensor not found: " + name);
        MemorySegment mmap = parser.getTensorSegment(name);
        return allocator.copyFrom(mmap, name);
    }

    private int[] readInt32FromMmap(SafetensorParser parser, String name) {
        MemorySegment seg = parser.getTensorSegment(name);
        int count = (int) (seg.byteSize() / Integer.BYTES);
        int[] arr = new int[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Integer.BYTES);
        }
        return arr;
    }

    private short[] readFp16FromMmap(SafetensorParser parser, String name) {
        MemorySegment seg = parser.getTensorSegment(name);
        int count = (int) (seg.byteSize() / Short.BYTES);
        short[] arr = new short[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Short.BYTES);
        }
        return arr;
    }

    // ── Auto-Detect Config ────────────────────────────────────────────────────

    /**
     * Auto-detects AutoRound configuration from model __metadata__.
     *
     * AutoRound metadata keys:
     * quant_method: "autoround" or "auto_round"
     * bits: "4"
     * group_size: "128"
     * sym: "False" / "True"
     * backend: "autoround:exllamav2" | "autoround:marlin" | "autoround:auto"
     */
    public static AutoRoundConfig autoDetectConfig(Path modelDir) throws IOException {
        var files = modelDir.toFile().listFiles(
                f -> f.isFile() && f.getName().endsWith(".safetensors"));

        if (files == null || files.length == 0) {
            log.warn("No safetensors found, using default AutoRound config");
            return AutoRoundConfig.autoRound4bit();
        }

        try (SafetensorParser parser = new SafetensorParser(files[0].toPath())) {
            Map<String, String> meta = parser.getHeader().fileMetadata();
            log.info("Model metadata: {}", meta);

            // Validate it's actually AutoRound
            String method = meta.getOrDefault("quant_method",
                    meta.getOrDefault("quantization_method", "")).toLowerCase();
            if (!method.contains("autoround") && !method.contains("auto_round")) {
                log.warn("quant_method='{}' does not look like AutoRound; proceeding anyway", method);
            }

            int bits = parseIntOr(meta.getOrDefault("bits", "4"), 4);
            int groupSize = parseIntOr(meta.getOrDefault("group_size", "128"), 128);
            boolean sym = "true".equalsIgnoreCase(
                    meta.getOrDefault("sym", "False").trim());

            // Detect scale dtype from backend hint or by probing tensor dtypes
            AutoRoundConfig.ScaleDtype scaleDtype = detectScaleDtype(parser.getHeader(), meta);

            // Detect pack format from backend hint
            String backend = meta.getOrDefault("backend", "").toLowerCase();
            AutoRoundConfig.PackFormat packFmt = AutoRoundConfig.PackFormat.AUTOROUND_NATIVE;
            if (backend.contains("gptq") || scaleDtype == AutoRoundConfig.ScaleDtype.FLOAT16) {
                packFmt = AutoRoundConfig.PackFormat.GPTQ_COMPAT;
            } else if (backend.contains("itrex") || backend.contains("ipex")) {
                packFmt = AutoRoundConfig.PackFormat.ITREX;
            }

            AutoRoundConfig cfg = new AutoRoundConfig(bits, groupSize, !sym,
                    scaleDtype, packFmt, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
            log.info("Auto-detected AutoRound config: {}", cfg);
            return cfg;
        }
    }

    private static AutoRoundConfig.ScaleDtype detectScaleDtype(
            SafetensorHeader header, Map<String, String> meta) {
        // Check a ".scale" tensor dtype
        for (var e : header.tensors().entrySet()) {
            String name = e.getKey();
            String dtype = e.getValue().dtype().jsonName();
            if (name.endsWith(".scale") && "F32".equals(dtype)) {
                return AutoRoundConfig.ScaleDtype.FLOAT32;
            }
            if (name.endsWith(".scales") && "F16".equals(dtype)) {
                return AutoRoundConfig.ScaleDtype.FLOAT16;
            }
            if (name.endsWith(".scale") && "BF16".equals(dtype)) {
                return AutoRoundConfig.ScaleDtype.BFLOAT16;
            }
        }
        // Default to FP32 (AutoRound native)
        return AutoRoundConfig.ScaleDtype.FLOAT32;
    }

    private static int parseIntOr(String s, int fallback) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public Map<String, AutoRoundLayer> getLayers() {
        return Collections.unmodifiableMap(layers);
    }

    public AutoRoundLayer getLayer(String name) {
        return layers.get(name);
    }

    public List<String> getLayerNames() {
        return new ArrayList<>(layers.keySet());
    }

    public AutoRoundConfig getConfig() {
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

    public NamingConvention getDetectedConvention() {
        return detectedConvention;
    }

    public void printSummary() {
        System.out.println("=== AutoRound Model Summary ===");
        System.out.printf("Config:     %s%n", config);
        System.out.printf("Convention: %s%n", detectedConvention);
        System.out.printf("Layers:     %d%n", layers.size());
        System.out.printf("Off-heap:   %.2f MB%n",
                allocator.getTotalAllocated() / 1_048_576.0);
        System.out.println();
        System.out.printf("%-65s %6s %6s %6s %5s %5s%n",
                "Layer", "InF", "OutF", "Groups", "ZP", "Bias");
        System.out.println("-".repeat(100));
        layers.forEach((name, layer) -> System.out.printf("%-65s %6d %6d %6d %5s %5s%n",
                name, layer.getInFeatures(), layer.getOutFeatures(),
                layer.numGroups(),
                layer.hasZp() ? "yes" : "no",
                layer.hasBias() ? "yes" : "no"));
    }

    @Override
    public void close() {
        log.info("Closing AutoRoundLoader: {} parsers, {:.2f} MB off-heap",
                openParsers.size(), allocator.getTotalAllocated() / 1_048_576.0);
        openParsers.forEach(p -> {
            try {
                p.close();
            } catch (Exception e) {
                log.warn("Parser close: {}", e.getMessage());
            }
        });
        allocator.close();
    }
}
