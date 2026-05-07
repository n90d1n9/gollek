package tech.kayys.gollek.quantizer.autoround;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.quantizer.autoround.AutoRoundLoader;
import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import tech.kayys.gollek.quantizer.autoround.AutoRoundLayer;
import tech.kayys.gollek.quantizer.autoround.AutoRoundDequantizer;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.CompletableFuture;
import java.util.*;

/**
 * Converts AutoRound-quantized models to dequantized FP32 safetensors.
 *
 * Pipeline:
 * 1. For each AutoRoundLayer: read weight / scale / zp from off-heap FFM
 * segments
 * 2. Invoke AutoRoundDequantizer (Vector API) → FP32 weight matrix [outF × inF]
 * 3. Build safetensor JSON header with corrected shapes, dtype=F32, new offsets
 * 4. Write output file via FFM memory-mapped I/O (zero-copy, OS page cache)
 *
 * Output tensor naming:
 * Input: "model.layers.0.mlp.up_proj.weight" (native)
 * "model.layers.0.mlp.up_proj.qweight" (gptq-compat)
 * Output: "model.layers.0.mlp.up_proj.weight" (always normalised)
 * Shape: [outFeatures, inFeatures]
 *
 * Scale / ZP tensors are NOT written to the output — they are consumed
 * during dequantization and are no longer needed in the FP32 output.
 *
 * Bias tensors (FP32 or FP16) are promoted to FP32 and written.
 *
 * Provenance metadata is written to __metadata__:
 * converted_by, source_format, source_bits, source_group_size,
 * source_sym, source_scale_dtype, source_pack_format, output_dtype
 */
public class AutoRoundSafetensorConverter {

    private static final Logger log = LoggerFactory.getLogger(AutoRoundSafetensorConverter.class);

    private final AutoRoundLoader loader;
    private final AutoRoundDequantizer dequantizer;
    private final ConversionConfig convConfig;
    private final ObjectMapper json = new ObjectMapper();
    private volatile boolean cancelled = false;

    public AutoRoundSafetensorConverter(AutoRoundLoader loader, ConversionConfig convConfig) {
        this.loader = loader;
        this.convConfig = convConfig;
        this.dequantizer = new AutoRoundDequantizer(loader.getConfig());
    }

    // ── Main Entry Point ──────────────────────────────────────────────────────

    /**
     * Dequantizes all AutoRound layers and writes {@code outputPath}.
     */
    public ConversionResult convert(Path outputPath) throws IOException {
        log.info("AutoRound → FP32 conversion → {}", outputPath);
        long t0 = System.currentTimeMillis();

        var layers = loader.getLayers();
        log.info("Dequantizing {} layers...", layers.size());

        // ── Step 1: Dequantize ────────────────────────────────────────────────
        Map<String, float[]> tensors = new LinkedHashMap<>();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        long totalElements = 0L;

        for (var entry : layers.entrySet()) {
            String prefix = entry.getKey();
            AutoRoundLayer layer = entry.getValue();

            if (!layer.isComplete()) {
                log.warn("Skipping incomplete layer: {}", prefix);
                continue;
            }

            // Dequantize weight
            float[] wFp32 = dequantizeLayer(layer);
            String outName = normalizeName(prefix) + ".weight";
            tensors.put(outName, wFp32);
            shapes.put(outName, new long[] { layer.getOutFeatures(), layer.getInFeatures() });
            totalElements += wFp32.length;

            // Include bias if present
            layer.getBias().ifPresent(biasSeg -> {
                String biasName = normalizeName(prefix) + ".bias";
                float[] biasFp32 = readBiasAsFp32(biasSeg, layer.getOutFeatures());
                tensors.put(biasName, biasFp32);
                shapes.put(biasName, new long[] { layer.getOutFeatures() });
            });

            if (convConfig.verbose()) {
                System.out.printf("[autoround-dequant] %-60s %d×%d = %,d elems%n",
                        prefix, layer.getOutFeatures(), layer.getInFeatures(), wFp32.length);
            }
        }

        log.info("Dequantized {} tensors ({} total elements)", tensors.size(), totalElements);

        // ── Step 2: Compute byte offsets ──────────────────────────────────────
        Map<String, long[]> offsets = new LinkedHashMap<>();
        long dataPos = 0L;
        for (var e : tensors.entrySet()) {
            long byteLen = (long) e.getValue().length * Float.BYTES;
            offsets.put(e.getKey(), new long[] { dataPos, dataPos + byteLen });
            dataPos += byteLen;
        }

        // ── Step 3: Build JSON header ─────────────────────────────────────────
        byte[] headerJson = buildHeader(tensors, shapes, offsets);
        long jsonLen = headerJson.length;
        long totalBytes = 8L + jsonLen + dataPos;

        log.info("Output: {} bytes JSON + {:.2f} MB data = {:.2f} MB total",
                jsonLen, dataPos / 1_048_576.0, totalBytes / 1_048_576.0);

        // ── Step 4: Write via FFM mmap ────────────────────────────────────────
        writeFile(outputPath, totalBytes, jsonLen, headerJson, tensors);

        long elapsed = System.currentTimeMillis() - t0;
        ConversionResult result = new ConversionResult(
                layers.size(), tensors.size(), totalElements, totalBytes, elapsed, outputPath);
        log.info("Conversion complete: {}", result);
        return result;
    }

    // ── Dequantization ────────────────────────────────────────────────────────

    private float[] dequantizeLayer(AutoRoundLayer layer) {
        AutoRoundConfig cfg = layer.getConfig();
        int inF = layer.getInFeatures();
        int outF = layer.getOutFeatures();
        int numGroups = layer.numGroups();

        // Read weight (INT32) from off-heap
        int[] qweightInts = readInt32(layer.getWeight());

        // Read scale — always stored as FP32 after loader normalisation
        float[] scalesFp32 = readFloat32(layer.getScale());

        // Read zp — stored as plain INT32 after loader normalisation (or null)
        int[] zpInts = layer.hasZp() ? readInt32(layer.getZp().get()) : null;

        float[] output = new float[outF * inF];
        dequantizer.dequantize(qweightInts, scalesFp32, zpInts, inF, outF, output);
        return output;
    }

    // ── FFM Segment Read Helpers ──────────────────────────────────────────────

    private int[] readInt32(MemorySegment seg) {
        int count = (int) (seg.byteSize() / Integer.BYTES);
        int[] arr = new int[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Integer.BYTES);
        }
        return arr;
    }

    private float[] readFloat32(MemorySegment seg) {
        int count = (int) (seg.byteSize() / Float.BYTES);
        float[] arr = new float[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Float.BYTES);
        }
        return arr;
    }

    private float[] readBiasAsFp32(MemorySegment seg, int outF) {
        // Auto-detect: FP16 bias if byteSize == outF * 2, else FP32
        float[] bias = new float[outF];
        if (seg.byteSize() == (long) outF * Short.BYTES) {
            for (int j = 0; j < outF; j++) {
                short fp16 = seg.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        (long) j * Short.BYTES);
                bias[j] = MemoryAllocator.fp16ToFloat32(fp16);
            }
        } else {
            for (int j = 0; j < outF; j++) {
                bias[j] = seg.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        (long) j * Float.BYTES);
            }
        }
        return bias;
    }

    // ── JSON Header Builder ───────────────────────────────────────────────────

    private byte[] buildHeader(
            Map<String, float[]> tensors,
            Map<String, long[]> shapes,
            Map<String, long[]> offsets) throws IOException {
        AutoRoundConfig cfg = loader.getConfig();
        Map<String, Object> header = new LinkedHashMap<>();

        // Provenance metadata
        Map<String, String> meta = new LinkedHashMap<>(loader.getModelMetadata());
        meta.put("converted_by", "gptq-java/AutoRoundConverter");
        meta.put("source_format", "AutoRound");
        meta.put("source_bits", String.valueOf(cfg.bits()));
        meta.put("source_group_size", String.valueOf(cfg.groupSize()));
        meta.put("source_sym", String.valueOf(!cfg.hasZeroPoint()));
        meta.put("source_scale_dtype", cfg.scaleDtype().name());
        meta.put("source_pack_format", cfg.packFormat().name());
        meta.put("output_dtype", "F32");
        header.put("__metadata__", meta);

        for (String name : tensors.keySet()) {
            long[] shape = shapes.get(name);
            long[] off = offsets.get(name);
            Map<String, Object> td = new LinkedHashMap<>();
            td.put("dtype", "F32");
            td.put("shape", Arrays.stream(shape).boxed().toList());
            td.put("data_offsets", List.of(off[0], off[1]));
            header.put(name, td);
        }

        return json.writeValueAsBytes(header);
    }

    // ── FFM Memory-Mapped File Writer ─────────────────────────────────────────

    /**
     * Writes the output .safetensors file using FFM READ_WRITE mmap.
     *
     * Layout: [8-byte LE json_len][json_bytes][fp32 tensor data...]
     */
    private void writeFile(
            Path outputPath, long totalBytes, long jsonLen,
            byte[] jsonBytes, Map<String, float[]> tensors) throws IOException {
        // Pre-allocate
        try (RandomAccessFile raf = new RandomAccessFile(outputPath.toFile(), "rw")) {
            raf.setLength(totalBytes);
        }

        try (FileChannel ch = FileChannel.open(outputPath,
                StandardOpenOption.READ, StandardOpenOption.WRITE);
                Arena arena = Arena.ofConfined()) {

            MemorySegment dest = ch.map(FileChannel.MapMode.READ_WRITE, 0, totalBytes, arena);
            long pos = 0L;

            // 8-byte header length
            dest.set(ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN), pos, jsonLen);
            pos += 8;

            // JSON header
            MemorySegment.copy(MemorySegment.ofArray(jsonBytes), 0, dest, pos, jsonLen);
            pos += jsonLen;

            // Tensor data (FP32 LE)
            for (float[] data : tensors.values()) {
                for (float v : data) {
                    dest.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
                    pos += Float.BYTES;
                }
            }

            dest.force();
            log.debug("Wrote {} bytes to {}", pos, outputPath);
        }
    }

    /**
     * Builds the JSON header byte array for the output safetensors file.
     * Alias for {@link #buildHeader(Map, Map, Map)} used by progress conversion.
     */
    private byte[] buildJsonHeader(
            Map<String, float[]> tensors,
            Map<String, long[]> shapes,
            Map<String, long[]> offsets) throws IOException {
        return buildHeader(tensors, shapes, offsets);
    }

    /**
     * Writes the output safetensors file.
     * Alias for {@link #writeFile(Path, long, long, byte[], Map)} used by progress conversion.
     */
    private void writeOutputFile(
            Path outputPath, long totalBytes, long jsonLen,
            byte[] jsonBytes, Map<String, float[]> tensors) throws IOException {
        writeFile(outputPath, totalBytes, jsonLen, jsonBytes, tensors);
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /**
     * Normalises a layer prefix by stripping GPTQ-compat name suffixes.
     * "model.layers.0.q_proj" → "model.layers.0.q_proj" (no change for native)
     */
    private String normalizeName(String prefix) {
        return prefix; // prefix never contains the tensor-type suffix
    }

    // ── Result & Config Records ───────────────────────────────────────────────

    public record ConversionResult(
            int layersConverted,
            int tensorsWritten,
            long totalElements,
            long outputFileSizeBytes,
            long elapsedMs,
            Path outputPath) {
        public double throughputMBps() {
            return (outputFileSizeBytes / 1e6) / Math.max(elapsedMs / 1000.0, 0.001);
        }

        @Override
        public String toString() {
            return ("AutoRoundConversionResult{layers=%d, tensors=%d, elements=%,d, " +
                    "size=%.2f MB, elapsed=%d ms, throughput=%.1f MB/s}")
                    .formatted(layersConverted, tensorsWritten, totalElements,
                            outputFileSizeBytes / 1e6, elapsedMs, throughputMBps());
        }
    }

    public record ConversionConfig(boolean verbose, String outputDtype) {
        public static ConversionConfig defaults() {
            return new ConversionConfig(false, "F32");
        }

        public static ConversionConfig verboseConfig() {
            return new ConversionConfig(true, "F32");
        }
    }

    // ── Enhanced Conversion Methods ───────────────────────────────────────────

    /**
     * Asynchronously converts the model with progress reporting.
     * 
     * @param outputPath Path for output file
     * @param progressCallback Callback for progress updates (layer name, current, total)
     * @return CompletableFuture with conversion result
     */
    public CompletableFuture<ConversionResult> convertAsync(Path outputPath, 
                                                             ProgressCallback progressCallback) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return convertWithProgress(outputPath, progressCallback);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Converts with progress reporting callbacks.
     */
    private ConversionResult convertWithProgress(Path outputPath, ProgressCallback progressCallback) 
            throws IOException {
        log.info("AutoRound conversion with progress reporting → {}", outputPath);
        long t0 = System.currentTimeMillis();

        var layers = loader.getLayers();
        int totalLayers = layers.size();
        int currentLayer = 0;

        Map<String, float[]> dequantized = new LinkedHashMap<>();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        long totalElements = 0L;

        for (var entry : layers.entrySet()) {
            if (cancelled) {
                throw new IOException("Conversion cancelled");
            }

            String prefix = entry.getKey();
            AutoRoundLayer layer = entry.getValue();

            if (!layer.isComplete()) {
                log.warn("Skipping incomplete layer: {}", prefix);
                currentLayer++;
                continue;
            }

            if (progressCallback != null) {
                progressCallback.onProgress(prefix, currentLayer, totalLayers);
            }

            float[] w = dequantizeLayer(layer);
            String outName = prefix + ".weight";
            dequantized.put(outName, w);
            shapes.put(outName, new long[] { layer.getInFeatures(), layer.getOutFeatures() });
            totalElements += w.length;
            currentLayer++;

            layer.getBias().ifPresent(biasSeg -> {
                String biasName = prefix + ".bias";
                float[] biasF32 = readBiasAsFp32(biasSeg, layer.getOutFeatures());
                dequantized.put(biasName, biasF32);
                shapes.put(biasName, new long[] { layer.getOutFeatures() });
            });

            if (convConfig.verbose()) {
                System.out.printf("[autoround-dequant] %-65s %d×%d = %,d elements (%d/%d)%n",
                        prefix, layer.getInFeatures(), layer.getOutFeatures(), w.length, 
                        currentLayer, totalLayers);
            }
        }

        // Build and write output (same as regular convert)
        Map<String, long[]> offsets = new LinkedHashMap<>();
        long dataPos = 0L;
        for (var e : dequantized.entrySet()) {
            long byteLen = (long) e.getValue().length * Float.BYTES;
            offsets.put(e.getKey(), new long[] { dataPos, dataPos + byteLen });
            dataPos += byteLen;
        }

        byte[] headerJson = buildJsonHeader(dequantized, shapes, offsets);
        long jsonLen = headerJson.length;
        long totalBytes = 8L + jsonLen + dataPos;

        writeOutputFile(outputPath, totalBytes, jsonLen, headerJson, dequantized);

        long elapsed = System.currentTimeMillis() - t0;

        return new ConversionResult(
                layers.size(), dequantized.size(), totalElements, totalBytes, elapsed, outputPath);
    }

    /**
     * Cancels an ongoing conversion operation.
     */
    public void cancel() {
        this.cancelled = true;
        log.info("AutoRound conversion cancellation requested");
    }

    // ── Progress Callback Interface ───────────────────────────────────────────

    /**
     * Callback interface for reporting conversion progress.
     */
    @FunctionalInterface
    public interface ProgressCallback {
        /**
         * Called during conversion progress updates.
         * 
         * @param layerName Current layer being processed
         * @param current Current layer index
         * @param total Total number of layers
         */
        void onProgress(String layerName, int current, int total);
    }
}
