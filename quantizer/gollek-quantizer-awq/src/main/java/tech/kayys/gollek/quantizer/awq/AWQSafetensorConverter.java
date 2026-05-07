package tech.kayys.gollek.quantizer.awq;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.quantizer.awq.AWQLoader;
import tech.kayys.gollek.quantizer.awq.AWQConfig;
import tech.kayys.gollek.quantizer.awq.AWQLayer;
import tech.kayys.gollek.quantizer.awq.AWQDequantizer;
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
import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * Converts AWQ-quantized models to dequantized FP32 safetensors.
 *
 * Pipeline:
 * 1. For each AWQ layer: read qweight / qzeros / scales from off-heap FFM segments
 * 2. Dequantize INT4 → FP32 using Vector API (AWQDequantizer)
 * 3. Build safetensor JSON header with updated shape/dtype/offsets
 * 4. Memory-map the output file and write all tensors sequentially
 *
 * Output tensor naming:
 * AWQ input: "model.layers.0.self_attn.q_proj.qweight"
 * Output: "model.layers.0.self_attn.q_proj.weight"
 * Shape: [inFeatures, outFeatures] (AWQ row-major convention)
 *
 * The converter also handles bias tensors (FP16 → FP32 promotion) and
 * writes provenance metadata into the output file's __metadata__ block.
 *
 * FFM write strategy:
 * The output file is memory-mapped via FFM (READ_WRITE mapping), allowing
 * zero-copy writes from the dequantized float[] arrays directly into the
 * OS page cache — the kernel flushes pages to disk asynchronously.
 * 
 * Enhanced features:
 * - Async conversion with progress reporting
 * - Cancellation support
 * - Progress callback interface
 */
public class AWQSafetensorConverter {

    private static final Logger log = LoggerFactory.getLogger(AWQSafetensorConverter.class);

    private final AWQLoader loader;
    private final AWQDequantizer dequantizer;
    private final ConversionConfig convConfig;
    private final ObjectMapper json = new ObjectMapper();
    private volatile boolean cancelled = false;

    public AWQSafetensorConverter(AWQLoader loader, ConversionConfig convConfig) {
        this.loader = loader;
        this.convConfig = convConfig;
        this.dequantizer = new AWQDequantizer(loader.getConfig());
    }

    // ── Main Entry Point ──────────────────────────────────────────────────────

    /**
     * Runs full AWQ → FP32 conversion and writes {@code outputPath}.
     *
     * @return a {@link ConversionResult} summary
     */
    public ConversionResult convert(Path outputPath) throws IOException {
        log.info("AWQ → FP32 conversion → {}", outputPath);
        long t0 = System.currentTimeMillis();

        var layers = loader.getLayers();
        log.info("Dequantizing {} layers...", layers.size());

        // ── Step 1: Dequantize all layers ─────────────────────────────────────
        // LinkedHashMap preserves insertion order for deterministic output layout
        Map<String, float[]> dequantized = new LinkedHashMap<>();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        long totalElements = 0L;

        for (var entry : layers.entrySet()) {
            String prefix = entry.getKey();
            AWQLayer layer = entry.getValue();

            if (!layer.isComplete()) {
                log.warn("Skipping incomplete layer: {}", prefix);
                continue;
            }

            float[] w = dequantizeLayer(layer);
            String outName = prefix + ".weight";
            dequantized.put(outName, w);
            shapes.put(outName, new long[] { layer.getInFeatures(), layer.getOutFeatures() });
            totalElements += w.length;

            // Handle bias: FP16 → FP32
            layer.getBias().ifPresent(biasSeg -> {
                String biasName = prefix + ".bias";
                float[] biasF32 = readBiasAsFp32(biasSeg, layer.getOutFeatures());
                dequantized.put(biasName, biasF32);
                shapes.put(biasName, new long[] { layer.getOutFeatures() });
            });

            if (convConfig.verbose()) {
                System.out.printf("[awq-dequant] %-65s %d×%d = %,d elements%n",
                        prefix, layer.getInFeatures(), layer.getOutFeatures(), w.length);
            }
        }

        log.info("Dequantized {} tensors, {} total elements", dequantized.size(), totalElements);

        // ── Step 2: Compute byte offsets and build header ─────────────────────
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

        log.info("Output: JSON header {} bytes + {:.2f} MB data = {:.2f} MB total",
                jsonLen, dataPos / 1_048_576.0, totalBytes / 1_048_576.0);

        // ── Step 3: Write output via FFM memory-mapped file ───────────────────
        writeOutputFile(outputPath, totalBytes, jsonLen, headerJson, dequantized);

        long elapsed = System.currentTimeMillis() - t0;

        ConversionResult result = new ConversionResult(
                layers.size(), dequantized.size(), totalElements, totalBytes, elapsed, outputPath);

        log.info("Conversion complete: {}", result);
        return result;
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
        log.info("AWQ conversion with progress reporting → {}", outputPath);
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
            AWQLayer layer = entry.getValue();

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
                System.out.printf("[awq-dequant] %-65s %d×%d = %,d elements (%d/%d)%n",
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
        log.info("AWQ conversion cancellation requested");
    }

    // ── Dequantization ────────────────────────────────────────────────────────

    /**
     * Reads AWQ tensor data from off-heap FFM segments and dequantizes to FP32.
     */
    private float[] dequantizeLayer(AWQLayer layer) {
        int inF = layer.getInFeatures();
        int outF = layer.getOutFeatures();

        int[] qweightInts = readInt32(layer.getQweight());
        int[] qzerosInts = layer.hasZeros() ? readInt32(layer.getQzeros()) : null;
        short[] scalesShorts = readFp16(layer.getScales());

        float[] output = new float[inF * outF];
        dequantizer.dequantize(qweightInts, qzerosInts, scalesShorts, inF, outF, output);
        return output;
    }

    // ── FFM Read Helpers ──────────────────────────────────────────────────────

    private int[] readInt32(MemorySegment seg) {
        int count = (int) (seg.byteSize() / Integer.BYTES);
        int[] arr = new int[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Integer.BYTES);
        }
        return arr;
    }

    private short[] readFp16(MemorySegment seg) {
        int count = (int) (seg.byteSize() / Short.BYTES);
        short[] arr = new short[count];
        for (int i = 0; i < count; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Short.BYTES);
        }
        return arr;
    }

    /**
     * Reads bias from off-heap: auto-detects FP16 or FP32 based on byte size.
     */
    private float[] readBiasAsFp32(MemorySegment seg, int outF) {
        long bytes = seg.byteSize();
        float[] bias = new float[outF];

        if (bytes == (long) outF * Short.BYTES) {
            // FP16 bias
            for (int j = 0; j < outF; j++) {
                short fp16 = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        (long) j * Short.BYTES);
                bias[j] = MemoryAllocator.fp16ToFloat32(fp16);
            }
        } else {
            // FP32 bias
            for (int j = 0; j < outF; j++) {
                bias[j] = seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        (long) j * Float.BYTES);
            }
        }
        return bias;
    }

    // ── JSON Header Builder ───────────────────────────────────────────────────

    private byte[] buildJsonHeader(
            Map<String, float[]> tensors,
            Map<String, long[]> shapes,
            Map<String, long[]> offsets) throws IOException {
        Map<String, Object> header = new LinkedHashMap<>();

        // Provenance metadata
        Map<String, String> meta = new LinkedHashMap<>(loader.getModelMetadata());
        meta.put("converted_by", "gptq-java/AWQConverter");
        meta.put("source_format", "AWQ");
        meta.put("source_bits", String.valueOf(loader.getConfig().bits()));
        meta.put("source_group_size", String.valueOf(loader.getConfig().groupSize()));
        meta.put("source_kernel", loader.getConfig().kernelFormat().name());
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

    // ── FFM Memory-Mapped Output Writer ──────────────────────────────────────

    /**
     * Writes the safetensor file using FFM memory-mapped I/O.
     *
     * File layout:
     * [8 bytes: LE uint64 JSON header length]
     * [N bytes: UTF-8 JSON header]
     * [M bytes: FP32 tensor data, LE, sequential]
     */
    private void writeOutputFile(
            Path outputPath,
            long totalBytes,
            long jsonLen,
            byte[] jsonBytes,
            Map<String, float[]> tensors) throws IOException {
        // Pre-allocate file to final size
        try (RandomAccessFile raf = new RandomAccessFile(outputPath.toFile(), "rw")) {
            raf.setLength(totalBytes);
        }

        try (FileChannel ch = FileChannel.open(outputPath,
                StandardOpenOption.READ, StandardOpenOption.WRITE);
                Arena arena = Arena.ofConfined()) {

            MemorySegment dest = ch.map(FileChannel.MapMode.READ_WRITE, 0, totalBytes, arena);
            long pos = 0L;

            // 8-byte LE header size
            dest.set(ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN), pos, jsonLen);
            pos += 8;

            // JSON header bytes
            MemorySegment.copy(MemorySegment.ofArray(jsonBytes), 0, dest, pos, jsonLen);
            pos += jsonLen;

            // Tensor data: FP32 LE
            for (float[] data : tensors.values()) {
                for (float v : data) {
                    dest.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), pos, v);
                    pos += Float.BYTES;
                }
            }

            dest.force(); // flush dirty pages to storage
            log.debug("Wrote {} bytes to {}", pos, outputPath);
        }
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
            return ("AWQConversionResult{layers=%d, tensors=%d, elements=%,d, " +
                    "size=%.2f MB, elapsed=%d ms, throughput=%.1f MB/s}")
                    .formatted(layersConverted, tensorsWritten, totalElements,
                            outputFileSizeBytes / 1e6, elapsedMs, throughputMBps());
        }
    }

    public record ConversionConfig(
            boolean verbose,
            String outputDtype) {
        public static ConversionConfig defaults() {
            return new ConversionConfig(false, "F32");
        }

        public static ConversionConfig verboseConfig() {
            return new ConversionConfig(true, "F32");
        }
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
