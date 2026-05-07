package tech.kayys.gollek.quantizer.quip;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * High-level service for QuIP# quantization and dequantization.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * try (QuipQuantizerService svc = new QuipQuantizerService(QuipConfig.quip2bit())) {
 *
 *     // Quantize a model (map of name → float[] weights with shapes)
 *     Map<String, QuipTensor> quantized = svc.quantize(weights, shapes);
 *
 *     // Dequantize back
 *     Map<String, float[]> approx = svc.dequantize(quantized);
 * }
 * }</pre>
 */
public class QuipQuantizerService implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(QuipQuantizerService.class);

    private final QuipConfig    config;
    private final QuipQuantizer quantizer;

    public QuipQuantizerService(QuipConfig config) {
        this.config    = config;
        this.quantizer = new QuipQuantizer(config);
        log.info("QuipQuantizerService ready — bits={}", config.bits());
    }

    /**
     * Quantize a map of named weight tensors.
     *
     * @param weights map of tensor name → float[] (row-major)
     * @param shapes  map of tensor name → [rows, cols]
     * @return map of tensor name → {@link QuipTensor}
     */
    public Map<String, QuipTensor> quantize(Map<String, float[]> weights,
                                             Map<String, int[]> shapes) {
        Map<String, QuipTensor> result = new LinkedHashMap<>();
        long totalOriginal = 0, totalCompressed = 0;

        for (var e : weights.entrySet()) {
            String name = e.getKey();
            float[] W   = e.getValue();
            int[] shape = shapes.getOrDefault(name, new int[]{1, W.length});
            int rows = shape[0], cols = shape.length > 1 ? shape[1] : W.length;

            QuipTensor qt = quantizer.quantize(name, W, rows, cols);
            result.put(name, qt);
            totalOriginal   += qt.originalBytes();
            totalCompressed += qt.compressedBytes();
        }

        log.info("QuIP# quantized {} tensors — {:.1f} MB → {:.1f} MB ({:.2f}x)",
                result.size(), totalOriginal / 1e6, totalCompressed / 1e6,
                (double) totalOriginal / totalCompressed);
        return result;
    }

    /**
     * Dequantize a map of {@link QuipTensor}s back to float arrays.
     */
    public Map<String, float[]> dequantize(Map<String, QuipTensor> quantized) {
        Map<String, float[]> result = new LinkedHashMap<>();
        for (var e : quantized.entrySet()) {
            result.put(e.getKey(), quantizer.dequantize(e.getValue()));
        }
        return result;
    }

    /**
     * Quantize from a safetensors file and write a QuIP# archive.
     * The archive is a directory containing one {@code .quip} file per tensor
     * (binary: rows(4) + cols(4) + seedU(8) + seedV(8) + numBlocks(4) +
     *  codes[numBlocks] + scales[numBlocks * 4]).
     */
    public void quantizeFile(Path inputSafetensors, Path outputDir) throws IOException {
        log.info("QuIP# quantizing {} → {}", inputSafetensors, outputDir);
        java.nio.file.Files.createDirectories(outputDir);

        // Load via SafetensorReader (reuse existing infrastructure)
        tech.kayys.gollek.ml.safetensors.SafetensorReader reader =
                new tech.kayys.gollek.ml.safetensors.SafetensorReader();
        Map<String, float[]> raw = tech.kayys.gollek.ml.safetensors.SafetensorReader.read(inputSafetensors);

        Map<String, int[]> shapes = new LinkedHashMap<>();
        raw.forEach((name, data) -> shapes.put(name, new int[]{1, data.length}));

        Map<String, QuipTensor> quantized = quantize(raw, shapes);
        writeArchive(quantized, outputDir);
        log.info("QuIP# archive written to {}", outputDir);
    }

    private void writeArchive(Map<String, QuipTensor> tensors, Path dir) throws IOException {
        for (var e : tensors.entrySet()) {
            Path out = dir.resolve(e.getKey().replace("/", "_").replace(".", "_") + ".quip");
            writeQuipFile(e.getValue(), out);
        }
    }

    private static void writeQuipFile(QuipTensor qt, Path path) throws IOException {
        int headerBytes = 4 + 4 + 8 + 8 + 4; // rows, cols, seedU, seedV, numBlocks
        int dataBytes   = qt.codes().length + qt.scales().length * 4;
        java.nio.ByteBuffer buf = java.nio.ByteBuffer
                .allocate(headerBytes + dataBytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.putInt(qt.rows());
        buf.putInt(qt.cols());
        buf.putLong(qt.seedU());
        buf.putLong(qt.seedV());
        buf.putInt(qt.codes().length);
        buf.put(qt.codes());
        for (float s : qt.scales()) buf.putFloat(s);
        java.nio.file.Files.write(path, buf.array());
    }

    @Override
    public void close() {
        log.debug("QuipQuantizerService closed");
    }
}
