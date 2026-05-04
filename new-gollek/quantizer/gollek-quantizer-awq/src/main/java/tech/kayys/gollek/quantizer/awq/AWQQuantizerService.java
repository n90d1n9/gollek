package tech.kayys.gollek.quantizer.awq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Unified service for AWQ quantization operations.
 * 
 * This service provides:
 * - Model quantization (FP32/FP16 → AWQ INT4)
 * - Model dequantization (AWQ → FP32/FP16)
 * - Model validation and inspection
 * - Integration with safetensor runner
 * 
 * AWQ differs from GPTQ in that it uses activation-aware quantization,
 * protecting the ~1% of weights that matter most to activations.
 * 
 * Usage:
 * <pre>{@code
 * AWQQuantizerService service = new AWQQuantizerService();
 * 
 * // Load quantized model
 * AWQLoader loader = service.loadQuantized(Path.of("/path/to/awq-model"));
 * 
 * // Dequantize to FP32
 * ConversionResult result = service.dequantize(
 *     Path.of("/path/to/awq-model"),
 *     Path.of("/path/to/output.safetensors"),
 *     AWQSafetensorConverter.ConversionConfig.defaults()
 * );
 * }</pre>
 */
public class AWQQuantizerService implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AWQQuantizerService.class);

    private boolean initialized = false;
    private AWQLoader activeLoader;
    private AWQDequantizer dequantizer;

    /**
     * Initialize the AWQ quantizer service.
     */
    public AWQQuantizerService() {
        log.info("AWQQuantizerService initialized");
        this.initialized = true;
    }

    // ── Loading Operations ────────────────────────────────────────────────────

    /**
     * Loads an AWQ-quantized model from safetensors.
     * 
     * @param modelPath Path to quantized model directory
     * @return AWQLoader instance
     * @throws IOException if loading fails
     */
    public AWQLoader loadQuantized(Path modelPath) throws IOException {
        log.info("Loading quantized AWQ model from: {}", modelPath);

        // Auto-detect config
        AWQConfig config = AWQLoader.autoDetectConfig(modelPath);
        log.info("Auto-detected config: {}", config);

        AWQLoader loader = new AWQLoader(modelPath, config);
        loader.load();
        
        this.activeLoader = loader;
        this.dequantizer = new AWQDequantizer(config);

        log.info("AWQ model loaded successfully: {} layers", loader.getLayerCount());
        return loader;
    }

    /**
     * Loads an AWQ model with explicit configuration.
     * 
     * @param modelPath Path to model
     * @param config    Explicit AWQ configuration
     * @return AWQLoader instance
     * @throws IOException if loading fails
     */
    public AWQLoader loadQuantized(Path modelPath, AWQConfig config) throws IOException {
        log.info("Loading AWQ model with explicit config: {}", config);

        AWQLoader loader = new AWQLoader(modelPath, config);
        loader.load();
        
        this.activeLoader = loader;
        this.dequantizer = new AWQDequantizer(config);

        return loader;
    }

    // ── Dequantization Operations ─────────────────────────────────────────────

    /**
     * Dequantizes an AWQ model back to FP32/FP16.
     * 
     * @param inputPath  Path to AWQ model
     * @param outputPath Path for dequantized output
     * @param config     Conversion configuration
     * @return Conversion result
     * @throws IOException if conversion fails
     */
    public AWQSafetensorConverter.ConversionResult dequantize(Path inputPath, Path outputPath, 
                                                            AWQSafetensorConverter.ConversionConfig config) throws IOException {
        log.info("Dequantizing AWQ model: {} → {}", inputPath, outputPath);

        AWQLoader loader = loadQuantized(inputPath);
        
        AWQSafetensorConverter converter = new AWQSafetensorConverter(loader, config);
        AWQSafetensorConverter.ConversionResult result = converter.convert(outputPath);

        log.info("Dequantization complete: {}", result);
        return result;
    }

    /**
     * Asynchronously dequantizes an AWQ model.
     * 
     * @param inputPath  Path to AWQ model
     * @param outputPath Path for dequantized output
     * @param config     Conversion configuration
     * @return CompletableFuture with conversion result
     */
    public CompletableFuture<AWQSafetensorConverter.ConversionResult> dequantizeAsync(
            Path inputPath, Path outputPath, AWQSafetensorConverter.ConversionConfig config) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return dequantize(inputPath, outputPath, config);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    // ── Model Inspection ──────────────────────────────────────────────────────

    /**
     * Inspects an AWQ model and returns detailed information.
     * 
     * @param modelPath Path to model
     * @return Model inspection result
     * @throws IOException if inspection fails
     */
    public ModelInspectionResult inspect(Path modelPath) throws IOException {
        log.info("Inspecting AWQ model: {}", modelPath);

        AWQConfig config = AWQLoader.autoDetectConfig(modelPath);
        AWQLoader loader = new AWQLoader(modelPath, config);
        loader.load();

        ModelInspectionResult result = new ModelInspectionResult(
                config,
                loader.getLayerCount(),
                loader.getTotalOffHeapBytes(),
                loader.getLayerNames(),
                loader.getModelMetadata()
        );

        loader.close();
        
        log.info("Inspection complete: {}", result);
        return result;
    }

    // ── Resource Management ───────────────────────────────────────────────────

    @Override
    public void close() {
        log.info("Closing AWQQuantizerService");
        if (activeLoader != null) {
            try {
                activeLoader.close();
            } catch (Exception e) {
                log.warn("Error closing loader", e);
            }
        }
        this.initialized = false;
    }

    public boolean isInitialized() {
        return initialized;
    }

    // ── Result Records ────────────────────────────────────────────────────────

    /**
     * Model inspection result.
     */
    public record ModelInspectionResult(
            AWQConfig config,
            int layerCount,
            long totalMemoryBytes,
            java.util.List<String> layerNames,
            Map<String, String> metadata) {
        
        public double totalMemoryMB() {
            return totalMemoryBytes / (1024.0 * 1024.0);
        }

        @Override
        public String toString() {
            return "ModelInspectionResult{config=%s, layers=%d, memory=%.2f MB}"
                    .formatted(config, layerCount, totalMemoryMB());
        }
    }
}
