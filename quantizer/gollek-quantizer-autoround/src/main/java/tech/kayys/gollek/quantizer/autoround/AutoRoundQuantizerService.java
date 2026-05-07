package tech.kayys.gollek.quantizer.autoround;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Unified service for AutoRound quantization operations.
 * 
 * This service provides:
 * - Model quantization (FP32/FP16 → AutoRound INT4/INT8)
 * - Model dequantization (AutoRound → FP32/FP16)
 * - Model validation and inspection
 * - Integration with safetensor runner
 * 
 * AutoRound differs from GPTQ/AWQ in that it optimizes BOTH
 * the rounding decisions AND the scales using SignSGD on a
 * block-wise reconstruction objective.
 * 
 * Usage:
 * <pre>{@code
 * AutoRoundQuantizerService service = new AutoRoundQuantizerService();
 * 
 * // Load quantized model
 * AutoRoundLoader loader = service.loadQuantized(Path.of("/path/to/autoround-model"));
 * 
 * // Dequantize to FP32
 * ConversionResult result = service.dequantize(
 *     Path.of("/path/to/autoround-model"),
 *     Path.of("/path/to/output.safetensors"),
 *     AutoRoundSafetensorConverter.ConversionConfig.defaults()
 * );
 * }</pre>
 */
public class AutoRoundQuantizerService implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AutoRoundQuantizerService.class);

    private boolean initialized = false;
    private AutoRoundLoader activeLoader;
    private AutoRoundDequantizer dequantizer;

    /**
     * Initialize the AutoRound quantizer service.
     */
    public AutoRoundQuantizerService() {
        log.info("AutoRoundQuantizerService initialized");
        this.initialized = true;
    }

    // ── Loading Operations ────────────────────────────────────────────────────

    /**
     * Loads an AutoRound-quantized model from safetensors.
     * 
     * @param modelPath Path to quantized model directory
     * @return AutoRoundLoader instance
     * @throws IOException if loading fails
     */
    public AutoRoundLoader loadQuantized(Path modelPath) throws IOException {
        log.info("Loading quantized AutoRound model from: {}", modelPath);

        // Auto-detect config
        AutoRoundConfig config = AutoRoundLoader.autoDetectConfig(modelPath);
        log.info("Auto-detected config: {}", config);

        AutoRoundLoader loader = new AutoRoundLoader(modelPath, config);
        loader.load();
        
        this.activeLoader = loader;
        this.dequantizer = new AutoRoundDequantizer(config);

        log.info("AutoRound model loaded successfully: {} layers", loader.getLayerCount());
        return loader;
    }

    /**
     * Loads an AutoRound model with explicit configuration.
     * 
     * @param modelPath Path to model
     * @param config    Explicit AutoRound configuration
     * @return AutoRoundLoader instance
     * @throws IOException if loading fails
     */
    public AutoRoundLoader loadQuantized(Path modelPath, AutoRoundConfig config) throws IOException {
        log.info("Loading AutoRound model with explicit config: {}", config);

        AutoRoundLoader loader = new AutoRoundLoader(modelPath, config);
        loader.load();
        
        this.activeLoader = loader;
        this.dequantizer = new AutoRoundDequantizer(config);

        return loader;
    }

    // ── Dequantization Operations ─────────────────────────────────────────────

    /**
     * Dequantizes an AutoRound model back to FP32/FP16.
     * 
     * @param inputPath  Path to AutoRound model
     * @param outputPath Path for dequantized output
     * @param config     Conversion configuration
     * @return Conversion result
     * @throws IOException if conversion fails
     */
    public AutoRoundSafetensorConverter.ConversionResult dequantize(Path inputPath, Path outputPath, 
                                                            AutoRoundSafetensorConverter.ConversionConfig config) throws IOException {
        log.info("Dequantizing AutoRound model: {} → {}", inputPath, outputPath);

        AutoRoundLoader loader = loadQuantized(inputPath);
        
        AutoRoundSafetensorConverter converter = new AutoRoundSafetensorConverter(loader, config);
        AutoRoundSafetensorConverter.ConversionResult result = converter.convert(outputPath);

        log.info("Dequantization complete: {}", result);
        return result;
    }

    /**
     * Asynchronously dequantizes an AutoRound model.
     * 
     * @param inputPath  Path to AutoRound model
     * @param outputPath Path for dequantized output
     * @param config     Conversion configuration
     * @return CompletableFuture with conversion result
     */
    public CompletableFuture<AutoRoundSafetensorConverter.ConversionResult> dequantizeAsync(
            Path inputPath, Path outputPath, AutoRoundSafetensorConverter.ConversionConfig config) {
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
     * Inspects an AutoRound model and returns detailed information.
     * 
     * @param modelPath Path to model
     * @return Model inspection result
     * @throws IOException if inspection fails
     */
    public ModelInspectionResult inspect(Path modelPath) throws IOException {
        log.info("Inspecting AutoRound model: {}", modelPath);

        AutoRoundConfig config = AutoRoundLoader.autoDetectConfig(modelPath);
        AutoRoundLoader loader = new AutoRoundLoader(modelPath, config);
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
        log.info("Closing AutoRoundQuantizerService");
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
            AutoRoundConfig config,
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
