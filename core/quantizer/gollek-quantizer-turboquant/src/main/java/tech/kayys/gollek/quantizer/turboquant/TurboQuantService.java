package tech.kayys.gollek.quantizer.turboquant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Unified service for TurboQuant multi-format quantization operations.
 * 
 * This service provides:
 * - Quantization format auto-detection (GPTQ, AWQ, AutoRound, GGUF, BnB, HQQ, SqueezeLLM)
 * - TurboQuant online vector quantization (arXiv:2504.19874)
 * - KV cache quantization
 * - Model inspection and reporting
 * 
 * TurboQuant supports two variants:
 * - TurboQuant_mse: MSE-optimal, minimizes E[‖x − x̃‖²]
 * - TurboQuant_prod: Unbiased inner product, E[⟨y,x̃⟩] = ⟨y,x⟩
 * 
 * Usage:
 * <pre>{@code
 * TurboQuantService service = new TurboQuantService();
 * 
 * // Auto-detect quantization format
 * QuantizerRegistry.Detection detection = service.detectFormat(modelPath);
 * System.out.println("Detected: " + detection.format());
 * 
 * // Apply TurboQuant to a vector
 * TurboQuantConfig config = TurboQuantConfig.mse4bit(4096);
 * TurboQuantEngine engine = new TurboQuantEngine(config);
 * float[] vector = ...;
 * TurboQuantEngine.QuantProdResult quantized = engine.quantizeProd(vector);
 * float[] output = new float[vector.length];
 * engine.dequantizeProd(quantized, output);
 * }<pre>
 */
public class TurboQuantService implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(TurboQuantService.class);

    private boolean initialized = false;

    /**
     * Initialize the TurboQuant service.
     */
    public TurboQuantService() {
        log.info("TurboQuantService initialized");
        this.initialized = true;
    }

    // ── Format Detection ──────────────────────────────────────────────────────

    /**
     * Auto-detects the quantization format of a model.
     * 
     * @param modelPath Path to model directory or file
     * @return Detection result with format, confidence, and evidence
     * @throws IOException if detection fails
     */
    public QuantizerRegistry.Detection detectFormat(Path modelPath) throws IOException {
        log.info("Detecting quantization format for: {}", modelPath);
        QuantizerRegistry.Detection detection = QuantizerRegistry.detect(modelPath);
        log.info("Detected format: {} (confidence: {}, evidence: {})", 
                detection.format(), detection.confidence(), detection.evidence());
        return detection;
    }

    /**
     * Prints a formatted detection report to stdout.
     */
    public void printDetectionReport(Path modelPath) throws IOException {
        QuantizerRegistry.printReport(modelPath);
    }

    /**
     * Returns all supported quantization formats.
     */
    public void printSupportedFormats() {
        QuantizerRegistry.printSupportedFormats();
    }

    // ── TurboQuant Operations ────────────────────────────────────────────────

    /**
     * Creates a TurboQuant engine for the given configuration.
     * 
     * @param config TurboQuant configuration
     * @return Configured TurboQuantEngine
     */
    public TurboQuantEngine createEngine(TurboQuantConfig config) {
        log.info("Creating TurboQuant engine: {}", config);
        return new TurboQuantEngine(config);
    }

    /**
     * Quantizes a vector using TurboQuant.
     * 
     * @param vector Input vector
     * @param config TurboQuant configuration
     * @return Quantized result
     */
    public TurboQuantEngine.QuantProdResult quantizeVector(float[] vector, TurboQuantConfig config) {
        TurboQuantEngine engine = createEngine(config);
        return engine.quantizeProd(vector);
    }

    /**
     * Dequantizes a TurboQuant result back to FP32.
     * 
     * @param quantized Quantized result
     * @param config TurboQuant configuration
     * @return Dequantized vector
     */
    public void dequantizeVector(TurboQuantEngine.QuantProdResult quantized, TurboQuantConfig config, float[] output) {
        TurboQuantEngine engine = createEngine(config);
        engine.dequantizeProd(quantized, output);
    }

    // ── KV Cache Quantization ────────────────────────────────────────────────

    /**
     * Creates a KV cache quantizer for transformer models.
     * 
     * @param config TurboQuant configuration with KV cache settings
     * @return Configured TurboQuantKVCache
     */
    public TurboQuantKVCache createKvCache(TurboQuantConfig config, int maxSeqLen) {
        log.info("Creating KV cache quantizer: {}, maxSeqLen={}", config, maxSeqLen);
        return new TurboQuantKVCache(config, maxSeqLen);
    }

    // ── Multi-Format Dequantization ──────────────────────────────────────────

    /**
     * Creates the appropriate dequantizer for a detected format.
     * 
     * @param format Detected quantization format
     * @return Dequantizer instance (or null if not supported)
     */
    public Object createDequantizer(QuantizerRegistry.QuantFormat format) {
        return switch (format) {
            case GPTQ -> null; // Use GPTQ module
            case AWQ -> null;  // Use AWQ module
            case BNB_NF4, BNB_INT8 -> new BnBDequantizer();
            case HQQ -> new HQQDequantizer(4, 128, HQQDequantizer.QuantAxis.INPUT); // Default: 4-bit, groupSize=128
            case SQUEEZELLM -> new SqueezeLLMDequantizer();
            case GGUF -> new GGUFDequantizer();
            default -> null;
        };
    }

    // ── Model Inspection ─────────────────────────────────────────────────────

    /**
     * Inspects a quantized model and returns metadata.
     * 
     * @param modelPath Path to model
     * @return Model inspection result
     * @throws IOException if inspection fails
     */
    public ModelInspectionResult inspect(Path modelPath) throws IOException {
        log.info("Inspecting model: {}", modelPath);
        QuantizerRegistry.Detection detection = detectFormat(modelPath);
        
        ModelInspectionResult result = new ModelInspectionResult(
                detection,
                modelPath,
                estimateModelSize(modelPath)
        );
        
        log.info("Inspection complete: {}", result);
        return result;
    }

    private long estimateModelSize(Path modelPath) throws IOException {
        if (java.nio.file.Files.isDirectory(modelPath)) {
            try (var stream = java.nio.file.Files.walk(modelPath)) {
                return stream.filter(java.nio.file.Files::isRegularFile)
                        .mapToLong(p -> {
                            try {
                                return java.nio.file.Files.size(p);
                            } catch (IOException e) {
                                return 0;
                            }
                        }).sum();
            }
        } else {
            return java.nio.file.Files.size(modelPath);
        }
    }

    // ── Resource Management ───────────────────────────────────────────────────

    @Override
    public void close() {
        log.info("Closing TurboQuantService");
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
            QuantizerRegistry.Detection detection,
            Path modelPath,
            long totalSizeBytes) {
        
        public double totalSizeMB() {
            return totalSizeBytes / (1024.0 * 1024.0);
        }

        @Override
        public String toString() {
            return "ModelInspectionResult{format=%s, confidence=%s, size=%.2f MB}"
                    .formatted(detection.format(), detection.confidence(), totalSizeMB());
        }
    }
}
