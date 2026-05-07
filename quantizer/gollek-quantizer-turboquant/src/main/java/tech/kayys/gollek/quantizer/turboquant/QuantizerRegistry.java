package tech.kayys.gollek.quantizer.turboquant;

import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Supplier;

/**
 * Unified registry and auto-detection hub for all supported quantization
 * formats.
 *
 * Supported formats and their detection signatures:
 * ┌─────────────────┬──────────────────────────────────────────────────────────┐
 * │ Format │ Detection heuristic │
 * ├─────────────────┼──────────────────────────────────────────────────────────┤
 * │ GPTQ │ .qweight INT32 [outF/pack, inF] + .scales FP16 │
 * │ AWQ │ .qweight INT32 [inF/pack, outF] + .scales FP16 │
 * │ AutoRound │ .weight INT32 + .scale FP32 OR GPTQ names + AR meta │
 * │ GGUF/GGML │ file extension .gguf OR .ggml, tensor dtype Q4_K etc. │
 * │ BitsAndBytes │ .weight INT8/FP16 + .weight_format "nf4"/"int8" meta │
 * │ HQQ │ .W INT32 + .scale FP16 + .zero FP16 (HQQ tensor names) │
 * │ SqueezeLLM │ .weight_nnz + .dense_idx tensors (sparse format) │
 * └─────────────────┴──────────────────────────────────────────────────────────┘
 *
 * Usage:
 * 
 * <pre>{@code
 * QuantFormat fmt = QuantizerRegistry.detect(modelDir);
 * System.out.println("Detected: " + fmt);
 *
 * // Or use the CLI dispatch helper
 * QuantizerRegistry.run(modelDir, outputPath, ConversionMode.CONVERT);
 * }</pre>
 */
public class QuantizerRegistry {

    private static final Logger log = LoggerFactory.getLogger(QuantizerRegistry.class);

    /** All recognised quantization formats, in detection priority order. */
    public enum QuantFormat {
        GPTQ("GPTQ", "AutoGPTQ / GPTQ-for-LLaMa"),
        AWQ("AWQ", "AutoAWQ / Activation-Aware Weight Quantization"),
        AUTOROUND("AutoRound", "Intel AutoRound / Neural Compressor"),
        GGUF("GGUF", "GGML/GGUF Q4_K_M, Q5_K_S, Q8_0 etc."),
        BNB_NF4("BnB-NF4", "BitsAndBytes NormalFloat-4"),
        BNB_INT8("BnB-INT8", "BitsAndBytes LLM.int8()"),
        HQQ("HQQ", "Half-Quadratic Quantization"),
        SQUEEZELLM("SqueezeLLM", "Sparse + Dense quantization"),
        UNKNOWN("Unknown", "Format not recognised");

        public final String shortName;
        public final String description;

        QuantFormat(String shortName, String description) {
            this.shortName = shortName;
            this.description = description;
        }
    }

    /** Confidence level returned alongside detection result. */
    public record Detection(QuantFormat format, Confidence confidence, String evidence) {
        public enum Confidence {
            HIGH, MEDIUM, LOW
        }

        public boolean isKnown() {
            return format != QuantFormat.UNKNOWN;
        }
    }

    // ── Detection Entry Point ─────────────────────────────────────────────────

    /**
     * Auto-detects the quantization format from a model directory or single file.
     *
     * Detection order (short-circuits on first HIGH-confidence match):
     * 1. File extension (.gguf → GGUF immediately)
     * 2. __metadata__ quant_method key
     * 3. Tensor dtype + name signature analysis
     * 4. Tensor shape ratio heuristics
     */
    public static Detection detect(Path modelPath) throws IOException {
        // Single .gguf file
        if (modelPath.toString().endsWith(".gguf")
                || modelPath.toString().endsWith(".ggml")) {
            return new Detection(QuantFormat.GGUF, Detection.Confidence.HIGH, "file extension");
        }

        // Probe first safetensor shard
        Path firstShard = findFirstShard(modelPath);
        if (firstShard == null) {
            return new Detection(QuantFormat.UNKNOWN, Detection.Confidence.LOW,
                    "no .safetensors or .gguf files found");
        }

        // Read header using safetensor-loader infrastructure
        SafetensorHeader header = readSafetensorHeader(firstShard);
        if (header == null) {
            return new Detection(QuantFormat.UNKNOWN, Detection.Confidence.LOW,
                    "failed to parse safetensor header");
        }

        Map<String, String> meta = header.fileMetadata();

        // ── Step 1: explicit metadata key ─────────────────────────────────
        Detection md = detectFromMetadata(meta);
        if (md.confidence() == Detection.Confidence.HIGH)
            return md;

        // ── Step 2: tensor name/dtype signatures ──────────────────────────
        Detection sig = detectFromTensors(header);
        if (sig.confidence() != Detection.Confidence.LOW)
            return sig;

        // ── Step 3: return metadata result if we had medium confidence ─────
        if (md.confidence() == Detection.Confidence.MEDIUM)
            return md;

        return sig;
    }

    /**
     * Reads safetensor header using standalone FFM approach.
     */
    private static SafetensorHeader readSafetensorHeader(Path filePath) throws IOException {
        // Use simple JSON parsing approach without requiring full safetensor-loader
        try (var channel = java.nio.channels.FileChannel.open(filePath, java.nio.file.StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < 8) return null;

            // Read 8-byte LE header size
            java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(8);
            buf.order(java.nio.ByteOrder.LITTLE_ENDIAN);
            channel.read(buf);
            buf.flip();
            long headerSize = buf.getLong();

            if (headerSize <= 0 || headerSize > 100_000_000L) return null;

            // Read JSON header
            byte[] jsonBytes = new byte[(int) headerSize];
            channel.read(java.nio.ByteBuffer.wrap(jsonBytes));
            String json = new String(jsonBytes, java.nio.charset.StandardCharsets.UTF_8);

            // Parse with Jackson
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            @SuppressWarnings("unchecked")
            Map<String, Object> root = mapper.readValue(json, Map.class);

            // Build SafetensorHeader
            return buildHeaderFromJson(root);
        }
    }

    /**
     * Builds a SafetensorHeader from parsed JSON.
     */
    private static SafetensorHeader buildHeaderFromJson(Map<String, Object> root) {
        Map<String, String> metadata = new java.util.HashMap<>();
        Map<String, SafetensorTensorInfo> tensors = new java.util.LinkedHashMap<>();

        for (var entry : root.entrySet()) {
            if ("__metadata__".equals(entry.getKey())) {
                if (entry.getValue() instanceof Map<?, ?> meta) {
                    meta.forEach((k, v) -> metadata.put(k.toString(), v != null ? v.toString() : ""));
                }
            } else if (entry.getValue() instanceof Map<?, ?> tensorMap) {
                String dtype = getString(tensorMap, "dtype");
                List<Long> shape = getShape(tensorMap);
                List<Long> offsets = getOffsets(tensorMap);
                if (dtype != null && shape != null && offsets != null) {
                    // Create a SafetensorTensorInfo from the parsed data
                    long[] shapeArray = shape.stream().mapToLong(Long::longValue).toArray();
                    long[] offsetsArray = offsets.stream().mapToLong(Long::longValue).toArray();
                    tensors.put(entry.getKey(), new SafetensorTensorInfo(entry.getKey(), dtype, shapeArray, offsetsArray));
                }
            }
        }

        return SafetensorHeader.of(8L, tensors, metadata);
    }

    @SuppressWarnings("unchecked")
    private static String getString(Map<?, ?> map, String key) {
        Object v = map.get(key);
        return v != null ? v.toString() : null;
    }

    @SuppressWarnings("unchecked")
    private static List<Long> getShape(Map<?, ?> map) {
        Object v = map.get("shape");
        if (!(v instanceof List<?> list)) return null;
        return list.stream().map(e -> ((Number) e).longValue()).toList();
    }

    @SuppressWarnings("unchecked")
    private static List<Long> getOffsets(Map<?, ?> map) {
        Object v = map.get("data_offsets");
        if (!(v instanceof List<?> list)) return null;
        return list.stream().map(e -> ((Number) e).longValue()).toList();
    }

    private static Detection detectFromMetadata(Map<String, String> meta) {
        String method = meta.getOrDefault("quant_method",
                meta.getOrDefault("quantization_method",
                        meta.getOrDefault("quantize_config", "")))
                .toLowerCase();

        if (method.contains("gptq"))
            return hi(QuantFormat.GPTQ, "quant_method=gptq");
        if (method.contains("awq"))
            return hi(QuantFormat.AWQ, "quant_method=awq");
        if (method.contains("autoround") || method.contains("auto_round"))
            return hi(QuantFormat.AUTOROUND, "quant_method=autoround");
        if (method.contains("hqq"))
            return hi(QuantFormat.HQQ, "quant_method=hqq");
        if (method.contains("squeezellm"))
            return hi(QuantFormat.SQUEEZELLM, "quant_method=squeezellm");
        if (method.contains("bitsandbytes") || method.contains("bnb"))
            return detectBnbVariant(meta);

        // Weaker signals
        String wBit = meta.getOrDefault("w_bit", "");
        String bits = meta.getOrDefault("bits", "");
        if (!wBit.isEmpty() || !bits.isEmpty()) {
            // Could be AWQ or GPTQ — need tensor analysis to distinguish
            return new Detection(QuantFormat.UNKNOWN, Detection.Confidence.LOW,
                    "bits found in metadata but format ambiguous");
        }

        return new Detection(QuantFormat.UNKNOWN, Detection.Confidence.LOW, "no quant metadata");
    }

    private static Detection detectBnbVariant(Map<String, String> meta) {
        String wFmt = meta.getOrDefault("weight_format",
                meta.getOrDefault("bnb_4bit_quant_type", "")).toLowerCase();
        if (wFmt.contains("nf4") || wFmt.contains("fp4"))
            return hi(QuantFormat.BNB_NF4, "weight_format=" + wFmt);
        if (wFmt.contains("int8"))
            return hi(QuantFormat.BNB_INT8, "weight_format=int8");
        return new Detection(QuantFormat.BNB_NF4, Detection.Confidence.MEDIUM,
                "bnb detected, defaulting to NF4");
    }

    private static Detection detectFromTensors(SafetensorHeader header) {
        boolean hasQweight = false;
        boolean hasWeight = false;
        boolean hasScalesFp16 = false;
        boolean hasScaleFp32 = false;
        boolean hasZeroFp16 = false; // HQQ uses .zero tensors
        boolean hasWeightNnz = false; // SqueezeLLM sparse
        boolean hasGIdx = false; // GPTQ act-order
        boolean hasWeightFormat = false; // BnB
        long qweightRows = 0, qweightCols = 0;

        for (var e : header.tensors().entrySet()) {
            String name = e.getKey();
            SafetensorTensorInfo info = e.getValue();
            SafetensorDType dtype = info.dtype();
            long[] shape = info.shape();

            if (name.endsWith(".qweight") && dtype == SafetensorDType.I32) {
                hasQweight = true;
                if (shape != null && shape.length == 2) {
                    qweightRows = shape[0];
                    qweightCols = shape[1];
                }
            }
            if ((name.endsWith(".weight") || name.endsWith(".W")) && dtype == SafetensorDType.I32)
                hasWeight = true;
            if (name.endsWith(".scales") && dtype == SafetensorDType.F16)
                hasScalesFp16 = true;
            if (name.endsWith(".scale") && dtype == SafetensorDType.F32)
                hasScaleFp32 = true;
            if (name.endsWith(".zero") && dtype == SafetensorDType.F16)
                hasZeroFp16 = true;
            if (name.endsWith(".weight_nnz"))
                hasWeightNnz = true;
            if (name.endsWith(".g_idx"))
                hasGIdx = true;
            if (name.endsWith(".weight") && (dtype == SafetensorDType.F16 || dtype == SafetensorDType.I8))
                hasWeightFormat = true;
        }

        // SqueezeLLM has unique sparse tensors
        if (hasWeightNnz)
            return hi(QuantFormat.SQUEEZELLM, "weight_nnz tensor found");

        // HQQ: INT32 weights + FP16 zero tensors
        if ((hasWeight || hasQweight) && hasZeroFp16)
            return hi(QuantFormat.HQQ, "FP16 .zero tensors found (HQQ signature)");

        // AutoRound native: weight INT32 + scale FP32
        if (hasWeight && hasScaleFp32)
            return hi(QuantFormat.AUTOROUND, "INT32 weight + FP32 scale (AutoRound native)");

        // GPTQ vs AWQ disambiguation via qweight shape ratio
        if (hasQweight && hasScalesFp16) {
            // GPTQ: qweight[outF/pack, inF] → rows * packFactor ≈ outF, cols = inF
            // AWQ: qweight[inF/pack, outF] → rows * packFactor ≈ inF, cols = outF
            // Usually inF ≥ outF for most attention layers — shapes differ by transpose
            if (hasGIdx)
                return hi(QuantFormat.GPTQ, "g_idx tensor present (GPTQ act-order)");
            // Without g_idx we check if rows < cols (AWQ) or rows > cols (GPTQ)
            // Both can be ambiguous for square matrices, so medium confidence
            if (qweightRows > 0 && qweightCols > 0) {
                // AWQ: rows = inF/8, cols = outF. Typically outF < inF → cols < rows
                // GPTQ: rows = outF/8, cols = inF. Typically inF > outF → cols > rows
                boolean probablyGptq = qweightCols >= qweightRows;
                return new Detection(
                        probablyGptq ? QuantFormat.GPTQ : QuantFormat.AWQ,
                        Detection.Confidence.MEDIUM,
                        "shape heuristic rows=%d cols=%d".formatted(qweightRows, qweightCols));
            }
            return new Detection(QuantFormat.GPTQ, Detection.Confidence.MEDIUM,
                    "qweight+scales FP16 found, defaulting to GPTQ");
        }

        // BnB: FP16/INT8 weight without packing
        if (hasWeightFormat)
            return new Detection(QuantFormat.BNB_NF4, Detection.Confidence.MEDIUM,
                    "FP16/INT8 weight tensors (likely BitsAndBytes)");

        return new Detection(QuantFormat.UNKNOWN, Detection.Confidence.LOW, "no match");
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private static Path findFirstShard(Path dir) {
        if (!dir.toFile().isDirectory()) {
            return dir.toString().endsWith(".safetensors") ? dir : null;
        }
        var files = dir.toFile().listFiles(
                f -> f.isFile() && f.getName().endsWith(".safetensors"));
        if (files == null || files.length == 0)
            return null;
        return Arrays.stream(files)
                .min(Comparator.comparing(f -> f.getName()))
                .map(java.io.File::toPath).orElse(null);
    }

    private static Detection hi(QuantFormat fmt, String evidence) {
        return new Detection(fmt, Detection.Confidence.HIGH, evidence);
    }

    /** Prints a formatted detection report to stdout. */
    public static void printReport(Path modelPath) throws IOException {
        Detection d = detect(modelPath);
        System.out.println("┌─── Quantization Format Detection ───────────────────────┐");
        System.out.printf("│  Path:       %-44s│%n", modelPath.getFileName());
        System.out.printf("│  Format:     %-44s│%n", d.format().shortName + " — " + d.format().description);
        System.out.printf("│  Confidence: %-44s│%n", d.confidence());
        System.out.printf("│  Evidence:   %-44s│%n", d.evidence());
        System.out.println("└──────────────────────────────────────────────────────────┘");
    }

    /** Returns all supported formats with descriptions. */
    public static void printSupportedFormats() {
        System.out.println("Supported quantization formats:");
        for (QuantFormat f : QuantFormat.values()) {
            if (f == QuantFormat.UNKNOWN)
                continue;
            System.out.printf("  %-12s  %s%n", f.shortName, f.description);
        }
    }
}
