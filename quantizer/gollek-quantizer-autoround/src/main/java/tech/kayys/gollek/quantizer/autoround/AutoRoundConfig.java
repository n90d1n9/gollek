package tech.kayys.gollek.quantizer.autoround;

/**
 * Configuration for AutoRound (Intel Neural Compressor AutoRound) quantization.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ AutoRound vs GPTQ vs AWQ — Core Algorithm Distinctions │
 * ├──────────────┬────────────────────────────────────────────────────────-─┤
 * │ GPTQ │ Second-order (Hessian) weight update per layer │
 * │ │ Minimizes ||WX − W_q X||² using OBQ framework │
 * ├──────────────┼─────────────────────────────────────────────────────────┤
 * │ AWQ │ Protects salient input channels via learned scale s │
 * │ │ W' = W × diag(s)⁻¹, no optimization of rounding │
 * ├──────────────┼─────────────────────────────────────────────────────────┤
 * │ AutoRound │ Optimizes BOTH the rounding decisions AND the scales │
 * │ │ using SignSGD on a block-wise reconstruction objective │
 * │ │ Learns: v ∈ {0,1} (round-up vs round-down per weight) │
 * │ │ s (scale per group) │
 * │ │ z (zero-point per group) │
 * └──────────────┴─────────────────────────────────────────────────────────┘
 *
 * AutoRound Algorithm (simplified):
 * For each transformer block B:
 * 1. Collect input activations X from calibration data
 * 2. Initialize s, z from min/max of each weight group
 * 3. For T iterations (default 200):
 * a. q(W) = clamp(round(W/s + 0.5*v) − z, 0, 2^b − 1) [quantize]
 * b. W̃ = (q(W) + z) × s [dequant]
 * c. L = ||BX − B̃X||² (block reconstruction loss)
 * d. Gradient step on s, z via Adam
 * e. v update via SignSGD: v ← v − η × sign(∂L/∂v)
 *
 * AutoRound Tensor Layout (Intel AutoRound / auto-round library):
 * ──────────────────────────────────────────────────────────────────────────
 * Asymmetric (standard):
 * layer.weight INT32 [outF/pack, inF] packed quantized weights
 * layer.scale FP32 [outF, inF/group] per-group scales
 * layer.zp INT32 [outF, inF/group] per-group zero-points (INT32, not packed)
 *
 * Symmetric (scale-only, no zero-point):
 * layer.weight INT32 [outF/pack, inF]
 * layer.scale FP32 [outF, inF/group]
 * (no zp tensor)
 *
 * Layout Notes:
 * - qweight is packed along the OUTPUT dimension (same as GPTQ)
 * - Groups are along the INPUT dimension (same as AWQ)
 * - Scales and zero-points are FP32 / INT32 (not FP16!)
 * - zero-points are stored as raw integer values (not packed INT4)
 * - The tensor names use "weight", "scale", "zp" (not "qweight", "scales",
 * "qzeros")
 *
 * GPTX format variant:
 * Some AutoRound exports use the GPTQ tensor name convention:
 * layer.qweight / layer.scales / layer.qzeros
 * with the AutoRound packing but GPTQ naming. The loader auto-detects.
 *
 * Backend variants:
 * AUTOROUND — Intel AutoRound native format (scale=FP32, zp=INT32 unpacked)
 * GPTQ_COMPAT — AutoRound with GPTQ tensor names (scales=FP16, qzeros=INT32
 * packed)
 * IPEX_WEIGHT_ONLY — Intel Extension for PyTorch weight-only format
 */
public record AutoRoundConfig(
        /** Bit width: 2, 3, 4, or 8 */
        int bits,

        /** Group size along INPUT dimension (typically 128) */
        int groupSize,

        /** Whether zero-points are stored (false = symmetric) */
        boolean hasZeroPoint,

        /** Scale tensor dtype stored on disk */
        ScaleDtype scaleDtype,

        /** Packing convention for qweight */
        PackFormat packFormat,

        /** Target dequantized dtype */
        String dequantDtype,

        /** Number of optimization iterations for rounding (default 200) */
        int numIters,

        /** Learning rate for scale/zero-point optimization */
        double learningRate,

        /** Whether to use Adam optimizer (false = SignSGD) */
        boolean useAdam,

        /** Number of calibration samples */
        int numSamples,

        /** Sequence length for calibration */
        int seqLen,

        /** Backend target for export (exllamav2, marlin, ipex) */
        String backendTarget,

        /** Path to calibration dataset (optional) */
        String calibrationDataPath) {

    /** Dtype of the scale tensor in the saved file */
    public enum ScaleDtype {
        /** FP32 scales — native AutoRound format */
        FLOAT32,
        /** FP16 scales — GPTQ-compat export */
        FLOAT16,
        /** BF16 scales — some newer AutoRound exports */
        BFLOAT16
    }

    /**
     * Packing convention for weight INT32 words.
     *
     * AUTOROUND_NATIVE:
     * qweight[outF/pack, inF] — packed along output dimension, LSB-first
     * Identical bit layout to GPTQ: weights packed column-by-column.
     *
     * GPTQ_COMPAT:
     * Identical layout to GPTQ qweight. AutoRound writes GPTQ-compatible
     * exports when targeting llama.cpp / vLLM consumption.
     *
     * ITREX:
     * Intel Extension for PyTorch packing — row-major, different endianness
     * within the INT32. Used by Intel itrex / IPEX backends.
     */
    public enum PackFormat {
        AUTOROUND_NATIVE,
        GPTQ_COMPAT,
        ITREX
    }

    // ── Preset Configurations ─────────────────────────────────────────────────

    /** AutoRound native 4-bit asymmetric, group=128, FP32 scales */
    public static AutoRoundConfig autoRound4bit() {
        return new AutoRoundConfig(4, 128, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound 4-bit symmetric (no zero-point), FP32 scales */
    public static AutoRoundConfig autoRound4bitSymmetric() {
        return new AutoRoundConfig(4, 128, false, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound GPTQ-compatible export (FP16 scales, packed qzeros) */
    public static AutoRoundConfig autoRoundGptqCompat() {
        return new AutoRoundConfig(4, 128, true, ScaleDtype.FLOAT16,
                PackFormat.GPTQ_COMPAT, "float32", 200, 0.001, false, 128, 2048, "gptq", null);
    }

    /** AutoRound 4-bit with group=32 (maximum quality) */
    public static AutoRoundConfig autoRound4bitGroup32() {
        return new AutoRoundConfig(4, 32, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound 2-bit with group=128 */
    public static AutoRoundConfig autoRound2bit() {
        return new AutoRoundConfig(2, 128, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound 8-bit (INT8) with group=128 */
    public static AutoRoundConfig autoRound8bit() {
        return new AutoRoundConfig(8, 128, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound 4-bit with FP16 output for memory efficiency */
    public static AutoRoundConfig autoRound4bitFP16() {
        return new AutoRoundConfig(4, 128, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float16", 200, 0.001, false, 128, 2048, "exllamav2", null);
    }

    /** AutoRound 4-bit optimized for Marlin backend */
    public static AutoRoundConfig autoRound4bitMarlin() {
        return new AutoRoundConfig(4, 128, true, ScaleDtype.FLOAT32,
                PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, true, 128, 2048, "marlin", null);
    }

    /** Custom configuration builder */
    public static Builder builder() {
        return new Builder();
    }

    // ── Derived Properties ────────────────────────────────────────────────────

    /** INT(bits) values packed per INT32 word (e.g., 8 for 4-bit) */
    public int packFactor() {
        return 32 / bits;
    }

    /** Bitmask for one quantized value (e.g., 0xF for 4-bit) */
    public int quantMask() {
        return (1 << bits) - 1;
    }

    /** Maximum representable quantized value (unsigned) */
    public int maxQuantValue() {
        return (1 << bits) - 1;
    }

    /**
     * Zero-point used when {@code hasZeroPoint == false} (symmetric).
     * For 4-bit: mid-range = 8, for 8-bit: 128.
     */
    public int implicitZeroPoint() {
        return hasZeroPoint ? 0 : (1 << (bits - 1));
    }

    /** Memory compression ratio vs FP32 baseline */
    public double compressionRatio() {
        return 32.0 / bits;
    }

    /** Whether scales are stored as native FP32 (AutoRound native format) */
    public boolean hasFp32Scales() {
        return scaleDtype == ScaleDtype.FLOAT32;
    }

    /** Whether scales are stored as FP16 (GPTQ-compat format) */
    public boolean hasFp16Scales() {
        return scaleDtype == ScaleDtype.FLOAT16;
    }

    /** Whether this uses AutoRound native packing */
    public boolean isNativeFormat() {
        return packFormat == PackFormat.AUTOROUND_NATIVE;
    }

    /** Whether this uses GPTQ-compatible packing */
    public boolean isGptqCompatFormat() {
        return packFormat == PackFormat.GPTQ_COMPAT;
    }

    /** Whether this uses Intel ITREX packing */
    public boolean isItrexFormat() {
        return packFormat == PackFormat.ITREX;
    }

    @Override
    public String toString() {
        return ("AutoRoundConfig{bits=%d, groupSize=%d, hasZP=%b, scale=%s, pack=%s, out='%s', iters=%d, lr=%.4f, adam=%b, backend='%s'}")
                .formatted(bits, groupSize, hasZeroPoint, scaleDtype, packFormat, dequantDtype, numIters, learningRate, useAdam, backendTarget);
    }

    // ── Builder Pattern ───────────────────────────────────────────────────────

    /**
     * Builder for constructing AutoRoundConfig instances with custom settings.
     */
    public static class Builder {
        private int bits = 4;
        private int groupSize = 128;
        private boolean hasZeroPoint = true;
        private ScaleDtype scaleDtype = ScaleDtype.FLOAT32;
        private PackFormat packFormat = PackFormat.AUTOROUND_NATIVE;
        private String dequantDtype = "float32";
        private int numIters = 200;
        private double learningRate = 0.001;
        private boolean useAdam = false;
        private int numSamples = 128;
        private int seqLen = 2048;
        private String backendTarget = "exllamav2";
        private String calibrationDataPath = null;

        public Builder bits(int bits) {
            this.bits = bits;
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder hasZeroPoint(boolean hasZeroPoint) {
            this.hasZeroPoint = hasZeroPoint;
            return this;
        }

        public Builder scaleDtype(ScaleDtype scaleDtype) {
            this.scaleDtype = scaleDtype;
            return this;
        }

        public Builder packFormat(PackFormat packFormat) {
            this.packFormat = packFormat;
            return this;
        }

        public Builder dequantDtype(String dequantDtype) {
            this.dequantDtype = dequantDtype;
            return this;
        }

        public Builder numIters(int numIters) {
            this.numIters = numIters;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder useAdam(boolean useAdam) {
            this.useAdam = useAdam;
            return this;
        }

        public Builder numSamples(int numSamples) {
            this.numSamples = numSamples;
            return this;
        }

        public Builder seqLen(int seqLen) {
            this.seqLen = seqLen;
            return this;
        }

        public Builder backendTarget(String backendTarget) {
            this.backendTarget = backendTarget;
            return this;
        }

        public Builder calibrationDataPath(String calibrationDataPath) {
            this.calibrationDataPath = calibrationDataPath;
            return this;
        }

        public AutoRoundConfig build() {
            if (bits < 2 || bits > 8 || (bits != 2 && bits != 3 && bits != 4 && bits != 8)) {
                throw new IllegalArgumentException("Invalid bits: " + bits + ". Must be 2, 3, 4, or 8.");
            }
            return new AutoRoundConfig(bits, groupSize, hasZeroPoint, scaleDtype, packFormat, dequantDtype,
                    numIters, learningRate, useAdam, numSamples, seqLen, backendTarget, calibrationDataPath);
        }
    }
}
