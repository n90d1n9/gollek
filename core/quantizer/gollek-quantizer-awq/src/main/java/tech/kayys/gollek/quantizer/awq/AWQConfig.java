package tech.kayys.gollek.quantizer.awq;

/**
 * Configuration for AWQ (Activation-Aware Weight Quantization).
 *
 * AWQ differs fundamentally from GPTQ:
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ GPTQ: minimizes reconstruction error per layer (Hessian) │
 * │ AWQ: protects the ~1% of weights that matter most │
 * │ to activations — scales input channels before quant │
 * └─────────────────────────────────────────────────────────────┘
 *
 * AWQ Key Insight:
 * Not all weights are equally important. Weights corresponding to
 * high-magnitude activation channels cause disproportionate error
 * when quantized. AWQ identifies these "salient" channels from a
 * calibration dataset and applies a per-channel scale factor s[c]
 * to the weight BEFORE quantization, effectively giving those
 * channels more precision.
 *
 * Mathematical formulation:
 * Pre-quant transform: W' = W × diag(s)^(-1)
 * Compensating input: X' = X × diag(s)
 * Equivalence: W'X' = W × diag(s)^(-1) × diag(s) × X = WX ✓
 *
 * AWQ Tensor Layout (AutoAWQ convention):
 * layer.qweight INT32 [inF/pack, outF] packed INT4 weights
 * layer.qzeros INT32 [inF/pack, outF/pack] packed INT4 zero-points (optional)
 * layer.scales FP16 [inF/group, outF] per-group scale factors
 *
 * Note: AWQ's qweight is transposed relative to GPTQ's layout:
 * GPTQ: qweight[outF/pack, inF]
 * AWQ: qweight[inF/pack, outF] ← transposed
 *
 * Dequantization:
 * w_fp32[i, j] = (q[i, j] - zero[g_i, j]) × scale[g_i, j]
 * where g_i = i / group_size (group over INPUT dimension)
 *
 * Supported variants:
 * - Standard AWQ w/ zeros (AutoAWQ gemm kernel format)
 * - Symmetric AWQ w/o zeros (zero-point = 8 implicitly for 4-bit)
 * - AWQ-Gemv format (different packing for gemv kernels)
 */
public record AWQConfig(
        /** Bit-width. AWQ almost exclusively uses 4-bit. */
        int bits,

        /** Group size along the INPUT dimension (typically 128). */
        int groupSize,

        /**
         * Which kernel format the weights were packed for.
         * Affects packing order within INT32 words.
         * GEMM — optimized for batch matrix multiply (default, AutoAWQ)
         * GEMV — optimized for single-token (vector × matrix) inference
         */
        KernelFormat kernelFormat,

        /** Whether zero-points are stored (false = symmetric, zero=8 for 4-bit) */
        boolean hasZeros,

        /** Target dtype for dequantized output */
        String dequantDtype,

        /** Whether to use exllama v2 optimized layout */
        boolean exllamaV2,

        /** Number of calibration samples used during quantization */
        int numSamples,

        /** Sequence length for calibration */
        int seqLen,

        /** Whether activation scaling was applied during quantization */
        boolean activationAware,

        /** Path to calibration dataset (optional) */
        String calibrationDataPath) {

    /** Packing format variants produced by different AWQ backends */
    public enum KernelFormat {
        /**
         * AutoAWQ GEMM format (most common).
         * Packs weights column-major: consecutive output features packed together.
         * Used by: llama.cpp AWQ, vLLM, AutoAWQ default.
         */
        GEMM,

        /**
         * AutoAWQ GEMV format.
         * Packs weights row-major: consecutive input features packed together.
         * Used by: single-token generation, small batch sizes.
         */
        GEMV,

        /**
         * ExLlama v2 AWQ format (marlin kernel).
         * Interleaved packing optimized for GPU memory access patterns.
         */
        MARLIN
    }

    // ── Preset Configurations ─────────────────────────────────────────────────

    /** Standard AutoAWQ 4-bit with zeros, group=128, GEMM format */
    public static AWQConfig awq4bit() {
        return new AWQConfig(4, 128, KernelFormat.GEMM, true, "float32",
                false, 128, 2048, true, null);
    }

    /** Symmetric AWQ 4-bit without zeros (zero-point implicitly = 8) */
    public static AWQConfig awq4bitSymmetric() {
        return new AWQConfig(4, 128, KernelFormat.GEMM, false, "float32",
                false, 128, 2048, true, null);
    }

    /** AWQ GEMV format (single-token inference) */
    public static AWQConfig awq4bitGemv() {
        return new AWQConfig(4, 128, KernelFormat.GEMV, true, "float32",
                false, 128, 2048, true, null);
    }

    /** AWQ with group size 64 (higher quality, more overhead) */
    public static AWQConfig awq4bitGroup64() {
        return new AWQConfig(4, 64, KernelFormat.GEMM, true, "float32",
                false, 128, 2048, true, null);
    }

    /** AWQ with FP16 output for memory efficiency */
    public static AWQConfig awq4bitFP16() {
        return new AWQConfig(4, 128, KernelFormat.GEMM, true, "float16",
                false, 128, 2048, true, null);
    }

    /** AWQ optimized for exllama v2 */
    public static AWQConfig awq4bitExllamaV2() {
        return new AWQConfig(4, 128, KernelFormat.GEMM, true, "float32",
                true, 128, 2048, true, null);
    }

    /** AWQ with Marlin kernel format */
    public static AWQConfig awq4bitMarlin() {
        return new AWQConfig(4, 128, KernelFormat.MARLIN, true, "float32",
                false, 128, 2048, true, null);
    }

    /** Custom configuration builder */
    public static Builder builder() {
        return new Builder();
    }

    // ── Derived Properties ────────────────────────────────────────────────────

    /** INT4 values packed per INT32 word (8 for 4-bit) */
    public int packFactor() {
        return 32 / bits;
    }

    /** Bitmask for one quantized element (0xF for 4-bit) */
    public int quantMask() {
        return (1 << bits) - 1;
    }

    /**
     * Implicit zero-point used when {@code hasZeros == false}.
     * For 4-bit symmetric: 2^(bits-1) = 8
     */
    public int implicitZero() {
        return hasZeros ? 0 : (1 << (bits - 1));
    }

    /** Max quantized value (unsigned). 15 for 4-bit. */
    public int maxQuantValue() {
        return (1 << bits) - 1;
    }

    /** Memory compression ratio vs FP32 */
    public double compressionRatio() {
        return 32.0 / bits;
    }

    /** Whether this config is using the GEMM kernel layout */
    public boolean isGemmFormat() {
        return kernelFormat == KernelFormat.GEMM;
    }

    /** Whether this config is using the GEMV kernel layout */
    public boolean isGemvFormat() {
        return kernelFormat == KernelFormat.GEMV;
    }

    /** Whether this config is using the Marlin kernel layout */
    public boolean isMarlinFormat() {
        return kernelFormat == KernelFormat.MARLIN;
    }

    /** Returns true if this is a valid AWQ bit-width */
    public boolean isValidBits() {
        return bits == 4; // AWQ almost exclusively uses 4-bit
    }

    /** Returns true if the given bit-width is valid for AWQ */
    public static boolean isValidBits(int bits) {
        return bits == 4;
    }

    @Override
    public String toString() {
        return ("AWQConfig{bits=%d, groupSize=%d, format=%s, hasZeros=%b, exllamaV2=%b, dequantDtype='%s', activationAware=%b}")
                .formatted(bits, groupSize, kernelFormat, hasZeros, exllamaV2, dequantDtype, activationAware);
    }

    // ── Builder Pattern ───────────────────────────────────────────────────────

    /**
     * Builder for constructing AWQConfig instances with custom settings.
     */
    public static class Builder {
        private int bits = 4;
        private int groupSize = 128;
        private KernelFormat kernelFormat = KernelFormat.GEMM;
        private boolean hasZeros = true;
        private String dequantDtype = "float32";
        private boolean exllamaV2 = false;
        private int numSamples = 128;
        private int seqLen = 2048;
        private boolean activationAware = true;
        private String calibrationDataPath = null;

        public Builder bits(int bits) {
            this.bits = bits;
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder kernelFormat(KernelFormat kernelFormat) {
            this.kernelFormat = kernelFormat;
            return this;
        }

        public Builder hasZeros(boolean hasZeros) {
            this.hasZeros = hasZeros;
            return this;
        }

        public Builder dequantDtype(String dequantDtype) {
            this.dequantDtype = dequantDtype;
            return this;
        }

        public Builder exllamaV2(boolean exllamaV2) {
            this.exllamaV2 = exllamaV2;
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

        public Builder activationAware(boolean activationAware) {
            this.activationAware = activationAware;
            return this;
        }

        public Builder calibrationDataPath(String calibrationDataPath) {
            this.calibrationDataPath = calibrationDataPath;
            return this;
        }

        public AWQConfig build() {
            if (!isValidBits(bits)) {
                throw new IllegalArgumentException("Invalid bits: " + bits + ". AWQ typically uses 4-bit.");
            }
            return new AWQConfig(bits, groupSize, kernelFormat, hasZeros, dequantDtype,
                    exllamaV2, numSamples, seqLen, activationAware, calibrationDataPath);
        }
    }
}
