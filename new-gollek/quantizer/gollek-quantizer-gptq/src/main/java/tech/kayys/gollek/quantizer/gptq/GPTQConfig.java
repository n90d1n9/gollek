package tech.kayys.gollek.quantizer.gptq;

/**
 * Configuration for GPTQ (Generative Pre-Trained Transformer Quantization).
 *
 * GPTQ uses weight-only quantization, storing weights in INT4 or INT8 format
 * packed into INT32 tensors. Each group of {@code groupSize} original weights
 * shares a single scale and zero-point.
 *
 * Standard GPTQ tensor naming convention:
 * layer.qweight - packed INT32 quantized weights
 * layer.qzeros - packed INT32 zero points
 * layer.scales - FP16 scale factors
 * layer.g_idx - INT32 group indices (optional, for act-order)
 * layer.bias - FP16/FP32 bias (optional)
 */
public record GPTQConfig(
        /** Bit-width for quantization (2, 3, 4, or 8) */
        int bits,

        /** Number of elements per quantization group (typically 128) */
        int groupSize,

        /**
         * Whether activation-order (act-order) was used during quantization.
         * If true, g_idx tensors are present and required for correct dequantization.
         */
        boolean actOrder,

        /** Whether to use symmetric quantization (zero = 0) */
        boolean symmetric,

        /** Whether the model uses exllama v2 format */
        boolean exllamaV2,

        /** Target dequantized dtype: "float32" or "float16" */
        String dequantDtype,

        /** Whether to use per-channel quantization (default: true) */
        boolean perChannel,

        /** Dampening percentage for numerical stability (default: 0.01 = 1%) */
        double dampPercent,

        /** Number of calibration samples for quantization (default: 128) */
        int numSamples,

        /** Sequence length for calibration (default: 2048) */
        int seqLen,

        /** Whether to quantize embeddings (default: false) */
        boolean quantizeEmbeddings,

        /** Calibration dataset path (optional) */
        String calibrationDataPath) {

    // ── Preset configurations ─────────────────────────────────────────────────

    /** Standard GPTQ 4-bit, group=128, asymmetric */
    public static GPTQConfig gptq4bit() {
        return new GPTQConfig(4, 128, false, false, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 4-bit with activation ordering (AutoGPTQ default) */
    public static GPTQConfig gptq4bitActOrder() {
        return new GPTQConfig(4, 128, true, false, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 8-bit, group=128 */
    public static GPTQConfig gptq8bit() {
        return new GPTQConfig(8, 128, false, false, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 3-bit, group=128 */
    public static GPTQConfig gptq3bit() {
        return new GPTQConfig(3, 128, false, false, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 2-bit, group=128 */
    public static GPTQConfig gptq2bit() {
        return new GPTQConfig(2, 128, false, false, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 4-bit with FP16 output for memory efficiency */
    public static GPTQConfig gptq4bitFP16() {
        return new GPTQConfig(4, 128, false, false, false, "float16",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 4-bit with symmetric quantization */
    public static GPTQConfig gptq4bitSymmetric() {
        return new GPTQConfig(4, 128, false, true, false, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** GPTQ 4-bit optimized for exllama v2 */
    public static GPTQConfig gptq4bitExllamaV2() {
        return new GPTQConfig(4, 128, false, false, true, "float32",
                true, 0.01, 128, 2048, false, null);
    }

    /** Custom configuration builder */
    public static Builder builder() {
        return new Builder();
    }

    // ── Derived properties ────────────────────────────────────────────────────

    /**
     * Number of INT4 values packed into each INT32 element.
     * For 4-bit: 8, for 8-bit: 4, for 3-bit: 10 (with padding), etc.
     */
    public int elementsPerInt32() {
        return 32 / bits;
    }

    /**
     * Bitmask for a single quantized value.
     * For 4-bit: 0xF, for 8-bit: 0xFF, for 3-bit: 0x7, etc.
     */
    public int quantMask() {
        return (1 << bits) - 1;
    }

    /**
     * The zero-point offset used in asymmetric dequantization.
     * For INT4 asymmetric: 2^(bits-1) = 8
     */
    public int zeroPointOffset() {
        return symmetric ? 0 : (1 << (bits - 1));
    }

    /**
     * Maximum quantized value (unsigned).
     * For 4-bit: 15, for 8-bit: 255
     */
    public int maxQuantValue() {
        return (1 << bits) - 1;
    }

    /**
     * Compression ratio vs float32 baseline.
     * e.g., 4-bit = 8x compression vs float32
     */
    public double compressionRatio() {
        return 32.0 / bits;
    }

    /** Returns true if the provided value is a valid GPTQ bit-width */
    public static boolean isValidBits(int b) {
        return b == 2 || b == 3 || b == 4 || b == 8;
    }

    /** Returns true if this config has a valid GPTQ bit-width */
    public boolean isValidBits() {
        return isValidBits(this.bits);
    }

    @Override
    public String toString() {
        return "GPTQConfig{bits=%d, groupSize=%d, actOrder=%b, symmetric=%b, exllamaV2=%b, dequantDtype='%s', perChannel=%b, dampPercent=%.2f, numSamples=%d, seqLen=%d, quantizeEmbeddings=%b}"
                .formatted(bits, groupSize, actOrder, symmetric, exllamaV2, dequantDtype, perChannel, dampPercent, numSamples, seqLen, quantizeEmbeddings);
    }

    // ── Builder Pattern ───────────────────────────────────────────────────────

    /**
     * Builder for constructing GPTQConfig instances with custom settings.
     */
    public static class Builder {
        private int bits = 4;
        private int groupSize = 128;
        private boolean actOrder = false;
        private boolean symmetric = false;
        private boolean exllamaV2 = false;
        private String dequantDtype = "float32";
        private boolean perChannel = true;
        private double dampPercent = 0.01;
        private int numSamples = 128;
        private int seqLen = 2048;
        private boolean quantizeEmbeddings = false;
        private String calibrationDataPath = null;

        public Builder bits(int bits) {
            this.bits = bits;
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder actOrder(boolean actOrder) {
            this.actOrder = actOrder;
            return this;
        }

        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder exllamaV2(boolean exllamaV2) {
            this.exllamaV2 = exllamaV2;
            return this;
        }

        public Builder dequantDtype(String dequantDtype) {
            this.dequantDtype = dequantDtype;
            return this;
        }

        public Builder perChannel(boolean perChannel) {
            this.perChannel = perChannel;
            return this;
        }

        public Builder dampPercent(double dampPercent) {
            this.dampPercent = dampPercent;
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

        public Builder quantizeEmbeddings(boolean quantizeEmbeddings) {
            this.quantizeEmbeddings = quantizeEmbeddings;
            return this;
        }

        public Builder calibrationDataPath(String calibrationDataPath) {
            this.calibrationDataPath = calibrationDataPath;
            return this;
        }

        public GPTQConfig build() {
            if (!isValidBits(bits)) {
                throw new IllegalArgumentException("Invalid bits: " + bits + ". Must be 2, 3, 4, or 8.");
            }
            return new GPTQConfig(bits, groupSize, actOrder, symmetric, exllamaV2, dequantDtype,
                    perChannel, dampPercent, numSamples, seqLen, quantizeEmbeddings, calibrationDataPath);
        }
    }
}
