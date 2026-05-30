package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.envInt;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.envTruthy;

final class DirectForwardNativeBf16MatvecPolicy {
    private static final boolean ENV_METAL_MATVEC_THREADS_128 =
            "128".equals(System.getenv("GOLLEK_METAL_MATVEC_THREADS"));
    private static final boolean ENV_DISABLE_BF16_MATVEC_X8 =
            envTruthy("GOLLEK_METAL_DISABLE_BF16_MATVEC_X8");
    private static final boolean ENV_DISABLE_BF16_MATVEC_X4 =
            envTruthy("GOLLEK_METAL_DISABLE_BF16_MATVEC_X4");
    private static final boolean ENV_DISABLE_BF16_PAIR_X8 =
            envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_X8");
    private static final boolean ENV_DISABLE_BF16_PAIR_X4 =
            envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_X4");
    private static final boolean ENV_ENABLE_BF16_PAIR_X4 =
            envTruthy("GOLLEK_METAL_ENABLE_BF16_PAIR_X4");
    private static final boolean ENV_DISABLE_BF16_PAIR_SIMD =
            envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_SIMD");
    private static final boolean ENV_ENABLE_BF16_PAIR_SIMD =
            envTruthy("GOLLEK_METAL_ENABLE_BF16_PAIR_SIMD");
    private static final boolean ENV_DISABLE_SIMDGROUP_REDUCTION =
            envTruthy("GOLLEK_METAL_DISABLE_SIMDGROUP_REDUCTION");
    private static final boolean ENV_ENABLE_SIMDGROUP_REDUCTION =
            envTruthy("GOLLEK_METAL_ENABLE_SIMDGROUP_REDUCTION");
    private static final int ENV_BF16_MATVEC_X8_MIN_INNER =
            envInt("GOLLEK_METAL_BF16_MATVEC_X8_MIN_INNER", 512);
    private static final int ENV_BF16_MATVEC_X8_MIN_OUTPUT =
            envInt("GOLLEK_METAL_BF16_MATVEC_X8_MIN_OUTPUT", 1024);
    private static final int ENV_BF16_MATVEC_X8_MAX_OUTPUT =
            envInt("GOLLEK_METAL_BF16_MATVEC_X8_MAX_OUTPUT", 65536);
    private static final int ENV_BF16_MATVEC_X4_MIN_INNER =
            envInt("GOLLEK_METAL_BF16_MATVEC_X4_MIN_INNER", 512);
    private static final int ENV_BF16_MATVEC_X4_MIN_OUTPUT =
            envInt("GOLLEK_METAL_BF16_MATVEC_X4_MIN_OUTPUT", 1024);
    private static final int ENV_BF16_MATVEC_X4_MAX_OUTPUT =
            envInt("GOLLEK_METAL_BF16_MATVEC_X4_MAX_OUTPUT", 8192);
    private static final int ENV_BF16_PAIR_SIMD_MIN_INNER =
            envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MIN_INNER", 1024);
    private static final int ENV_BF16_PAIR_SIMD_MIN_OUTPUT =
            envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MIN_OUTPUT", 4096);
    private static final int ENV_BF16_PAIR_SIMD_MAX_OUTPUT =
            envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MAX_OUTPUT", 0);

    private DirectForwardNativeBf16MatvecPolicy() {
    }

    static String describeNativeBf16MatvecPath(int inputDim, int outputDim) {
        if (shouldUseNativeBf16MatvecX8(inputDim, outputDim)) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_matvec_x8_simd"
                    : "bf16_matvec_x8";
        }
        if (shouldUseNativeBf16MatvecX4(inputDim, outputDim)) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_matvec_x4_simd"
                    : "bf16_matvec_x4";
        }
        return "bf16_matvec";
    }

    static String describeNativeBf16PairMatvecPath(int inputDim, int outputDim) {
        boolean pairX8 = !ENV_DISABLE_BF16_PAIR_X8 && shouldUseNativeBf16MatvecX8(inputDim, outputDim);
        if (pairX8) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_pair_matvec_x8_simd"
                    : "bf16_pair_matvec_x8";
        }
        boolean pairX4 = !ENV_DISABLE_BF16_PAIR_X4
                && (ENV_ENABLE_BF16_PAIR_X4 || shouldUseNativeBf16MatvecX4(inputDim, outputDim));
        if (pairX4) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_pair_matvec_x4_simd"
                    : "bf16_pair_matvec_x4";
        }
        if (shouldUseNativeBf16PairSimdReduction(inputDim, outputDim)) {
            return "bf16_pair_matvec_simd";
        }
        return "bf16_pair_matvec";
    }

    private static boolean shouldUseNativeBf16MatvecX8(int inputDim, int outputDim) {
        if (ENV_DISABLE_BF16_MATVEC_X8 || ENV_METAL_MATVEC_THREADS_128) {
            return false;
        }
        return (ENV_BF16_MATVEC_X8_MIN_INNER <= 0 || inputDim >= ENV_BF16_MATVEC_X8_MIN_INNER)
                && (ENV_BF16_MATVEC_X8_MIN_OUTPUT <= 0 || outputDim >= ENV_BF16_MATVEC_X8_MIN_OUTPUT)
                && (ENV_BF16_MATVEC_X8_MAX_OUTPUT <= 0 || outputDim <= ENV_BF16_MATVEC_X8_MAX_OUTPUT);
    }

    private static boolean shouldUseNativeBf16MatvecX4(int inputDim, int outputDim) {
        if (ENV_DISABLE_BF16_MATVEC_X4 || ENV_METAL_MATVEC_THREADS_128) {
            return false;
        }
        return (ENV_BF16_MATVEC_X4_MIN_INNER <= 0 || inputDim >= ENV_BF16_MATVEC_X4_MIN_INNER)
                && (ENV_BF16_MATVEC_X4_MIN_OUTPUT <= 0 || outputDim >= ENV_BF16_MATVEC_X4_MIN_OUTPUT)
                && (ENV_BF16_MATVEC_X4_MAX_OUTPUT <= 0 || outputDim <= ENV_BF16_MATVEC_X4_MAX_OUTPUT);
    }

    private static boolean shouldUseNativeBf16PairSimdReduction(int inputDim, int outputDim) {
        if (ENV_DISABLE_BF16_PAIR_SIMD) {
            return false;
        }
        if (ENV_ENABLE_BF16_PAIR_SIMD) {
            return true;
        }
        if (ENV_METAL_MATVEC_THREADS_128) {
            return false;
        }
        return (ENV_BF16_PAIR_SIMD_MIN_INNER <= 0 || inputDim >= ENV_BF16_PAIR_SIMD_MIN_INNER)
                && (ENV_BF16_PAIR_SIMD_MIN_OUTPUT <= 0 || outputDim >= ENV_BF16_PAIR_SIMD_MIN_OUTPUT)
                && (ENV_BF16_PAIR_SIMD_MAX_OUTPUT <= 0 || outputDim <= ENV_BF16_PAIR_SIMD_MAX_OUTPUT);
    }

    private static boolean shouldUseNativeSimdgroupReduction() {
        return !ENV_DISABLE_SIMDGROUP_REDUCTION && ENV_ENABLE_SIMDGROUP_REDUCTION;
    }
}
