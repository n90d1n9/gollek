/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

final class FlashAttentionRuntimeOptions {
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long METAL_F16_WEIGHT_CACHE_MAX_BYTES = Long.getLong(
            METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES);

    private static final String ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple_matvec";
    private static final String METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_mixed_half_linear_triple_matvec_max_output";
    private static final int DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT = 4096;

    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final String AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_attention_half_matvec";
    private static final boolean AUTO_METAL_HALF_MATVEC_ENABLED =
            resolveAutoMetalHalfMatvecEnabled();
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 8192;
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec_all";
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 262144;

    private static final boolean EXPERIMENTAL_METAL_LINEAR_ENABLED =
            resolveExperimentalMetalLinearEnabled();
    private static final boolean METAL_MIXED_HALF_LINEAR_PAIR_ENABLED =
            resolveMetalMixedHalfLinearPairEnabled();

    private FlashAttentionRuntimeOptions() {
    }

    static boolean forceCpuForwardEnabled() {
        return Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY);
    }

    static long metalF16WeightCacheMaxBytes() {
        return METAL_F16_WEIGHT_CACHE_MAX_BYTES;
    }

    static boolean experimentalMetalLinearEnabled() {
        return EXPERIMENTAL_METAL_LINEAR_ENABLED;
    }

    static boolean allowMetalBf16Linear() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    static boolean disableGemma4MetalBf16Linear() {
        return Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    }

    static String enableGemma4MetalBf16LinearValue() {
        return System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    }

    static boolean metalMixedHalfLinearPairEnabled() {
        return METAL_MIXED_HALF_LINEAR_PAIR_ENABLED;
    }

    static boolean disableMetalMixedHalfLinearTriple() {
        return Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY);
    }

    static boolean shouldUseMetalHalfMatvec(FlashAttentionModelPolicy modelPolicy, int outputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY);
        int maxOutput = Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit) && outputDim <= maxOutput;
        }
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            return outputDim <= maxOutput;
        }
        if (AUTO_METAL_HALF_MATVEC_ENABLED) {
            return outputDim <= maxOutput
                    && modelPolicy.metalHalfMatvecAutoCandidate();
        }
        return !autoMetalAttentionHalfMatvecExplicitlyDisabled()
                && outputDim <= maxOutput
                && modelPolicy.compactAttentionMatvecCandidate();
    }

    static boolean shouldUseMetalTransposedHalfMatvec(FlashAttentionModelPolicy modelPolicy, int outputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY);
        boolean enabled = explicit != null && !explicit.isBlank() && Boolean.parseBoolean(explicit);
        boolean allowAllExperimentalProjectors =
                Boolean.getBoolean(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY);
        if (!modelPolicy.preferNativeMetalBf16Linear() && !allowAllExperimentalProjectors) {
            return false;
        }
        int maxOutput = Integer.getInteger(METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT);
        return enabled && outputDim <= maxOutput;
    }

    static boolean shouldUseMetalHalfTripleMatvec(int firstOutputDim, int secondOutputDim, int thirdOutputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY)) {
            return false;
        }
        int maxOutput = Integer.getInteger(METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT);
        return firstOutputDim > 0
                && secondOutputDim > 0
                && thirdOutputDim > 0
                && firstOutputDim + secondOutputDim + thirdOutputDim <= maxOutput;
    }

    private static boolean resolveExperimentalMetalLinearEnabled() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    private static boolean resolveMetalMixedHalfLinearPairEnabled() {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    private static boolean resolveAutoMetalHalfMatvecEnabled() {
        String explicit = System.getProperty(AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        explicit = System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return false;
    }

    private static boolean autoMetalAttentionHalfMatvecExplicitlyDisabled() {
        String explicit = System.getProperty(AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return !Boolean.parseBoolean(explicit);
        }
        explicit = System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY);
        return explicit != null && !explicit.isBlank() && !Boolean.parseBoolean(explicit);
    }
}
