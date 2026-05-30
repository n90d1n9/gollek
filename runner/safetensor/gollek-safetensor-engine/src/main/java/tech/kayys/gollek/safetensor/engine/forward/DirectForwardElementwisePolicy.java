/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

final class DirectForwardElementwisePolicy {
    /** Gemma-4 Metal elementwise is opt-in until RMSNorm/add parity is proven. */
    private static final String ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_elementwise";
    private static final String DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_elementwise";
    private static final boolean DISABLE_METAL_GEMMA4_ELEMENTWISE_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY);
    private static final boolean ENABLE_GEMMA4_METAL_ELEMENTWISE_ENABLED =
            Boolean.getBoolean(ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY);
    private static final String METAL_ELEMENTWISE_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_elementwise_min_seq";
    private static final int METAL_ELEMENTWISE_MIN_SEQ_VALUE =
            Integer.getInteger(METAL_ELEMENTWISE_MIN_SEQ_PROPERTY, -1);
    private static final String DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY =
            "gollek.safetensor.disable_gemma4_per_layer_input";
    private static final boolean DISABLE_GEMMA4_PER_LAYER_INPUT_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY);
    private static final String DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_layer_scalar";
    private static final boolean DISABLE_GEMMA4_LAYER_SCALAR_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY);
    private static final String ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY =
            "gollek.safetensor.enable_metal_layer_scalar_decode";
    private static final boolean ENABLE_METAL_LAYER_SCALAR_DECODE_ENABLED =
            Boolean.getBoolean(ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY);
    private static final String METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_layer_scalar_min_seq";
    private static final int DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ = 64;
    private static final int METAL_LAYER_SCALAR_MIN_SEQ =
            Integer.getInteger(METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY, DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ);
    private static final String ENABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_post_ffn_norm";
    private static final String ENABLE_METAL_POST_FFN_NORM_VALUE =
            System.getProperty(ENABLE_METAL_POST_FFN_NORM_PROPERTY);
    private static final Boolean ENABLE_METAL_POST_FFN_NORM_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_POST_FFN_NORM_VALUE);
    private static final String DISABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_post_ffn_norm";
    private static final boolean DISABLE_METAL_POST_FFN_NORM_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_POST_FFN_NORM_PROPERTY);

    private DirectForwardElementwisePolicy() {
    }

    static boolean canUseMetalElementwise(
            ModelConfigTraits traits,
            int seqLen,
            boolean forceCpuForward,
            boolean canUseMetal,
            boolean nativeElementwiseKernelsAvailable,
            boolean nativeElementwiseFallbackAvailable) {
        if (forceCpuForward) {
            return false;
        }
        boolean gemma4 = traits.gemma4Text();
        if (gemma4) {
            if (DISABLE_METAL_GEMMA4_ELEMENTWISE_ENABLED) {
                return false;
            }
            if (!canUseMetal || !nativeElementwiseKernelsAvailable) {
                return false;
            }
        } else if (!canUseMetal && !nativeElementwiseFallbackAvailable) {
            return false;
        }
        int defaultMinSeq = gemma4 ? 1 : 16;
        int minSeq = METAL_ELEMENTWISE_MIN_SEQ_VALUE >= 0 ? METAL_ELEMENTWISE_MIN_SEQ_VALUE : defaultMinSeq;
        if (seqLen < minSeq) {
            return false;
        }
        if (!gemma4) {
            return true;
        }
        return ENABLE_GEMMA4_METAL_ELEMENTWISE_ENABLED
                || nativeElementwiseKernelsAvailable;
    }

    static boolean canUseMetalLayerScalarScale(
            boolean useMetalElementwise,
            int seqLen,
            boolean metalBindingAvailable,
            boolean metalNativeScaleKernelAvailable) {
        return useMetalElementwise
                && metalBindingAvailable
                && metalNativeScaleKernelAvailable
                && (seqLen >= METAL_LAYER_SCALAR_MIN_SEQ
                || (seqLen == 1 && ENABLE_METAL_LAYER_SCALAR_DECODE_ENABLED));
    }

    static boolean shouldUseMetalPostFfnNorm(ModelConfigTraits traits) {
        if (DISABLE_METAL_POST_FFN_NORM_ENABLED) {
            return false;
        }
        if (ENABLE_METAL_POST_FFN_NORM_EXPLICIT != null) {
            return ENABLE_METAL_POST_FFN_NORM_EXPLICIT;
        }
        return traits.gemma4Text() || traits.gemma4StylePerLayerInputs();
    }

    static boolean shouldBuildPerLayerInputs(ModelConfigTraits traits, int hiddenSizePerLayerInput) {
        if (hiddenSizePerLayerInput <= 0) {
            return false;
        }
        return !isGemma4PerLayerInputDisabled(traits);
    }

    static boolean shouldApplyLayerScalar(ModelConfigTraits traits) {
        if (!traits.gemma4Text()) {
            return true;
        }
        return !DISABLE_GEMMA4_LAYER_SCALAR_ENABLED;
    }

    static boolean isGemma4FfnPolicyTarget(ModelConfigTraits traits) {
        return traits.gemma4Text() || traits.gemma4StylePerLayerInputs();
    }

    private static boolean isGemma4PerLayerInputDisabled(ModelConfigTraits traits) {
        return traits.gemma4Text() && DISABLE_GEMMA4_PER_LAYER_INPUT_ENABLED;
    }
}
