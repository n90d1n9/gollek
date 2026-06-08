/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardElementwiseOptions(
        boolean enableGemma4MetalElementwise,
        boolean disableMetalGemma4Elementwise,
        int metalElementwiseMinSeq,
        boolean disableGemma4PerLayerInput,
        boolean disableGemma4LayerScalar,
        boolean enableMetalLayerScalarDecode,
        int metalLayerScalarMinSeq,
        Boolean enableMetalPostFfnNorm,
        boolean disableMetalPostFfnNorm) {

    private static final String ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_elementwise";
    private static final String DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_elementwise";
    private static final String METAL_ELEMENTWISE_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_elementwise_min_seq";
    private static final String DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY =
            "gollek.safetensor.disable_gemma4_per_layer_input";
    private static final String DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_layer_scalar";
    private static final String ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY =
            "gollek.safetensor.enable_metal_layer_scalar_decode";
    private static final String METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_layer_scalar_min_seq";
    private static final int DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ = 64;
    private static final String ENABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_post_ffn_norm";
    private static final String DISABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_post_ffn_norm";

    static DirectForwardElementwiseOptions fromSystemProperties() {
        return new DirectForwardElementwiseOptions(
                Boolean.getBoolean(ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY),
                Boolean.getBoolean(DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY),
                Integer.getInteger(METAL_ELEMENTWISE_MIN_SEQ_PROPERTY, -1),
                Boolean.getBoolean(DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY),
                Boolean.getBoolean(DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY),
                Boolean.getBoolean(ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY),
                Integer.getInteger(METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY, DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_POST_FFN_NORM_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_POST_FFN_NORM_PROPERTY));
    }

    static DirectForwardElementwiseOptions defaults() {
        return new DirectForwardElementwiseOptions(
                false,
                false,
                -1,
                false,
                false,
                false,
                DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ,
                null,
                false);
    }

    DirectForwardElementwiseOptions withMetalElementwise(
            boolean enableGemma4, boolean disableGemma4, int minSeq) {
        return new DirectForwardElementwiseOptions(
                enableGemma4,
                disableGemma4,
                minSeq,
                disableGemma4PerLayerInput,
                disableGemma4LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withPerLayerInputDisabled(boolean disabled) {
        return new DirectForwardElementwiseOptions(
                enableGemma4MetalElementwise,
                disableMetalGemma4Elementwise,
                metalElementwiseMinSeq,
                disabled,
                disableGemma4LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withLayerScalar(boolean disableGemma4, boolean enableDecode, int minSeq) {
        return new DirectForwardElementwiseOptions(
                enableGemma4MetalElementwise,
                disableMetalGemma4Elementwise,
                metalElementwiseMinSeq,
                disableGemma4PerLayerInput,
                disableGemma4,
                enableDecode,
                minSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withPostFfnNorm(Boolean enable, boolean disable) {
        return new DirectForwardElementwiseOptions(
                enableGemma4MetalElementwise,
                disableMetalGemma4Elementwise,
                metalElementwiseMinSeq,
                disableGemma4PerLayerInput,
                disableGemma4LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enable,
                disable);
    }
}
