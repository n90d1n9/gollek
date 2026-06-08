/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardGemma4Bf16LinearOptions(
        Boolean enableMetalBf16Linear,
        boolean disableMetalBf16Linear,
        Boolean enableBf16ToF16Linear,
        boolean disableBf16ToF16Linear,
        Boolean enableBf16ToF16FfnLinear,
        boolean disableBf16ToF16FfnLinear,
        Boolean enableBf16ToF16LogitsLinear,
        boolean disableBf16ToF16LogitsLinear) {

    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_linear";
    private static final String DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_ffn_linear";
    private static final String DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_ffn_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_logits_linear";
    private static final String DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_logits_linear";

    static DirectForwardGemma4Bf16LinearOptions fromSystemProperties() {
        return new DirectForwardGemma4Bf16LinearOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY));
    }

    static DirectForwardGemma4Bf16LinearOptions defaults() {
        return new DirectForwardGemma4Bf16LinearOptions(null, false, null, false, null, false, null, false);
    }

    DirectForwardGemma4Bf16LinearOptions withMetalBf16(Boolean enable, boolean disable) {
        return new DirectForwardGemma4Bf16LinearOptions(
                enable,
                disable,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardGemma4Bf16LinearOptions withBf16ToF16(Boolean enable, boolean disable) {
        return new DirectForwardGemma4Bf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enable,
                disable,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardGemma4Bf16LinearOptions withBf16ToF16Ffn(Boolean enable, boolean disable) {
        return new DirectForwardGemma4Bf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enable,
                disable,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardGemma4Bf16LinearOptions withBf16ToF16Logits(Boolean enable, boolean disable) {
        return new DirectForwardGemma4Bf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enable,
                disable);
    }
}
