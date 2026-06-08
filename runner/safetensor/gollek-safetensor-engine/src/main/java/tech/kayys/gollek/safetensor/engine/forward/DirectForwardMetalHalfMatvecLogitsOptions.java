/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardMetalHalfMatvecLogitsOptions(
        Boolean enableMetalLogitsMpsMatvec,
        boolean disableMetalLogitsMpsMatvec,
        int metalLogitsMpsMatvecMinOutput,
        int metalLogitsMpsMatvecMaxInput,
        int gemma4LogitsMetalHalfMatvecMaxOutput,
        Boolean enableGemma3LogitsMetalHalfMatvec,
        boolean disableGemma3LogitsMetalHalfMatvec,
        int gemma3LogitsMetalHalfMatvecMaxOutput,
        int qwenLogitsMetalHalfMatvecMaxOutput) {

    private static final String ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_logits_mps_matvec";
    private static final String DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_logits_mps_matvec";
    private static final String METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_min_output";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT = 65536;
    private static final String METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_max_input";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT = 4096;
    private static final String GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.gemma4_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final String ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_gemma3_logits_metal_half_matvec";
    private static final String DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_gemma3_logits_metal_half_matvec";
    private static final String GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.gemma3_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final String QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.qwen_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;

    static DirectForwardMetalHalfMatvecLogitsOptions fromSystemProperties() {
        return new DirectForwardMetalHalfMatvecLogitsOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY),
                Integer.getInteger(METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY,
                        DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT),
                Integer.getInteger(METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY,
                        DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT),
                Integer.getInteger(GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT),
                parseOptionalBoolean(System.getProperty(ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY),
                Integer.getInteger(GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT),
                Integer.getInteger(QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT));
    }

    static DirectForwardMetalHalfMatvecLogitsOptions defaults() {
        return new DirectForwardMetalHalfMatvecLogitsOptions(
                null,
                false,
                DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT,
                DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT,
                DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT,
                null,
                false,
                DEFAULT_GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT,
                DEFAULT_QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT);
    }

    DirectForwardMetalHalfMatvecLogitsOptions withLogitsMpsMatvec(Boolean enable, boolean disable,
            int minOutput, int maxInput) {
        return new DirectForwardMetalHalfMatvecLogitsOptions(
                enable,
                disable,
                minOutput,
                maxInput,
                gemma4LogitsMetalHalfMatvecMaxOutput,
                enableGemma3LogitsMetalHalfMatvec,
                disableGemma3LogitsMetalHalfMatvec,
                gemma3LogitsMetalHalfMatvecMaxOutput,
                qwenLogitsMetalHalfMatvecMaxOutput);
    }

    DirectForwardMetalHalfMatvecLogitsOptions withLogitsMaxOutputs(
            int gemma4MaxOutput, int gemma3MaxOutput, int qwenMaxOutput) {
        return new DirectForwardMetalHalfMatvecLogitsOptions(
                enableMetalLogitsMpsMatvec,
                disableMetalLogitsMpsMatvec,
                metalLogitsMpsMatvecMinOutput,
                metalLogitsMpsMatvecMaxInput,
                gemma4MaxOutput,
                enableGemma3LogitsMetalHalfMatvec,
                disableGemma3LogitsMetalHalfMatvec,
                gemma3MaxOutput,
                qwenMaxOutput);
    }

    DirectForwardMetalHalfMatvecLogitsOptions withGemma3Logits(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecLogitsOptions(
                enableMetalLogitsMpsMatvec,
                disableMetalLogitsMpsMatvec,
                metalLogitsMpsMatvecMinOutput,
                metalLogitsMpsMatvecMaxInput,
                gemma4LogitsMetalHalfMatvecMaxOutput,
                enable,
                disable,
                gemma3LogitsMetalHalfMatvecMaxOutput,
                qwenLogitsMetalHalfMatvecMaxOutput);
    }

}
