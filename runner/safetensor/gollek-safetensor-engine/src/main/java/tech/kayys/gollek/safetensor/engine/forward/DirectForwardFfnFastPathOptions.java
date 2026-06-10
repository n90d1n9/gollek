/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardFfnFastPathOptions(
        Boolean gemma4FusedHalfFfnExplicit,
        boolean disableGemma4FusedHalfFfn,
        boolean disableMetalFusedFfn,
        boolean enableQwenMetalFusedFfn,
        boolean enableMetalGegluFusedFfn,
        Boolean enableMetalFusedFfnPrefill,
        int gemma4FusedFfnPrefillMinRows,
        Boolean enableMetalMatvecFfnPrefillRows,
        Boolean preferMetalMatvecFfnPrefillRows,
        int metalMatvecFfnPrefillMaxRows,
        Boolean enableMetalGegluMatvecFfn,
        Boolean enableMetalSwigluMatvecFfn,
        Boolean enableMetalGateUpMatvecFfn,
        boolean disableMetalMatvecFfn,
        boolean validateMetalMatvecFfn) {

    static final String ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.allow_gemma4_fused_half_ffn";
    private static final String DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.disable_gemma4_fused_half_ffn";
    private static final String DISABLE_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_fused_ffn";
    private static final String ENABLE_QWEN_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_qwen_metal_fused_ffn";
    private static final String ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_fused_ffn";
    static final String ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY =
            "gollek.safetensor.enable_metal_fused_ffn_prefill";
    static final String GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY =
            "gollek.safetensor.gemma4_fused_ffn_prefill_min_rows";
    private static final int DEFAULT_GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS = 2;
    static final String ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY =
            "gollek.safetensor.enable_metal_matvec_ffn_prefill_rows";
    static final String PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY =
            "gollek.safetensor.prefer_metal_matvec_ffn_prefill_rows";
    static final String METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY =
            "gollek.safetensor.metal_matvec_ffn_prefill_max_rows";
    private static final int DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS = 16;
    static final String ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_matvec_ffn";
    static final String ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_swiglu_matvec_ffn";
    static final String ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_gate_up_matvec_ffn";
    private static final String DISABLE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_matvec_ffn";
    private static final String VALIDATE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.validate_metal_matvec_ffn";

    static DirectForwardFfnFastPathOptions fromSystemProperties() {
        return new DirectForwardFfnFastPathOptions(
                parseOptionalBoolean(System.getProperty(ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY)),
                Boolean.getBoolean(DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY),
                Boolean.getBoolean(DISABLE_METAL_FUSED_FFN_PROPERTY),
                Boolean.TRUE.equals(parseOptionalBoolean(System.getProperty(ENABLE_QWEN_METAL_FUSED_FFN_PROPERTY))),
                Boolean.getBoolean(ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY)),
                Integer.getInteger(GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY,
                        DEFAULT_GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY)),
                parseOptionalBoolean(System.getProperty(PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY)),
                Integer.getInteger(METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY,
                        DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY)),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY)),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_MATVEC_FFN_PROPERTY),
                Boolean.getBoolean(VALIDATE_METAL_MATVEC_FFN_PROPERTY));
    }

    static DirectForwardFfnFastPathOptions defaults() {
        return new DirectForwardFfnFastPathOptions(
                null, false, false, false, false, null,
                DEFAULT_GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS, null, null,
                DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS, null, null, null, false, false);
    }

    DirectForwardFfnFastPathOptions withGemma4FusedHalfFfn(Boolean allow, boolean disable) {
        return new DirectForwardFfnFastPathOptions(
                allow,
                disable,
                disableMetalFusedFfn,
                enableQwenMetalFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                gemma4FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalFusedFfn(
            boolean disabled, boolean qwenEnabled, boolean gegluEnabled) {
        return new DirectForwardFfnFastPathOptions(
                gemma4FusedHalfFfnExplicit,
                disableGemma4FusedHalfFfn,
                disabled,
                qwenEnabled,
                gegluEnabled,
                enableMetalFusedFfnPrefill,
                gemma4FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalFusedFfnPrefill(Boolean enabled, int minRows) {
        return new DirectForwardFfnFastPathOptions(
                gemma4FusedHalfFfnExplicit,
                disableGemma4FusedHalfFfn,
                disableMetalFusedFfn,
                enableQwenMetalFusedFfn,
                enableMetalGegluFusedFfn,
                enabled,
                minRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfnPrefillRows(Boolean enabled, int maxRows) {
        return withMetalMatvecFfnPrefillRows(enabled, maxRows, preferMetalMatvecFfnPrefillRows);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfnPrefillRows(
            Boolean enabled, int maxRows, Boolean preferRows) {
        return new DirectForwardFfnFastPathOptions(
                gemma4FusedHalfFfnExplicit,
                disableGemma4FusedHalfFfn,
                disableMetalFusedFfn,
                enableQwenMetalFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                gemma4FusedFfnPrefillMinRows,
                enabled,
                preferRows,
                maxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfn(
            Boolean gegluEnabled, Boolean swigluEnabled, Boolean gateUpEnabled, boolean disabled) {
        return new DirectForwardFfnFastPathOptions(
                gemma4FusedHalfFfnExplicit,
                disableGemma4FusedHalfFfn,
                disableMetalFusedFfn,
                enableQwenMetalFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                gemma4FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                gegluEnabled,
                swigluEnabled,
                gateUpEnabled,
                disabled,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withValidateMetalMatvecFfn(boolean validate) {
        return new DirectForwardFfnFastPathOptions(
                gemma4FusedHalfFfnExplicit,
                disableGemma4FusedHalfFfn,
                disableMetalFusedFfn,
                enableQwenMetalFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                gemma4FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validate);
    }
}
