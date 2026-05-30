/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.runtimeOptionalBooleanProperty;

final class DirectForwardFfnFastPathPolicy {
    private static final String ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.allow_gemma4_fused_half_ffn";
    private static final String ALLOW_GEMMA4_FUSED_HALF_FFN_VALUE =
            System.getProperty(ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY);
    private static final Boolean ALLOW_GEMMA4_FUSED_HALF_FFN_EXPLICIT =
            parseOptionalBoolean(ALLOW_GEMMA4_FUSED_HALF_FFN_VALUE);
    private static final String DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.disable_gemma4_fused_half_ffn";
    private static final boolean DISABLE_GEMMA4_FUSED_HALF_FFN_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY);
    private static final String DISABLE_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_fused_ffn";
    private static final boolean DISABLE_METAL_FUSED_FFN_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_FUSED_FFN_PROPERTY);
    private static final String ENABLE_QWEN_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_qwen_metal_fused_ffn";
    private static final String ENABLE_QWEN_METAL_FUSED_FFN_VALUE =
            System.getProperty(ENABLE_QWEN_METAL_FUSED_FFN_PROPERTY);
    private static final boolean ENABLE_QWEN_METAL_FUSED_FFN_ENABLED =
            Boolean.TRUE.equals(parseOptionalBoolean(ENABLE_QWEN_METAL_FUSED_FFN_VALUE));
    private static final String ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_fused_ffn";
    private static final boolean ENABLE_METAL_GEGLU_FUSED_FFN_ENABLED =
            Boolean.getBoolean(ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY);
    private static final String ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY =
            "gollek.safetensor.enable_metal_fused_ffn_prefill";
    private static final String ENABLE_METAL_FUSED_FFN_PREFILL_VALUE =
            System.getProperty(ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY);
    private static final Boolean ENABLE_METAL_FUSED_FFN_PREFILL_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_FUSED_FFN_PREFILL_VALUE);
    private static final String GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY =
            "gollek.safetensor.gemma4_fused_ffn_prefill_min_rows";
    private static final int DEFAULT_GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS = 2;
    private static final int GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS =
            Integer.getInteger(GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY,
                    DEFAULT_GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS);
    private static final String ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_matvec_ffn";
    private static final String ENABLE_METAL_GEGLU_MATVEC_FFN_VALUE =
            System.getProperty(ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY);
    private static final Boolean ENABLE_METAL_GEGLU_MATVEC_FFN_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_GEGLU_MATVEC_FFN_VALUE);
    private static final String ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_swiglu_matvec_ffn";
    private static final String ENABLE_METAL_SWIGLU_MATVEC_FFN_VALUE =
            System.getProperty(ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY);
    private static final Boolean ENABLE_METAL_SWIGLU_MATVEC_FFN_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_SWIGLU_MATVEC_FFN_VALUE);
    private static final String ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_gate_up_matvec_ffn";
    private static final String ENABLE_METAL_GATE_UP_MATVEC_FFN_VALUE =
            System.getProperty(ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY);
    private static final boolean ENABLE_METAL_GATE_UP_MATVEC_FFN_ENABLED =
            Boolean.TRUE.equals(parseOptionalBoolean(ENABLE_METAL_GATE_UP_MATVEC_FFN_VALUE));
    private static final String DISABLE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_matvec_ffn";
    private static final boolean DISABLE_METAL_MATVEC_FFN_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_MATVEC_FFN_PROPERTY);
    private static final String VALIDATE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.validate_metal_matvec_ffn";
    private static final boolean VALIDATE_METAL_MATVEC_FFN_ENABLED =
            Boolean.getBoolean(VALIDATE_METAL_MATVEC_FFN_PROPERTY);

    private DirectForwardFfnFastPathPolicy() {
    }

    static boolean isMetalFusedFfnDisabled() {
        return DISABLE_METAL_FUSED_FFN_ENABLED;
    }

    static boolean shouldUseMetalGegluFusedFfn(ModelConfigTraits traits) {
        return isGemma4FfnPolicyTarget(traits) || ENABLE_METAL_GEGLU_FUSED_FFN_ENABLED;
    }

    static boolean shouldUseQwenMetalFusedFfn() {
        return ENABLE_QWEN_METAL_FUSED_FFN_ENABLED;
    }

    static boolean shouldTryLocalFusedHalfFfn(ModelConfigTraits traits) {
        return !isGemma4FfnPolicyTarget(traits)
                || Boolean.TRUE.equals(allowGemma4FusedHalfFfnExplicit());
    }

    static boolean allowGemma4FusedHalfFfn() {
        if (DISABLE_GEMMA4_FUSED_HALF_FFN_ENABLED) {
            return false;
        }
        Boolean explicit = allowGemma4FusedHalfFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        return false;
    }

    static boolean allowGemma4FusedHalfFfn(long rows, ModelConfigTraits traits) {
        if (!isGemma4FfnPolicyTarget(traits)) {
            return true;
        }
        if (DISABLE_GEMMA4_FUSED_HALF_FFN_ENABLED) {
            return false;
        }
        Boolean explicit = allowGemma4FusedHalfFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        Boolean prefillExplicit = enableMetalFusedFfnPrefillExplicit();
        if (prefillExplicit != null) {
            return rows > 1L && prefillExplicit;
        }
        long effectiveMinRows = Math.max(2L, (long) gemma4FusedFfnPrefillMinRows());
        return rows >= effectiveMinRows && shouldUseMetalFusedFfnPrefill(traits);
    }

    static boolean shouldUseMetalFusedFfnPrefill(ModelConfigTraits traits) {
        Boolean explicit = enableMetalFusedFfnPrefillExplicit();
        if (explicit != null) {
            return explicit;
        }
        // M4 safetensor profiles now prefer the fused GEGLU prefill path for
        // Gemma-4 BF16: one native dispatch replaces gate/up projection,
        // activation, and down projection orchestration across prompt rows.
        return isGemma4FfnPolicyTarget(traits);
    }

    static boolean shouldUseMetalGegluMatvecFfn(ModelConfigTraits traits) {
        if (DISABLE_METAL_MATVEC_FFN_ENABLED) {
            return false;
        }
        Boolean explicit = enableMetalGegluMatvecFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Decode is memory-bound: the native GEGLU matvec avoids three Java
        // orchestrated launches for gate/up, activation, and down projection.
        return traits.gemma3Text() || isGemma4FfnPolicyTarget(traits);
    }

    static boolean shouldUseMetalSwigluMatvecFfn(ModelConfigTraits traits) {
        if (DISABLE_METAL_MATVEC_FFN_ENABLED) {
            return false;
        }
        Boolean explicit = enableMetalSwigluMatvecFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Qwen decode benefits from the native single-token SWIGLU FFN path:
        // one Metal dispatch replaces gate/up projection, activation, and down
        // projection orchestration. The caller still rejects prefill rows.
        return traits.qwenText();
    }

    static boolean shouldUseMetalGateUpMatvecFfn() {
        if (DISABLE_METAL_MATVEC_FFN_ENABLED) {
            return false;
        }
        return Boolean.TRUE.equals(enableMetalGateUpMatvecFfnExplicit());
    }

    static boolean shouldValidateMetalMatvecFfn(boolean traceFfnFastPath) {
        return VALIDATE_METAL_MATVEC_FFN_ENABLED || traceFfnFastPath;
    }

    private static Boolean allowGemma4FusedHalfFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY,
                ALLOW_GEMMA4_FUSED_HALF_FFN_EXPLICIT);
    }

    private static Boolean enableMetalFusedFfnPrefillExplicit() {
        return runtimeOptionalBooleanProperty(
                ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY,
                ENABLE_METAL_FUSED_FFN_PREFILL_EXPLICIT);
    }

    private static Boolean enableMetalGegluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY,
                ENABLE_METAL_GEGLU_MATVEC_FFN_EXPLICIT);
    }

    private static Boolean enableMetalSwigluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY,
                ENABLE_METAL_SWIGLU_MATVEC_FFN_EXPLICIT);
    }

    private static Boolean enableMetalGateUpMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY,
                ENABLE_METAL_GATE_UP_MATVEC_FFN_ENABLED);
    }

    private static int gemma4FusedFfnPrefillMinRows() {
        String runtimeValue = System.getProperty(GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY);
        if (runtimeValue == null || runtimeValue.isBlank()) {
            return GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS;
        }
        try {
            return Integer.parseInt(runtimeValue.trim());
        } catch (NumberFormatException ignored) {
            return GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS;
        }
    }

    private static boolean isGemma4FfnPolicyTarget(ModelConfigTraits traits) {
        return traits.gemma4Text() || traits.gemma4StylePerLayerInputs();
    }
}
