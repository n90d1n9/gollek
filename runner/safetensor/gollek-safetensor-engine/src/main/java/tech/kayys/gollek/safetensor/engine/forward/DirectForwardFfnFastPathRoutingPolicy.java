/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.runtimeOptionalBooleanProperty;

import java.util.Objects;

record DirectForwardFfnFastPathRoutingPolicy(DirectForwardFfnFastPathOptions options) {

    DirectForwardFfnFastPathRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardFfnFastPathRoutingPolicy from(DirectForwardFfnFastPathOptions options) {
        return new DirectForwardFfnFastPathRoutingPolicy(options);
    }

    boolean shouldUseMetalGegluFusedFfn(ModelConfigTraits traits) {
        return isGemma4FfnPolicyTarget(traits) || options.enableMetalGegluFusedFfn();
    }

    boolean shouldTryLocalFusedHalfFfn(ModelConfigTraits traits) {
        return !isGemma4FfnPolicyTarget(traits)
                || Boolean.TRUE.equals(allowGemma4FusedHalfFfnExplicit());
    }

    boolean isGemma4FusedHalfFfnAllowed() {
        if (options.disableGemma4FusedHalfFfn()) {
            return false;
        }
        Boolean explicit = allowGemma4FusedHalfFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        return false;
    }

    boolean allowGemma4FusedHalfFfn(long rows, ModelConfigTraits traits) {
        if (!isGemma4FfnPolicyTarget(traits)) {
            return true;
        }
        if (options.disableGemma4FusedHalfFfn()) {
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

    boolean shouldUseMetalFusedFfnPrefill(ModelConfigTraits traits) {
        Boolean explicit = enableMetalFusedFfnPrefillExplicit();
        if (explicit != null) {
            return explicit;
        }
        // M4 safetensor profiles now prefer the fused GEGLU prefill path for
        // Gemma-4 BF16: one native dispatch replaces gate/up projection,
        // activation, and down projection orchestration across prompt rows.
        return isGemma4FfnPolicyTarget(traits);
    }

    boolean shouldUseMetalGegluMatvecFfn(ModelConfigTraits traits) {
        if (options.disableMetalMatvecFfn()) {
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

    boolean shouldUseMetalSwigluMatvecFfn(ModelConfigTraits traits) {
        if (options.disableMetalMatvecFfn()) {
            return false;
        }
        Boolean explicit = enableMetalSwigluMatvecFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Qwen and IBM Granite decode benefit from the native single-token SwiGLU
        // FFN path: one Metal dispatch replaces gate/up projection, activation, and
        // down projection orchestration. Both architectures use SILU activation and
        // the same gated FFN weight layout. The caller still rejects prefill rows.
        return traits.qwenText() || traits.siluGated();
    }

    boolean shouldUseMetalGateUpMatvecFfn() {
        if (options.disableMetalMatvecFfn()) {
            return false;
        }
        return Boolean.TRUE.equals(enableMetalGateUpMatvecFfnExplicit());
    }

    boolean shouldUseMetalMatvecFfnPrefillRows(ModelConfigTraits traits, long rows) {
        if (options.disableMetalMatvecFfn() || !isGemma4FfnPolicyTarget(traits)) {
            return false;
        }
        if (!Boolean.TRUE.equals(enableMetalMatvecFfnPrefillRowsExplicit())) {
            return false;
        }
        long maxRows = Math.max(0L, (long) metalMatvecFfnPrefillMaxRows());
        return rows > 1L && rows <= maxRows;
    }

    boolean shouldPreferMetalFusedFfnPrefillOverMatvecRows(ModelConfigTraits traits, long rows) {
        return rows > 1L
                && isGemma4FfnPolicyTarget(traits)
                && !Boolean.TRUE.equals(preferMetalMatvecFfnPrefillRowsExplicit())
                && !options.disableMetalFusedFfn()
                && shouldUseMetalFusedFfnPrefill(traits)
                && allowGemma4FusedHalfFfn(rows, traits);
    }

    boolean shouldValidateMetalMatvecFfn(boolean traceFfnFastPath) {
        return options.validateMetalMatvecFfn() || traceFfnFastPath;
    }

    private Boolean allowGemma4FusedHalfFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY,
                options.gemma4FusedHalfFfnExplicit());
    }

    private Boolean enableMetalFusedFfnPrefillExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY,
                options.enableMetalFusedFfnPrefill());
    }

    private Boolean enableMetalGegluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY,
                options.enableMetalGegluMatvecFfn());
    }

    private Boolean enableMetalSwigluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY,
                options.enableMetalSwigluMatvecFfn());
    }

    private Boolean enableMetalGateUpMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY,
                options.enableMetalGateUpMatvecFfn());
    }

    private Boolean enableMetalMatvecFfnPrefillRowsExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY,
                options.enableMetalMatvecFfnPrefillRows());
    }

    private Boolean preferMetalMatvecFfnPrefillRowsExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY,
                options.preferMetalMatvecFfnPrefillRows());
    }

    private int gemma4FusedFfnPrefillMinRows() {
        String runtimeValue = System.getProperty(
                DirectForwardFfnFastPathOptions.GEMMA4_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY);
        if (runtimeValue == null || runtimeValue.isBlank()) {
            return options.gemma4FusedFfnPrefillMinRows();
        }
        try {
            return Integer.parseInt(runtimeValue.trim());
        } catch (NumberFormatException ignored) {
            return options.gemma4FusedFfnPrefillMinRows();
        }
    }

    private int metalMatvecFfnPrefillMaxRows() {
        String runtimeValue = System.getProperty(
                DirectForwardFfnFastPathOptions.METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY);
        if (runtimeValue == null || runtimeValue.isBlank()) {
            return options.metalMatvecFfnPrefillMaxRows();
        }
        try {
            return Integer.parseInt(runtimeValue.trim());
        } catch (NumberFormatException ignored) {
            return options.metalMatvecFfnPrefillMaxRows();
        }
    }

    private static boolean isGemma4FfnPolicyTarget(ModelConfigTraits traits) {
        return traits.gemma4Text() || traits.gemma4StylePerLayerInputs();
    }
}
