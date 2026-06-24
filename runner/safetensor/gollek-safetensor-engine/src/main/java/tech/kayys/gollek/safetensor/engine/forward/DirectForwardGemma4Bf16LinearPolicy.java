/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardGemma4Bf16LinearPolicy(DirectForwardGemma4Bf16LinearOptions options) {
    private static final String LOGITS_PROFILE_KEY = "logits";
    private static final String FFN_PROFILE_PREFIX = "ffn_";

    DirectForwardGemma4Bf16LinearPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardGemma4Bf16LinearPolicy from(DirectForwardGemma4Bf16LinearOptions options) {
        return new DirectForwardGemma4Bf16LinearPolicy(options);
    }

    boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        if (!traits.gemma4Text() || options.disableMetalBf16Linear()) {
            return false;
        }
        if (options.enableMetalBf16Linear() != null) {
            return options.enableMetalBf16Linear();
        }
        return true;
    }

    boolean preferNativeMetalBf16Linear(ModelConfigTraits traits) {
        if (options.enableMetalBf16Linear() != null) {
            return options.enableMetalBf16Linear() && allowMetalBf16Linear(traits);
        }
        return allowMetalBf16Linear(traits);
    }

    boolean allowBf16ToF16Linear(ModelConfigTraits traits, String profileKey) {
        if (!traits.gemma4Text() || options.disableBf16ToF16Linear()) {
            return false;
        }
        if (LOGITS_PROFILE_KEY.equals(profileKey) && options.disableBf16ToF16LogitsLinear()) {
            return false;
        }
        if (options.enableBf16ToF16Linear() != null) {
            return options.enableBf16ToF16Linear();
        }
        if (LOGITS_PROFILE_KEY.equals(profileKey)) {
            if (options.enableBf16ToF16LogitsLinear() != null) {
                return options.enableBf16ToF16LogitsLinear();
            }
            return true;
        }
        if (options.enableBf16ToF16FfnLinear() != null) {
            return options.enableBf16ToF16FfnLinear();
        }
        return true;
    }

    boolean allowBf16ToF16LinearForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        // Keep prompt prefill on native BF16. Cached F16 conversion is useful
        // for repeated decode matvecs, but paying it before first token hurts TTFT.
        if (rows != 1L || !allowBf16ToF16Linear(traits, profileKey)) {
            return false;
        }
        return !LOGITS_PROFILE_KEY.equals(profileKey) || decodeLogitsPhase;
    }

    private static boolean isFfnProjection(String profileKey) {
        return profileKey != null && profileKey.startsWith(FFN_PROFILE_PREFIX);
    }
}
