/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardMetalHalfMatvecLogitsPolicy(DirectForwardMetalHalfMatvecLogitsOptions options) {
    private static final String LOGITS_PROFILE_KEY = "logits";

    DirectForwardMetalHalfMatvecLogitsPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardMetalHalfMatvecLogitsPolicy from(DirectForwardMetalHalfMatvecLogitsOptions options) {
        return new DirectForwardMetalHalfMatvecLogitsPolicy(options);
    }

    boolean shouldUseMetalLogitsMpsMatvec(
            ModelConfigTraits traits,
            int outputDim,
            int inputDim,
            String profileKey) {
        if (!isLogitsProfile(profileKey) || options.disableMetalLogitsMpsMatvec()) {
            return false;
        }
        if (!Boolean.TRUE.equals(options.enableMetalLogitsMpsMatvec())) {
            return false;
        }
        if (traits.gemma4Text()) {
            return false;
        }
        return outputDim >= options.metalLogitsMpsMatvecMinOutput()
                && (options.metalLogitsMpsMatvecMaxInput() <= 0
                || inputDim <= options.metalLogitsMpsMatvecMaxInput());
    }

    int metalHalfMatvecMaxOutput(ModelConfigTraits traits, String profileKey, int defaultMaxOutput) {
        if (!isLogitsProfile(profileKey)) {
            return defaultMaxOutput;
        }
        if (traits.gemma4Text()) {
            return options.gemma4LogitsMetalHalfMatvecMaxOutput();
        }
        if (traits.gemma3Text()) {
            return options.gemma3LogitsMetalHalfMatvecMaxOutput();
        }
        if (traits.qwenText()) {
            return options.qwenLogitsMetalHalfMatvecMaxOutput();
        }
        return defaultMaxOutput;
    }

    boolean shouldUseGemma3LogitsMetalHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey,
            int maxOutput) {
        if (!isLogitsProfile(profileKey)
                || !traits.gemma3Text()
                || options.disableGemma3LogitsMetalHalfMatvec()) {
            return false;
        }
        if (maxOutput <= 0 || outputDim > maxOutput) {
            return false;
        }
        // M4 profiles currently prefer the default MPS matmul logits path for
        // FunctionGemma; keep this branch explicit so experiments cannot drift
        // into the default decode policy.
        return Boolean.TRUE.equals(options.enableGemma3LogitsMetalHalfMatvec());
    }

    private static boolean isLogitsProfile(String profileKey) {
        return LOGITS_PROFILE_KEY.equals(profileKey);
    }
}
