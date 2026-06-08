/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

final class DirectForwardMetalLinearPolicy {
    private static final DirectForwardMetalLinearRoutingPolicy POLICY =
            DirectForwardMetalLinearRoutingPolicy.from(
                    DirectForwardMetalLinearOptions.fromSystemProperties());

    private DirectForwardMetalLinearPolicy() {
    }

    static boolean experimentalMetalLinearEnabled() {
        return POLICY.experimentalMetalLinearEnabled();
    }

    static boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        return POLICY.allowMetalBf16Linear(traits);
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return POLICY.shouldUseNativeMetalBf16Linear(weight, traits, profileKey);
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        return POLICY.shouldUseNativeMetalBf16Linear(weight, traits, profileKey, allowBf16ToF16);
    }

    static boolean canUseMetalHalfLinearCandidate(
            boolean canUseExperimentalMetalLinear,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return POLICY.canUseMetalHalfLinearCandidate(canUseExperimentalMetalLinear, input, weight, traits, profileKey);
    }

    static boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfigTraits traits, String profileKey) {
        return POLICY.canUseMetalHalfWeight(weight, traits, profileKey);
    }

    static boolean allowGemma4Bf16ToF16Linear(ModelConfigTraits traits, String profileKey) {
        return POLICY.allowGemma4Bf16ToF16Linear(traits, profileKey);
    }

    static boolean allowGemma4Bf16ToF16LinearForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        return POLICY.allowGemma4Bf16ToF16LinearForRows(rows, traits, profileKey, decodeLogitsPhase);
    }
}
