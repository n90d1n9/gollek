/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalLinearRoutingPolicy(
        DirectForwardMetalLinearOptions options,
        DirectForwardGemma4Bf16LinearPolicy gemma4Bf16Policy) {

    DirectForwardMetalLinearRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
        gemma4Bf16Policy = Objects.requireNonNull(gemma4Bf16Policy, "gemma4Bf16Policy");
    }

    static DirectForwardMetalLinearRoutingPolicy from(DirectForwardMetalLinearOptions options) {
        DirectForwardMetalLinearOptions resolved =
                options == null ? DirectForwardMetalLinearOptions.defaults() : options;
        return new DirectForwardMetalLinearRoutingPolicy(
                resolved,
                DirectForwardGemma4Bf16LinearPolicy.from(resolved.gemma4Bf16Options()));
    }

    boolean experimentalMetalLinearEnabled() {
        return options.experimentalMetalLinearEnabled();
    }

    boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        if (traits.gemma4Text()) {
            return gemma4Bf16Policy.allowMetalBf16Linear(traits);
        }
        return allowGenericMetalBf16Linear();
    }

    boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return shouldUseNativeMetalBf16Linear(
                weight,
                traits,
                profileKey,
                allowGemma4Bf16ToF16Linear(traits, profileKey));
    }

    boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        return preferNativeMetalBf16Linear(traits, profileKey, allowBf16ToF16)
                && allowMetalBf16Linear(traits)
                && weight != null
                && weight.quantType() == AccelTensor.QuantType.BF16;
    }

    boolean canUseMetalHalfLinearCandidate(
            boolean canUseExperimentalMetalLinear,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseExperimentalMetalLinear) {
            return false;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (weight == null) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.gemma4Text()
                && !shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                && !allowGemma4Bf16ToF16Linear(traits, profileKey)) {
            return false;
        }
        if (quantType != AccelTensor.QuantType.F16
                && (quantType != AccelTensor.QuantType.BF16 || !allowMetalBf16Linear(traits))) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        int inputRank = input.rank();
        if (inputRank < 2 || weight.rank() != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, input.size(-1));
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputRank - 2; i++) {
            batchProduct *= input.size(i);
        }
        return batchProduct == 1L;
    }

    boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfigTraits traits, String profileKey) {
        if (weight == null || weight.rank() != 2 || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.gemma4Text()) {
            return shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                    || allowGemma4Bf16ToF16Linear(traits, profileKey);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(traits));
    }

    boolean allowGemma4Bf16ToF16Linear(ModelConfigTraits traits, String profileKey) {
        return gemma4Bf16Policy.allowBf16ToF16Linear(traits, profileKey);
    }

    boolean allowGemma4Bf16ToF16LinearForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        return gemma4Bf16Policy.allowBf16ToF16LinearForRows(rows, traits, profileKey, decodeLogitsPhase);
    }

    private boolean allowGenericMetalBf16Linear() {
        if (options.disableExperimentalMetalBf16Linear()) {
            return false;
        }
        if (options.experimentalMetalBf16Linear() != null) {
            return options.experimentalMetalBf16Linear();
        }
        return false;
    }

    private boolean preferNativeMetalBf16Linear(
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        if (traits.gemma4Text()) {
            if (allowBf16ToF16) {
                return false;
            }
            return gemma4Bf16Policy.preferNativeMetalBf16Linear(traits);
        }
        return options.preferNativeMetalBf16Linear();
    }
}
