/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isSingleTokenHalfLinearCandidate;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardLinearCachePolicy {
    private static final DirectForwardLinearCacheOptions OPTIONS =
            DirectForwardLinearCacheOptions.fromSystemProperties();
    private static final DirectForwardLinearCacheRoutingPolicy ROUTING =
            DirectForwardLinearCacheRoutingPolicy.from(OPTIONS);

    private DirectForwardLinearCachePolicy() {
    }

    static AccelTensor cachedFfnDownHalfWeight(AccelTensor input, AccelTensor weight,
            ModelConfig config, String profileKey) {
        if (!ROUTING.isFfnDownHalfCacheProfile(profileKey)) {
            return null;
        }
        if (!ROUTING.shouldCacheFfnDownHalfWeight(
                profileKey,
                isSingleTokenHalfLinearCandidate(input, weight),
                weight.dequantizedByteSize(),
                config.getNumHiddenLayers())) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(OPTIONS.ffnDownLargeHalfCachePerTensorMaxBytes());
        return dequantized == weight ? null : dequantized;
    }

    static AccelTensor cachedLogitsLargeHalfWeight(AccelTensor input, AccelTensor weight, String profileKey) {
        if (!ROUTING.isLogitsLargeHalfCacheProfile(profileKey)) {
            return null;
        }
        if (!ROUTING.shouldCacheLogitsLargeHalfWeight(
                profileKey,
                isSingleTokenHalfLinearCandidate(input, weight),
                weight.dequantizedByteSize())) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(OPTIONS.logitsLargeHalfCacheMaxBytes());
        return dequantized == weight ? null : dequantized;
    }

    static AccelTensor toMetalHalfWeight(AccelTensor weight, boolean nativeBf16, boolean gemma4Text,
            boolean allowBf16ToF16, boolean allowMetalBf16Linear) {
        if (weight == null) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.F16) {
            return weight;
        }
        if (nativeBf16 && weight.quantType() == AccelTensor.QuantType.BF16) {
            return weight;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && gemma4Text) {
            return allowBf16ToF16
                    ? weight.toF16CachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes())
                    : null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear) {
            return weight.toF16CachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes());
        }
        // For quantized weights (INT4, BNB, INT8): dequantize to F16 once and cache.
        // This converts the weight to a Metal-compatible format with a one-time cost,
        // eliminating the per-token CPU dequantize+sgemm bottleneck on every decode step.
        if (weight.isQuantized()) {
            return weight.toQuantizedF16CachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes());
        }
        return null;
    }


    static AccelTensor cachedTransposedF16Weight(AccelTensor weight) {
        return weight.toF16Transposed2dCachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes());
    }
}
