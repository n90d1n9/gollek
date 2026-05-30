/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isSingleTokenHalfLinearCandidate;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.multiplySaturating;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardLinearCachePolicy {
    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long METAL_F16_WEIGHT_CACHE_MAX_BYTES = Long.getLong(
            METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES);
    private static final String LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.logits_large_half_cache_max_bytes";
    private static final long DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long LOGITS_LARGE_HALF_CACHE_MAX_BYTES = Long.getLong(
            LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES);
    private static final String FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_total_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES = 1536L * 1024L * 1024L;
    private static final long FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES = Long.getLong(
            FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY,
            DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES);
    private static final String FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_per_tensor_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES = 64L * 1024L * 1024L;
    private static final long FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES = Long.getLong(
            FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY,
            DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES);

    private DirectForwardLinearCachePolicy() {
    }

    static AccelTensor cachedFfnDownHalfWeight(AccelTensor input, AccelTensor weight,
            ModelConfig config, String profileKey) {
        if (!"ffn_down".equals(profileKey) && !"ffn_down_nongated".equals(profileKey)) {
            return null;
        }
        if (!isSingleTokenHalfLinearCandidate(input, weight)) {
            return null;
        }
        long perTensorMaxBytes = FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES;
        if (perTensorMaxBytes <= 0L || weight.dequantizedByteSize() > perTensorMaxBytes) {
            return null;
        }
        long totalMaxBytes = FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES;
        if (totalMaxBytes <= 0L) {
            return null;
        }
        long estimatedModelBytes = multiplySaturating(weight.dequantizedByteSize(), config.numHiddenLayers());
        if (estimatedModelBytes > totalMaxBytes) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(perTensorMaxBytes);
        return dequantized == weight ? null : dequantized;
    }

    static AccelTensor cachedLogitsLargeHalfWeight(AccelTensor input, AccelTensor weight, String profileKey) {
        if (!"logits".equals(profileKey)) {
            return null;
        }
        if (!isSingleTokenHalfLinearCandidate(input, weight)) {
            return null;
        }
        long maxBytes = LOGITS_LARGE_HALF_CACHE_MAX_BYTES;
        if (maxBytes <= 0L || weight.dequantizedByteSize() > maxBytes) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(maxBytes);
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
                    ? weight.toF16CachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES)
                    : null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear) {
            return weight.toF16CachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
        }
        return null;
    }

    static AccelTensor cachedTransposedF16Weight(AccelTensor weight) {
        return weight.toF16Transposed2dCachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
    }
}
