/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

record FlashAttentionHeadLayout(int numQueryHeads, int numKeyValueHeads, int headDim) {

    static FlashAttentionHeadLayout resolve(AttentionInput in, ModelConfig config, int layerIdx) {
        int numQueryHeads = config.numAttentionHeads();
        int numKeyValueHeads = config.resolvedNumKvHeadsForLayer(layerIdx);
        int headDim = usesPackedQkv(in)
                ? resolvePackedHeadDim(in.qW, config, numQueryHeads, numKeyValueHeads)
                : resolveProjectedHeadDim(in.qW, config, numQueryHeads);
        return new FlashAttentionHeadLayout(numQueryHeads, numKeyValueHeads, headDim);
    }

    int queryProjectionDim() {
        return numQueryHeads * headDim;
    }

    int keyValueProjectionDim() {
        return numKeyValueHeads * headDim;
    }

    int packedQkvProjectionDim() {
        return queryProjectionDim() + keyValueProjectionDim() + keyValueProjectionDim();
    }

    boolean matchesPackedProjection(AccelTensor tensor) {
        return tensor != null && tensor.size(-1) == packedQkvProjectionDim();
    }

    private static boolean usesPackedQkv(AttentionInput in) {
        return in != null && in.arch != null && in.arch.hasFusedQKV();
    }

    private static int resolvePackedHeadDim(AccelTensor packedWeight, ModelConfig config,
            int numQueryHeads, int numKeyValueHeads) {
        int configured = config.resolvedHeadDim();
        long expectedRows = (long) (numQueryHeads + 2 * numKeyValueHeads) * configured;
        if (configured > 0 && packedWeight != null && packedWeight.size(0) == expectedRows) {
            return configured;
        }

        int denominator = numQueryHeads + 2 * numKeyValueHeads;
        if (packedWeight != null && denominator > 0 && packedWeight.size(0) % denominator == 0) {
            return Math.toIntExact(packedWeight.size(0) / denominator);
        }
        return configured;
    }

    private static int resolveProjectedHeadDim(AccelTensor queryWeight, ModelConfig config, int numQueryHeads) {
        if (queryWeight != null && numQueryHeads > 0 && queryWeight.size(0) % numQueryHeads == 0) {
            return Math.toIntExact(queryWeight.size(0) / numQueryHeads);
        }
        return config.resolvedHeadDim();
    }
}
