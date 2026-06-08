/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

record FlashAttentionDispatchRequest(AccelTensor query, AccelTensor key, AccelTensor value,
        KVCacheManager.KVCacheSession kvSession, SharedKvState sharedKvState, ModelConfig config,
        FlashAttentionModelPolicy modelPolicy, int layerIdx, int kvLayerIdx, int startPos, int seqLen,
        int numQueryHeads, int numKeyValueHeads, int headDim, float scale, boolean causal,
        float attentionSoftCap, boolean useDenseSharedKvState) {
}
