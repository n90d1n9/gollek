/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.model.ModelRuntimeTraits.AttentionRuntimeTraits;

/**
 * Model-family attention runtime defaults.
 *
 * <p>This keeps attention kernel and projection preferences separate from
 * broader runtime traits. Model farms can add family-specific attention policy
 * here without growing {@link ModelRuntimeTraits}.
 */
public final class ModelAttentionTraitsPolicy {

    public static final int DEFAULT_QWEN_PAGED_METAL_PREFILL_MAX_TOKENS = 128;

    private ModelAttentionTraitsPolicy() {
    }

    public static AttentionRuntimeTraits empty() {
        return new AttentionRuntimeTraits(
                false, false, false, false, false, false, false, 0, false, false, false);
    }

    public static AttentionRuntimeTraits fromFlags(boolean gemma4Text, boolean gemma3Text,
            boolean qwenText, boolean perLayerInputPath, ModelConfig config) {
        if (gemma4Text) {
            return gemma4Text();
        }
        if (gemma3Text) {
            return gemma3Text();
        }
        if (qwenText) {
            return qwenText(config);
        }
        return generic(config, perLayerInputPath);
    }

    public static AttentionRuntimeTraits gemma4Text() {
        return new AttentionRuntimeTraits(
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                0,
                false,
                false,
                false);
    }

    public static AttentionRuntimeTraits gemma3Text() {
        return new AttentionRuntimeTraits(
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                0,
                false,
                false,
                false);
    }

    public static AttentionRuntimeTraits qwenText(ModelConfig config) {
        boolean compact = isCompactAttentionMatvecCandidate(config);
        return new AttentionRuntimeTraits(
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                compact ? DEFAULT_QWEN_PAGED_METAL_PREFILL_MAX_TOKENS : 0,
                compact,
                isLargeAttentionMatvecCandidate(config, false, false),
                false);
    }

    public static AttentionRuntimeTraits phiText(ModelConfig config) {
        return new AttentionRuntimeTraits(
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                0,
                false,
                isLargeAttentionMatvecCandidate(config, false, false),
                true);
    }

    public static AttentionRuntimeTraits generic(ModelConfig config, boolean perLayerInputPath) {
        return new AttentionRuntimeTraits(
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                0,
                false,
                isLargeAttentionMatvecCandidate(config, false, perLayerInputPath),
                false);
    }

    public static boolean isCompactAttentionMatvecCandidate(ModelConfig config) {
        return config != null
                && config.numHiddenLayers() >= 20
                && config.hiddenSize() <= 2048
                && config.intermediateSize() >= 2048;
    }

    public static boolean isLargeAttentionMatvecCandidate(ModelConfig config, boolean gemma4Text,
            boolean perLayerInputPath) {
        if (config == null || gemma4Text || perLayerInputPath) {
            return false;
        }
        return config.numHiddenLayers() >= 30
                && config.intermediateSize() >= 4096
                && config.hiddenSize() <= 4096;
    }
}
