/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Centralizes model-family normalization and scale decisions for attention.
 */
final class FlashAttentionNormalizationPolicy {
    private final boolean gemma4Text;
    private final boolean architectureAddsOneToRmsNorm;
    private final double queryPreAttentionScalar;
    private final FlashAttentionNormalizationOptions options;

    private FlashAttentionNormalizationPolicy(boolean gemma4Text,
            boolean architectureAddsOneToRmsNorm,
            double queryPreAttentionScalar,
            FlashAttentionNormalizationOptions options) {
        this.gemma4Text = gemma4Text;
        this.architectureAddsOneToRmsNorm = architectureAddsOneToRmsNorm;
        this.queryPreAttentionScalar = queryPreAttentionScalar;
        this.options = options;
    }

    static FlashAttentionNormalizationPolicy resolve(ModelArchitecture architecture, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy) {
        return resolve(architecture, config, modelPolicy, FlashAttentionNormalizationOptions.fromSystemProperties());
    }

    static FlashAttentionNormalizationPolicy resolve(ModelArchitecture architecture, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy, FlashAttentionNormalizationOptions options) {
        boolean gemma4Text = modelPolicy != null && modelPolicy.gemma4Text();
        boolean addOne = architecture != null && architecture.addOneToRmsNormWeight();
        double scalar = config == null ? 1.0 : config.queryPreAttnScalar();
        if (options == null) {
            options = FlashAttentionNormalizationOptions.defaults();
        }
        return new FlashAttentionNormalizationPolicy(gemma4Text, addOne, scalar, options);
    }

    float attentionScale() {
        if (gemma4Text) {
            return 1.0f;
        }
        return (float) (1.0 / Math.sqrt(Math.max(1.0, queryPreAttentionScalar)));
    }

    boolean addOneToRmsNormWeight() {
        return architectureAddsOneToRmsNorm && !gemma4Text;
    }

    boolean qkNormEnabled() {
        return !gemma4Text || !options.disableGemma4QkNorm();
    }

    boolean valueNormEnabled() {
        return gemma4Text && !options.disableGemma4ValueNorm();
    }
}
