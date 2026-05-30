/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.spi.model;

import java.util.Locale;
import java.util.Set;

/**
 * Runtime-only model family traits used by inference engines for policy
 * decisions that do not belong in tensor-name resolution.
 */
public record ModelRuntimeTraits(
        boolean gemma4Text,
        boolean gemma3Text,
        boolean qwenText,
        boolean perLayerInputPath,
        PromptBosPolicy promptBosPolicy,
        Set<String> allowedControlTokenTexts,
        boolean validateContinuationTokensByDecode,
        boolean rejectEmptyDecodedTokens,
        AttentionRuntimeTraits attention) {

    public enum PromptBosPolicy {
        DEFAULT,
        NEVER,
        GEMMA_TURN_AWARE
    }

    public record AttentionRuntimeTraits(
            boolean splitHalfRope,
            boolean attentionSoftCapAppliesToFinalLogitsOnly,
            boolean preferMetalPerHeadRmsNorm,
            boolean preferNativeMetalBf16Linear,
            boolean disallowBf16ToF16LinearConversion,
            boolean restrictLegacyMetalAttentionBridge,
            boolean supportsForcedDenseAttention,
            int defaultPagedMetalPrefillMaxTokens,
            boolean compactAttentionMatvecCandidate,
            boolean largeAttentionMatvecCandidate) {

        private static final int DEFAULT_QWEN_PAGED_METAL_PREFILL_MAX_TOKENS = 128;

        public static final AttentionRuntimeTraits EMPTY = new AttentionRuntimeTraits(
                false, false, false, false, false, false, false, 0, false, false);

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
                    isLargeAttentionMatvecCandidate(config, false, false));
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
                    isLargeAttentionMatvecCandidate(config, false, perLayerInputPath));
        }
    }

    public static final Set<String> GEMMA4_CONTROL_TOKEN_TEXTS = Set.of(
            "<|channel>",
            "<channel|>",
            "<|think|>");

    public static final ModelRuntimeTraits EMPTY = new ModelRuntimeTraits(false, false, false, false);

    public ModelRuntimeTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        attention = attention == null
                ? defaultAttentionTraits(gemma4Text, gemma3Text, qwenText, perLayerInputPath, null)
                : attention;
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                defaultPromptBosPolicy(gemma4Text, gemma3Text),
                gemma4Text ? GEMMA4_CONTROL_TOKEN_TEXTS : Set.of(),
                gemma4Text,
                gemma4Text,
                null);
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, null);
    }

    public static ModelRuntimeTraits fromConfig(ModelConfig config) {
        if (config == null) {
            return EMPTY;
        }
        String modelType = config.modelType() == null ? "" : config.modelType().toLowerCase(Locale.ROOT);
        boolean gemma4Text = modelType.startsWith("gemma4");
        boolean gemma3Text = modelType.startsWith("gemma3");
        boolean gemmaFamily = modelType.startsWith("gemma");
        return new ModelRuntimeTraits(
                gemma4Text,
                gemma3Text,
                modelType.contains("qwen"),
                config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0,
                gemma4Text
                        ? PromptBosPolicy.NEVER
                        : (gemmaFamily ? PromptBosPolicy.GEMMA_TURN_AWARE : PromptBosPolicy.DEFAULT),
                gemma4Text ? GEMMA4_CONTROL_TOKEN_TEXTS : Set.of(),
                gemma4Text,
                gemma4Text,
                defaultAttentionTraits(gemma4Text, gemma3Text, modelType.contains("qwen"),
                        config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0, config));
    }

    private static PromptBosPolicy defaultPromptBosPolicy(boolean gemma4Text, boolean gemma3Text) {
        if (gemma4Text) {
            return PromptBosPolicy.NEVER;
        }
        if (gemma3Text) {
            return PromptBosPolicy.GEMMA_TURN_AWARE;
        }
        return PromptBosPolicy.DEFAULT;
    }

    private static AttentionRuntimeTraits defaultAttentionTraits(boolean gemma4Text, boolean gemma3Text,
            boolean qwenText, boolean perLayerInputPath, ModelConfig config) {
        if (gemma4Text) {
            return AttentionRuntimeTraits.gemma4Text();
        }
        if (gemma3Text) {
            return AttentionRuntimeTraits.gemma3Text();
        }
        if (qwenText) {
            return AttentionRuntimeTraits.qwenText(config);
        }
        return AttentionRuntimeTraits.generic(config, perLayerInputPath);
    }

    private static boolean isCompactAttentionMatvecCandidate(ModelConfig config) {
        return config != null
                && config.numHiddenLayers() >= 20
                && config.hiddenSize() <= 2048
                && config.intermediateSize() >= 2048;
    }

    private static boolean isLargeAttentionMatvecCandidate(ModelConfig config, boolean gemma4Text,
            boolean perLayerInputPath) {
        if (config == null || gemma4Text || perLayerInputPath) {
            return false;
        }
        return config.numHiddenLayers() >= 30
                && config.intermediateSize() >= 4096
                && config.hiddenSize() <= 4096;
    }
}
