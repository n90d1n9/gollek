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
        AttentionRuntimeTraits attention,
        boolean audioModel,
        boolean visionModel,
        boolean multimodalModel) {

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
            boolean largeAttentionMatvecCandidate,
            boolean packedQkvProjection) {

        public static final AttentionRuntimeTraits EMPTY = ModelAttentionTraitsPolicy.empty();

        public static AttentionRuntimeTraits gemma4Text() {
            return ModelAttentionTraitsPolicy.gemma4Text();
        }

        public static AttentionRuntimeTraits gemma3Text() {
            return ModelAttentionTraitsPolicy.gemma3Text();
        }

        public static AttentionRuntimeTraits qwenText(ModelConfig config) {
            return ModelAttentionTraitsPolicy.qwenText(config);
        }

        public static AttentionRuntimeTraits phiText(ModelConfig config) {
            return ModelAttentionTraitsPolicy.phiText(config);
        }

        public static AttentionRuntimeTraits generic(ModelConfig config, boolean perLayerInputPath) {
            return ModelAttentionTraitsPolicy.generic(config, perLayerInputPath);
        }
    }

    public static final Set<String> GEMMA4_CONTROL_TOKEN_TEXTS = ModelPromptTraits.GEMMA4_CONTROL_TOKEN_TEXTS;
    public static final String DEFAULT_SYSTEM_PROMPT = ModelPromptTraits.DEFAULT_SYSTEM_PROMPT;
    public static final String QWEN_DEFAULT_SYSTEM_PROMPT = ModelPromptTraits.QWEN_DEFAULT_SYSTEM_PROMPT;

    public static final ModelRuntimeTraits EMPTY = new ModelRuntimeTraits(false, false, false, false);

    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(ModelRuntimeTraits traits) {
        return new Builder().copyFrom(traits);
    }

    public static ModelRuntimeTraits phiText(ModelConfig config) {
        return builder()
                .attention(AttentionRuntimeTraits.phiText(config))
                .build();
    }

    public ModelRuntimeTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        attention = attention == null
                ? defaultAttentionTraits(gemma4Text, gemma3Text, qwenText, perLayerInputPath, null)
                : attention;
        multimodalModel = multimodalModel || audioModel || visionModel;
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                ModelPromptTraits.defaultPromptBosPolicy(gemma4Text, gemma3Text),
                ModelPromptTraits.allowedControlTokenTexts(gemma4Text),
                ModelPromptTraits.validatesContinuationTokensByDecode(gemma4Text),
                ModelPromptTraits.rejectsEmptyDecodedTokens(gemma4Text),
                null,
                false,
                false,
                false);
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, null, false, false, false);
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, attention, false, false, false);
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention, boolean audioModel) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, attention, audioModel, false, audioModel);
    }

    public ModelRuntimeTraits(boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention, boolean audioModel, boolean multimodalModel) {
        this(gemma4Text, gemma3Text, qwenText, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, attention,
                audioModel, false, multimodalModel);
    }

    /**
     * Compatibility alias for callers that still derive traits directly from
     * config metadata. New internal runtime paths should call
     * {@link #fallbackFromConfig(ModelConfig)} to make the fallback boundary
     * explicit.
     */
    public static ModelRuntimeTraits fromConfig(ModelConfig config) {
        return fallbackFromConfig(config);
    }

    /**
     * Derives coarse runtime traits directly from config metadata.
     *
     * <p>This is intentionally a fallback path for generic loaders and legacy
     * adapters. Model-family modules should prefer overriding runtime traits so
     * tokenizer, attention, and modality policy stay owned by the farm.</p>
     */
    public static ModelRuntimeTraits fallbackFromConfig(ModelConfig config) {
        if (config == null) {
            return EMPTY;
        }
        String modelType = normalizedModelType(config);
        boolean gemma4Text = modelType.startsWith("gemma4");
        boolean gemma3Text = modelType.startsWith("gemma3");
        boolean gemmaFamily = modelType.startsWith("gemma");
        boolean qwenText = modelType.contains("qwen");
        ModelPromptTraits prompt = ModelPromptTraits.fromFlags(gemma4Text, gemma3Text, gemmaFamily, qwenText);
        ModelModalityTraits modality = ModelModalityTraits.fromConfig(config);
        boolean perLayerInputPath = config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0;
        return builder()
                .gemma4Text(gemma4Text)
                .gemma3Text(gemma3Text)
                .qwenText(qwenText)
                .perLayerInputPath(perLayerInputPath)
                .prompt(prompt)
                .attention(defaultAttentionTraits(gemma4Text, gemma3Text, qwenText, perLayerInputPath, config))
                .modalities(modality)
                .build();
    }

    public ModelRuntimeTraits withDetectedModalities(ModelConfig config) {
        if (config == null) {
            return this;
        }
        ModelModalityTraits modality = ModelModalityTraits.fromConfig(config);
        if ((!modality.audioModel() || audioModel)
                && (!modality.visionModel() || visionModel)
                && (!modality.multimodalModel() || multimodalModel)) {
            return this;
        }
        return builder(this)
                .audioModel(audioModel || modality.audioModel())
                .visionModel(visionModel || modality.visionModel())
                .multimodalModel(multimodalModel || modality.multimodalModel())
                .build();
    }

    public static boolean detectAudioModel(ModelConfig config) {
        return ModelModalityTraits.detectAudioModel(config);
    }

    public static boolean detectVisionModel(ModelConfig config) {
        return ModelModalityTraits.detectVisionModel(config);
    }

    public static boolean detectMultimodalModel(ModelConfig config) {
        return ModelModalityTraits.detectMultimodalModel(config);
    }

    public boolean skipDefaultSystemPromptInjection() {
        return ModelPromptTraits.skipsDefaultSystemPromptInjection(gemma4Text);
    }

    public String defaultSystemPrompt() {
        return ModelPromptTraits.defaultSystemPrompt(qwenText);
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }

    private static AttentionRuntimeTraits defaultAttentionTraits(boolean gemma4Text, boolean gemma3Text,
            boolean qwenText, boolean perLayerInputPath, ModelConfig config) {
        return ModelAttentionTraitsPolicy.fromFlags(
                gemma4Text, gemma3Text, qwenText, perLayerInputPath, config);
    }

    /**
     * Named builder for model-family profiles. This keeps farm-owned runtime
     * policy readable and avoids positional boolean mistakes as traits grow.
     */
    public static final class Builder {
        private boolean gemma4Text;
        private boolean gemma3Text;
        private boolean qwenText;
        private boolean perLayerInputPath;
        private PromptBosPolicy promptBosPolicy;
        private Set<String> allowedControlTokenTexts;
        private boolean allowedControlTokenTextsSet;
        private boolean validateContinuationTokensByDecode;
        private boolean validateContinuationTokensByDecodeSet;
        private boolean rejectEmptyDecodedTokens;
        private boolean rejectEmptyDecodedTokensSet;
        private AttentionRuntimeTraits attention;
        private boolean audioModel;
        private boolean visionModel;
        private boolean multimodalModel;

        private Builder() {
        }

        public Builder gemma4Text() {
            return gemma4Text(true);
        }

        public Builder gemma4Text(boolean gemma4Text) {
            this.gemma4Text = gemma4Text;
            return this;
        }

        public Builder gemma3Text() {
            return gemma3Text(true);
        }

        public Builder gemma3Text(boolean gemma3Text) {
            this.gemma3Text = gemma3Text;
            return this;
        }

        public Builder qwenText() {
            return qwenText(true);
        }

        public Builder qwenText(boolean qwenText) {
            this.qwenText = qwenText;
            return this;
        }

        public Builder perLayerInputPath() {
            return perLayerInputPath(true);
        }

        public Builder perLayerInputPath(boolean perLayerInputPath) {
            this.perLayerInputPath = perLayerInputPath;
            return this;
        }

        public Builder prompt(ModelPromptTraits prompt) {
            if (prompt == null) {
                return this;
            }
            return promptBosPolicy(prompt.promptBosPolicy())
                    .allowedControlTokenTexts(prompt.allowedControlTokenTexts())
                    .validateContinuationTokensByDecode(prompt.validateContinuationTokensByDecode())
                    .rejectEmptyDecodedTokens(prompt.rejectEmptyDecodedTokens());
        }

        public Builder promptBosPolicy(PromptBosPolicy promptBosPolicy) {
            this.promptBosPolicy = promptBosPolicy;
            return this;
        }

        public Builder allowedControlTokenTexts(Set<String> allowedControlTokenTexts) {
            this.allowedControlTokenTexts = allowedControlTokenTexts == null
                    ? Set.of()
                    : Set.copyOf(allowedControlTokenTexts);
            this.allowedControlTokenTextsSet = true;
            return this;
        }

        public Builder validateContinuationTokensByDecode(boolean validateContinuationTokensByDecode) {
            this.validateContinuationTokensByDecode = validateContinuationTokensByDecode;
            this.validateContinuationTokensByDecodeSet = true;
            return this;
        }

        public Builder rejectEmptyDecodedTokens(boolean rejectEmptyDecodedTokens) {
            this.rejectEmptyDecodedTokens = rejectEmptyDecodedTokens;
            this.rejectEmptyDecodedTokensSet = true;
            return this;
        }

        public Builder attention(AttentionRuntimeTraits attention) {
            this.attention = attention;
            return this;
        }

        public Builder audioModel() {
            return audioModel(true);
        }

        public Builder audioModel(boolean audioModel) {
            this.audioModel = audioModel;
            return this;
        }

        public Builder visionModel() {
            return visionModel(true);
        }

        public Builder visionModel(boolean visionModel) {
            this.visionModel = visionModel;
            return this;
        }

        public Builder multimodalModel() {
            return multimodalModel(true);
        }

        public Builder multimodalModel(boolean multimodalModel) {
            this.multimodalModel = multimodalModel;
            return this;
        }

        public Builder modalities(ModelModalityTraits modality) {
            if (modality == null) {
                return this;
            }
            return audioModel(modality.audioModel())
                    .visionModel(modality.visionModel())
                    .multimodalModel(modality.multimodalModel());
        }

        public ModelRuntimeTraits build() {
            return new ModelRuntimeTraits(
                    gemma4Text,
                    gemma3Text,
                    qwenText,
                    perLayerInputPath,
                    promptBosPolicy == null
                            ? ModelPromptTraits.defaultPromptBosPolicy(gemma4Text, gemma3Text)
                            : promptBosPolicy,
                    allowedControlTokenTextsSet
                            ? allowedControlTokenTexts
                            : ModelPromptTraits.allowedControlTokenTexts(gemma4Text),
                    validateContinuationTokensByDecodeSet
                            ? validateContinuationTokensByDecode
                            : ModelPromptTraits.validatesContinuationTokensByDecode(gemma4Text),
                    rejectEmptyDecodedTokensSet
                            ? rejectEmptyDecodedTokens
                            : ModelPromptTraits.rejectsEmptyDecodedTokens(gemma4Text),
                    attention,
                    audioModel,
                    visionModel,
                    multimodalModel);
        }

        private Builder copyFrom(ModelRuntimeTraits traits) {
            if (traits == null) {
                return this;
            }
            return gemma4Text(traits.gemma4Text())
                    .gemma3Text(traits.gemma3Text())
                    .qwenText(traits.qwenText())
                    .perLayerInputPath(traits.perLayerInputPath())
                    .promptBosPolicy(traits.promptBosPolicy())
                    .allowedControlTokenTexts(traits.allowedControlTokenTexts())
                    .validateContinuationTokensByDecode(traits.validateContinuationTokensByDecode())
                    .rejectEmptyDecodedTokens(traits.rejectEmptyDecodedTokens())
                    .attention(traits.attention())
                    .audioModel(traits.audioModel())
                    .visionModel(traits.visionModel())
                    .multimodalModel(traits.multimodalModel());
        }
    }
}
