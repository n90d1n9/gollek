/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import java.util.Locale;
import java.util.Set;

/**
 * Prompt and tokenizer-control policy derived from model family traits.
 *
 * <p>This keeps prompt defaults, BOS insertion, and control-token validation
 * policy out of broader runtime traits so model-family prompt behavior can
 * evolve independently from attention and modality policy.
 */
public record ModelPromptTraits(
        PromptBosPolicy promptBosPolicy,
        Set<String> allowedControlTokenTexts,
        boolean validateContinuationTokensByDecode,
        boolean rejectEmptyDecodedTokens,
        boolean skipDefaultSystemPromptInjection,
        String defaultSystemPrompt) {

    public static final Set<String> GEMMA4_CONTROL_TOKEN_TEXTS = Set.of(
            "<|channel>",
            "<channel|>",
            "<|think|>");
    public static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
    public static final String QWEN_DEFAULT_SYSTEM_PROMPT =
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    public ModelPromptTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        defaultSystemPrompt = defaultSystemPrompt == null || defaultSystemPrompt.isBlank()
                ? DEFAULT_SYSTEM_PROMPT
                : defaultSystemPrompt;
    }

    public static ModelPromptTraits fromConfig(ModelConfig config) {
        String modelType = config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
        boolean gemma4Text = modelType.startsWith("gemma4");
        boolean gemma3Text = modelType.startsWith("gemma3");
        boolean gemmaFamily = modelType.startsWith("gemma");
        boolean qwenText = modelType.contains("qwen");
        return fromFlags(gemma4Text, gemma3Text, gemmaFamily, qwenText);
    }

    public static ModelPromptTraits fromRuntimeFlags(boolean gemma4Text, boolean gemma3Text, boolean qwenText) {
        return fromFlags(gemma4Text, gemma3Text, gemma3Text, qwenText);
    }

    public static ModelPromptTraits fromFlags(
            boolean gemma4Text,
            boolean gemma3Text,
            boolean gemmaFamily,
            boolean qwenText) {
        return new ModelPromptTraits(
                defaultPromptBosPolicy(gemma4Text, gemma3Text, gemmaFamily),
                allowedControlTokenTexts(gemma4Text),
                validatesContinuationTokensByDecode(gemma4Text),
                rejectsEmptyDecodedTokens(gemma4Text),
                skipsDefaultSystemPromptInjection(gemma4Text),
                defaultSystemPrompt(qwenText));
    }

    public static PromptBosPolicy defaultPromptBosPolicy(boolean gemma4Text, boolean gemma3Text) {
        return defaultPromptBosPolicy(gemma4Text, gemma3Text, gemma3Text);
    }

    public static PromptBosPolicy defaultPromptBosPolicy(
            boolean gemma4Text,
            boolean gemma3Text,
            boolean gemmaFamily) {
        if (gemma4Text) {
            return PromptBosPolicy.NEVER;
        }
        if (gemma3Text || gemmaFamily) {
            return PromptBosPolicy.GEMMA_TURN_AWARE;
        }
        return PromptBosPolicy.DEFAULT;
    }

    public static Set<String> allowedControlTokenTexts(boolean gemma4Text) {
        return gemma4Text ? GEMMA4_CONTROL_TOKEN_TEXTS : Set.of();
    }

    public static boolean validatesContinuationTokensByDecode(boolean gemma4Text) {
        return gemma4Text;
    }

    public static boolean rejectsEmptyDecodedTokens(boolean gemma4Text) {
        return gemma4Text;
    }

    public static boolean skipsDefaultSystemPromptInjection(boolean gemma4Text) {
        return gemma4Text;
    }

    public static String defaultSystemPrompt(boolean qwenText) {
        return qwenText ? QWEN_DEFAULT_SYSTEM_PROMPT : DEFAULT_SYSTEM_PROMPT;
    }
}
