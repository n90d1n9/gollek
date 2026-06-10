/*
 * Gollek Inference Engine - SafeTensor Text Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.models.core;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatTemplateFormatterTest {

    @Test
    void supportsSpecializedTemplateFamilies() {
        assertTrue(ChatTemplateFormatter.supportsModelType("llama3.1"));
        assertTrue(ChatTemplateFormatter.supportsModelType("gemma-4-text"));
        assertTrue(ChatTemplateFormatter.supportsModelType("qwen2.5"));
    }

    @Test
    void doesNotClaimFallbackChatMlForUnknownOrBlankModelTypes() {
        assertFalse(ChatTemplateFormatter.supportsModelType("custom"));
        assertFalse(ChatTemplateFormatter.supportsModelType(""));
        assertFalse(ChatTemplateFormatter.supportsModelType(null));
    }

    @Test
    void unknownModelTypesStillFormatAsChatMlFallback() {
        String prompt = ChatTemplateFormatter.format(List.of(Message.user("where is jakarta")), "custom");

        assertTrue(prompt.startsWith("<|im_start|>user\nwhere is jakarta"));
    }

    @Test
    void runtimeTraitsOverrideTraitManagedModelTypes() {
        String prompt = ChatTemplateFormatter.format(
                List.of(Message.user("where is jakarta")),
                "gemma4_text",
                ModelRuntimeTraits.builder()
                        .qwenText()
                        .build());

        assertFalse(prompt.startsWith("<bos><|turn>"));
        assertTrue(prompt.startsWith("<|im_start|>system\nYou are Qwen"));
    }

    @Test
    void emptyRuntimeTraitsSuppressTraitManagedFallbackModelTypes() {
        String prompt = ChatTemplateFormatter.format(
                List.of(Message.user("where is jakarta")),
                "qwen2.5",
                ModelRuntimeTraits.EMPTY);

        assertTrue(prompt.startsWith("<|im_start|>user\nwhere is jakarta"));
    }
}
