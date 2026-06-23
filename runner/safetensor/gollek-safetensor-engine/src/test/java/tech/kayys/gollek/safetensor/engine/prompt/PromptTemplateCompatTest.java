/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.prompt;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PromptTemplateCompatTest {

    @Test
    void traitAwareFormattingUsesGemma4RuntimeTrait() {
        String prompt = PromptTemplateCompat.format(
                List.of(Message.system("be brief"), Message.user("where is jakarta")),
                "custom",
                ModelRuntimeTraits.builder()
                        .gemma4Text()
                        .build());

        assertTrue(prompt.startsWith("<bos><|turn>user\nbe brief\n\nwhere is jakarta"));
        assertTrue(prompt.endsWith("<|turn>model\n"));
    }

    @Test
    void traitAwareFormattingLetsRuntimeTraitsOverrideGemma4ModelType() {
        String prompt = PromptTemplateCompat.format(
                List.of(Message.user("where is jakarta")),
                "gemma4_text",
                ModelRuntimeTraits.builder()
                        .qwenText()
                        .build());

        assertFalse(prompt.startsWith("<bos><|turn>"));
        assertTrue(prompt.startsWith("<|im_start|>user\nwhere is jakarta"));
    }

    @Test
    void legacyFormattingStillSupportsGemma4ModelType() {
        String prompt = PromptTemplateCompat.format(
                List.of(Message.user("where is jakarta")),
                "gemma4_text");

        assertTrue(prompt.startsWith("<bos><|turn>user\nwhere is jakarta"));
    }
}
