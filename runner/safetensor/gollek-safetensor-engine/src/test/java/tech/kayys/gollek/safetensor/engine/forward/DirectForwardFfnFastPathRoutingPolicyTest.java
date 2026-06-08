/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardFfnFastPathRoutingPolicyTest {

    @Test
    void gemma4UsesMetalGegluFusedFfnByDefault() {
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldUseMetalGegluFusedFfn(gemma4()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldUseMetalGegluFusedFfn(generic()));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withMetalFusedFfn(false, false, true))
                .shouldUseMetalGegluFusedFfn(generic()));
    }

    @Test
    void localFusedHalfFfnSkipsGemma4UnlessExplicitlyAllowed() {
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldTryLocalFusedHalfFfn(generic()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldTryLocalFusedHalfFfn(gemma4()));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withGemma4FusedHalfFfn(true, false))
                .shouldTryLocalFusedHalfFfn(gemma4()));
    }

    @Test
    void gemma4FusedHalfFfnGlobalFlagHonorsDisableAndExplicit() {
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).isGemma4FusedHalfFfnAllowed());
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withGemma4FusedHalfFfn(true, false))
                .isGemma4FusedHalfFfnAllowed());
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()
                .withGemma4FusedHalfFfn(true, true))
                .isGemma4FusedHalfFfnAllowed());
    }

    @Test
    void gemma4FusedHalfFfnPrefillRequiresEffectiveMinimumRows() {
        DirectForwardFfnFastPathRoutingPolicy policy = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfnPrefill(null, 3));

        assertFalse(policy.allowGemma4FusedHalfFfn(1, gemma4()));
        assertFalse(policy.allowGemma4FusedHalfFfn(2, gemma4()));
        assertTrue(policy.allowGemma4FusedHalfFfn(3, gemma4()));
        assertTrue(policy.allowGemma4FusedHalfFfn(1, generic()));
    }

    @Test
    void gemma4FusedHalfFfnPrefillExplicitStillRejectsDecodeRows() {
        DirectForwardFfnFastPathRoutingPolicy policy = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfnPrefill(true, 2));

        assertFalse(policy.allowGemma4FusedHalfFfn(1, gemma4()));
        assertTrue(policy.allowGemma4FusedHalfFfn(2, gemma4()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()
                .withMetalFusedFfnPrefill(false, 2))
                .allowGemma4FusedHalfFfn(2, gemma4()));
    }

    @Test
    void matvecFfnDefaultsFollowModelFamiliesAndDisableWins() {
        DirectForwardFfnFastPathRoutingPolicy defaults = policy(DirectForwardFfnFastPathOptions.defaults());

        assertTrue(defaults.shouldUseMetalGegluMatvecFfn(gemma3()));
        assertTrue(defaults.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(defaults.shouldUseMetalGegluMatvecFfn(generic()));
        assertTrue(defaults.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(defaults.shouldUseMetalSwigluMatvecFfn(generic()));
        assertFalse(defaults.shouldUseMetalGateUpMatvecFfn());

        DirectForwardFfnFastPathRoutingPolicy disabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(true, true, true, true));
        assertFalse(disabled.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(disabled.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(disabled.shouldUseMetalGateUpMatvecFfn());
    }

    @Test
    void matvecFfnExplicitOverridesFamilyDefaults() {
        DirectForwardFfnFastPathRoutingPolicy options = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(true, true, true, false));

        assertTrue(options.shouldUseMetalGegluMatvecFfn(generic()));
        assertTrue(options.shouldUseMetalSwigluMatvecFfn(generic()));
        assertTrue(options.shouldUseMetalGateUpMatvecFfn());

        DirectForwardFfnFastPathRoutingPolicy rejected = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(false, false, false, false));
        assertFalse(rejected.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(rejected.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(rejected.shouldUseMetalGateUpMatvecFfn());
    }

    @Test
    void validationCanBeForcedOrEnabledByTrace() {
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldValidateMetalMatvecFfn(false));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldValidateMetalMatvecFfn(true));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withValidateMetalMatvecFfn(true))
                .shouldValidateMetalMatvecFfn(false));
    }

    private static DirectForwardFfnFastPathRoutingPolicy policy(DirectForwardFfnFastPathOptions options) {
        return DirectForwardFfnFastPathRoutingPolicy.from(options);
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false, false, false, false);
    }

    private static ModelConfigTraits gemma3() {
        return traits("gemma3_text", false, true, false, false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true, false, false, false);
    }

    private static ModelConfigTraits qwen() {
        return traits("qwen2", false, false, true, false);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean gemma4Text,
            boolean gemma3Text,
            boolean qwenText,
            boolean gemma4StylePerLayerInputs) {
        return new ModelConfigTraits(
                null, modelType, 0, 0, gemma4Text, gemma3Text, qwenText, gemma4StylePerLayerInputs);
    }
}
