/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionProjectionStageTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void gemma4AlternativeAttentionDerivesValueFromKeyWhenVProjectionIsAbsent() throws Exception {
        ModelConfig config = gemma4AlternativeAttentionConfig();
        try (AccelTensor x = AccelTensor.fromFloatArray(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        }, 1, 1, 8);
                AccelTensor qW = AccelTensor.zeros(8, 8);
                AccelTensor kW = firstFourInputColumnsWeight()) {
            AttentionInput input = new AttentionInput(
                    x, qW, kW, null, null,
                    null, null, null, null,
                    null, config, null, 1, 0, true,
                    null, null, null, null);
            FlashAttentionPlan plan = FlashAttentionPlan.resolve(
                    input,
                    new FlashAttentionKvCacheStage(),
                    new FlashAttentionNormalizationOptions(true, true));

            assertTrue(plan.alternativeAttention());
            assertEquals(1, plan.numKeyValueHeads());
            assertEquals(4, plan.headDim());

            FlashAttentionProjectionStage stage = new FlashAttentionProjectionStage(
                    new FlashAttentionProjector(null, () -> false),
                    new FlashAttentionNormalizer(() -> null));
            FlashAttentionProjectionStage.PreparedTensors prepared = stage.prepare(
                    input,
                    config,
                    plan.modelPolicy(),
                    plan.headLayout(),
                    plan.normalizationPolicy(),
                    plan.sharedKvState(),
                    plan.sharedKv(),
                    plan.useDenseSharedKvState(),
                    plan.alternativeAttention());

            try {
                assertTrue(prepared.key().hasShape(new long[] { 1, 1, 1, 4 }));
                assertTrue(prepared.value().hasShape(new long[] { 1, 1, 1, 4 }));
                assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f, 4.0f },
                        prepared.key().toFloatArray(), 0.0001f);
                assertArrayEquals(prepared.key().toFloatArray(), prepared.value().toFloatArray(), 0.0001f);
            } finally {
                prepared.query().close();
                prepared.key().close();
                prepared.value().close();
            }
        }
    }

    private static ModelConfig gemma4AlternativeAttentionConfig() throws Exception {
        return OBJECT_MAPPER.readValue("""
                {
                  "model_type": "gemma4_text",
                  "architectures": ["Gemma4ForCausalLM"],
                  "hidden_size": 8,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "num_global_key_value_heads": 1,
                  "head_dim": 2,
                  "global_head_dim": 4,
                  "attention_k_eq_v": true,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
    }

    private static AccelTensor firstFourInputColumnsWeight() {
        float[] weights = new float[4 * 8];
        for (int row = 0; row < 4; row++) {
            weights[row * 8 + row] = 1.0f;
        }
        return AccelTensor.fromFloatArray(weights, 4, 8);
    }
}
