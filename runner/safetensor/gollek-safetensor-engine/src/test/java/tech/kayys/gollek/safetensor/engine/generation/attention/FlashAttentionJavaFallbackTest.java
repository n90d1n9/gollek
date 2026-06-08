/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class FlashAttentionJavaFallbackTest {

    @Test
    void denseSharedAttentionMixesValuesWithOnlineSoftmax() {
        try (AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] {
                        10.0f, 0.0f,
                        0.0f, 20.0f
                }, 1, 2, 1, 2);
                AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                        query, key, value, null, 0, 0, 1, 1, 2, 1.0f, false, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] { 10.0f * expOne * invSum, 20.0f * invSum },
                    output.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void denseSharedAttentionMasksFutureKeysWhenCausal() {
        try (AccelTensor query = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] {
                        10.0f, 0.0f,
                        0.0f, 20.0f
                }, 1, 2, 1, 2);
                AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                        query, key, value, null, 0, 0, 1, 1, 2, 1.0f, true, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] {
                    10.0f, 0.0f,
                    10.0f * invSum, 20.0f * expOne * invSum
            }, output.toFloatArray(), 0.0001f);
        }
    }
}
