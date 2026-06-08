/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationGreedyArgmaxTest {

    @Test
    void selectJavaArrayReturnsHighestFiniteLogit() {
        float[] logits = { 0.25f, Float.NaN, 1.5f, -2.0f };

        assertEquals(2, GenerationGreedyArgmax.selectJava(logits));
    }

    @Test
    void selectJavaArraySkipsRejectedCandidates() {
        float[] logits = { 0.25f, 4.0f, 3.5f, 1.0f };

        assertEquals(2, GenerationGreedyArgmax.selectJava(logits, logits.length, new int[] { 1 }, 1));
    }

    @Test
    void selectJavaArrayReturnsMinusOneWhenEveryCandidateIsRejectedOrNan() {
        float[] logits = { Float.NaN, 4.0f, 3.5f };

        assertEquals(-1, GenerationGreedyArgmax.selectJava(logits, logits.length, new int[] { 1, 2 }, 2));
    }

    @Test
    void selectJavaReturnsHighestFiniteLogit() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, Float.NaN, 1.5f, -2.0f);

            assertEquals(2, GenerationGreedyArgmax.selectJava(logits, 4, null, 0));
        }
    }

    @Test
    void selectJavaSkipsRejectedCandidates() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, 4.0f, 3.5f, 1.0f);

            assertEquals(2, GenerationGreedyArgmax.selectJava(logits, 4, new int[] { 1 }, 1));
        }
    }

    @Test
    void selectJavaReturnsMinusOneWhenEveryCandidateIsRejectedOrNan() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, Float.NaN, 4.0f, 3.5f);

            assertEquals(-1, GenerationGreedyArgmax.selectJava(logits, 3, new int[] { 1, 2 }, 2));
        }
    }

    private static MemorySegment logits(Arena arena, float... values) {
        MemorySegment segment = arena.allocate(ValueLayout.JAVA_FLOAT.byteSize() * values.length);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }
        return segment;
    }
}
