/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class PagedKvCacheMaterializerTest {

    @Test
    void packsDenseSharedKvIntoHeadMajorPagedPool() {
        try (AccelTensor key = tensor(1.0f);
                AccelTensor value = tensor(101.0f);
                Arena arena = Arena.ofConfined()) {
            MemorySegment packedKey = arena.allocate(8L * Float.BYTES, 64);
            MemorySegment packedValue = arena.allocate(8L * Float.BYTES, 64);

            PagedKvCacheMaterializer.packDenseSharedKvIntoTemporaryPagedPool(
                    key, value, 2, 2, packedKey, packedValue);

            assertArrayEquals(new float[] { 1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f },
                    toFloats(packedKey, 8), 0.0001f);
            assertArrayEquals(new float[] { 101.0f, 102.0f, 105.0f, 106.0f, 103.0f, 104.0f, 107.0f, 108.0f },
                    toFloats(packedValue, 8), 0.0001f);
        }
    }

    private static AccelTensor tensor(float start) {
        AccelTensor tensor = AccelTensor.zeros(1, 2, 2, 2);
        for (int i = 0; i < 8; i++) {
            tensor.setFlat(i, start + i);
        }
        return tensor;
    }

    private static float[] toFloats(MemorySegment segment, int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
            values[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return values;
    }
}
