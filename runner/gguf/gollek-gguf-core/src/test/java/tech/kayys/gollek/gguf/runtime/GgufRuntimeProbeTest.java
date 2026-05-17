package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRuntimeProbeTest {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void samplesPreferredTensorWithJavaRowDotPrimitive() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Float.BYTES);
            float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            for (int i = 0; i < values.length; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, values[i]);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "blk.0.attn_q.weight",
                    new long[]{3, 2},
                    0,
                    0,
                    6L * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 24, 2, 2);

            assertTrue(probe.hasTensorProbe());
            assertEquals("blk.0.attn_q.weight", probe.tensorName());
            assertEquals("F32", probe.tensorType());
            assertEquals(2, probe.rows());
            assertEquals(3, probe.columns());
            assertEquals(2, probe.sampledRows());
            assertEquals(-8.411765f, probe.rowDotChecksum(), 0.00001f);
            assertTrue(probe.rowDotNanos() >= 0);
            assertEquals(2, probe.matVecRows());
            assertEquals(-8.411765f, probe.matVecChecksum(), 0.00001f);
            assertTrue(probe.matrixCacheNanos() >= 0);
            assertTrue(probe.matVecNanos() >= 0);
            assertTrue(probe.cachedMatVecNanos() >= 0);
        }
    }

    @Test
    void probesPreparedQ32MatrixPathForQ4_0Tensor() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "blk.0.attn_q.weight",
                    new long[]{32, 2},
                    2,
                    0,
                    2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 36, 2, 2);

            assertTrue(probe.hasTensorProbe());
            assertEquals("Q4_0", probe.tensorType());
            assertEquals(2, probe.rows());
            assertEquals(32, probe.columns());
            assertEquals(-1.7058823f, probe.rowDotChecksum(), 0.00001f);
            assertEquals(-1.7058823f, probe.matVecChecksum(), 0.00001f);
            assertEquals(probe.matVecChecksum(), probe.cachedMatVecChecksum(), 0.00001f);
            assertTrue(probe.matrixCacheNanos() >= 0);
            assertTrue(probe.cachedMatVecNanos() >= 0);
        }
    }

    @Test
    void probesPreparedQ8MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(34);
            writeQ8Block(segment, (byte) -2);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "blk.0.attn_q.weight",
                    new long[]{32, 1},
                    8,
                    0,
                    34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 34, 1, 1);

            assertTrue(probe.hasTensorProbe());
            assertEquals("Q8_0", probe.tensorType());
            assertEquals(1, probe.rows());
            assertEquals(32, probe.columns());
            assertEquals(1.7647059f, probe.rowDotChecksum(), 0.00001f);
            assertEquals(1.7647059f, probe.matVecChecksum(), 0.00001f);
            assertEquals(probe.matVecChecksum(), probe.cachedMatVecChecksum(), 0.00001f);
            assertTrue(probe.matrixCacheNanos() >= 0);
            assertTrue(probe.cachedMatVecNanos() >= 0);
        }
    }

    private static void writeQ4_0Block(MemorySegment block, byte packedQuant) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedQuant);
        }
    }

    private static void writeQ8Block(MemorySegment block, byte quant) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, quant);
        }
    }
}
