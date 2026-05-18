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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufTensorOpsTest {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void dequantizesQ4KBlockWithCurrentGgmlScaleLayout() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(144);
            writeSimpleQ4KBlock(block);

            float[] out = new float[256];
            GgufTensorOps.dequantizeQ4KBlock(block, 0, out, 0);

            for (int superBlock = 0; superBlock < 4; superBlock++) {
                int base = superBlock * 64;
                for (int i = 0; i < 32; i++) {
                    assertEquals(1.0f, out[base + i], 0.0f);
                    assertEquals(2.0f, out[base + 32 + i], 0.0f);
                }
            }
            assertEquals(384.0f, sum(out), 0.0f);
        }
    }

    @Test
    void computesQ4KRowDotWithoutMaterializingTheWholeMatrix() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(144);
            writeSimpleQ4KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[256];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }

            assertEquals(384.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
        }
    }

    @Test
    void supportsQ4_0RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{32, 2}, 2, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(0.0f, row[i], 0.0f);
                assertEquals(1.0f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(16.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(48.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
        }
    }

    @Test
    void supportsQ4_1RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(20);
            writeQ4_1Block(segment, (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_1", new long[]{32, 1}, 3, 0, 20);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(1.5f, row[i], 0.0f);
                assertEquals(2.5f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(64.0f, output[0], 0.0f);
        }
    }

    @Test
    void supportsQ5_0RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 22);
            writeQ5_0Block(segment.asSlice(0, 22), (short) 0x3c00, -1, (byte) 0x10);
            writeQ5_0Block(segment.asSlice(22, 22), (short) 0x3c00, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_0", new long[]{32, 2}, 6, 0, 2L * 22);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(6));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(0.0f, row[i], 0.0f);
                assertEquals(1.0f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(16.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(48.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
        }
    }

    @Test
    void supportsQ5_1RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(24);
            writeQ5_1Block(segment, (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_1", new long[]{32, 1}, 7, 0, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(7));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(17.5f, row[i], 0.0f);
                assertEquals(18.5f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(576.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(576.0f, output[0], 0.0f);
        }
    }

    @Test
    void preparesAndCachesQ32MatrixForGenericMatVec() {
        String previous = System.getProperty("gollek.gguf.q32.cache_min_rows");
        System.setProperty("gollek.gguf.q32.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{32, 2}, 2, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(32);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(80L, GgufTensorOps.q32MatrixCacheBytes(model));

            GgufTensorOps.Q32Matrix first = GgufTensorOps.q32MatrixCached(model, tensor);
            GgufTensorOps.Q32Matrix second = GgufTensorOps.q32MatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(16.0f, preparedOutput[0], 0.0f);
            assertEquals(48.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ32MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q32.cache_min_rows", previous);
        }
    }

    @Test
    void preparesQ32MatrixForQ5_1HighBitsAndBias() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(24);
            writeQ5_1Block(segment, (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_1", new long[]{32, 1}, 7, 0, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q32Matrix matrix = GgufTensorOps.q32Matrix(model, tensor);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(matrix, ones(32), output, 1, true);

            assertEquals(40L, matrix.estimatedBytes());
            assertEquals(576.0f, output[0], 0.0f);
        }
    }

    @Test
    void boundsPreparedQ32MatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("gollek.gguf.q32.cache_max_bytes");
        System.setProperty("gollek.gguf.q32.cache_max_bytes", "40");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4_0.first", new long[]{32, 1}, 2, 0, 18);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4_0.second", new long[]{32, 1}, 2, 18, 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q32Matrix first = GgufTensorOps.q32MatrixCached(model, firstTensor);
            assertEquals(40L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(40L, GgufTensorOps.q32MatrixCacheBytes(model));

            GgufTensorOps.q32MatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertTrue(GgufTensorOps.q32MatrixCacheBytes(model) <= 40L);

            GgufTensorOps.Q32Matrix firstAfterEviction = GgufTensorOps.q32MatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ32MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q32.cache_max_bytes", previous);
        }
    }

    @Test
    void supportsQ2KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q2k.cache_min_rows");
        System.setProperty("gollek.gguf.q2k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeQ2KBlock(segment.asSlice(84, 84), (byte) 0x01, (byte) 0xAA);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2_k", new long[]{256, 2}, 10, 0, 2L * 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(10));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(512.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(768L, GgufTensorOps.q2KMatrixCacheBytes(model));

            GgufTensorOps.Q2KMatrix first = GgufTensorOps.q2KMatrixCached(model, tensor);
            GgufTensorOps.Q2KMatrix second = GgufTensorOps.q2KMatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ2KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q2k.cache_min_rows", previous);
        }
    }

    @Test
    void supportsQ3KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q3k.cache_min_rows");
        System.setProperty("gollek.gguf.q3k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 110);
            writeQ3KBlock(segment.asSlice(0, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ3KBlock(segment.asSlice(110, 110), 1, (byte) 0xAA, (byte) 0xFF);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3_k", new long[]{256, 2}, 11, 0, 2L * 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(11));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(512.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q3KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q3KMatrixCacheBytes(model));

            GgufTensorOps.Q3KMatrix first = GgufTensorOps.q3KMatrixCached(model, tensor);
            GgufTensorOps.Q3KMatrix second = GgufTensorOps.q3KMatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ3KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q3k.cache_min_rows", previous);
        }
    }

    @Test
    void supportsQ5KRowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(176);
            writeQ5KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_k", new long[]{256, 1}, 13, 0, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(13));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(16.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(4096.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(4096.0f, output[0], 0.0f);
        }
    }

    @Test
    void preparesAndCachesQ5KMatrixForGenericMatVec() {
        String previous = System.getProperty("gollek.gguf.q5k.cache_min_rows");
        System.setProperty("gollek.gguf.q5k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_k", new long[]{256, 2}, 13, 0, 2L * 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(4096.0f, output[0], 0.0f);
            assertEquals(4352.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q5KMatrixCacheBytes(model));

            GgufTensorOps.Q5KMatrix first = GgufTensorOps.q5KMatrixCached(model, tensor);
            GgufTensorOps.Q5KMatrix second = GgufTensorOps.q5KMatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(4096.0f, preparedOutput[0], 0.0f);
            assertEquals(4352.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ5KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q5k.cache_min_rows", previous);
        }
    }

    @Test
    void boundsPreparedQ5KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("gollek.gguf.q5k.cache_max_bytes");
        System.setProperty("gollek.gguf.q5k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q5.first", new long[]{256, 1}, 13, 0, 176);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q5.second", new long[]{256, 1}, 13, 176, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q5KMatrix first = GgufTensorOps.q5KMatrixCached(model, firstTensor);
            assertEquals(320L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q5KMatrixCacheBytes(model));

            GgufTensorOps.q5KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q5KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q5KMatrix firstAfterEviction = GgufTensorOps.q5KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ5KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q5k.cache_max_bytes", previous);
        }
    }

    @Test
    void supportsQ6KRowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(210);
            writeQ6KBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6_k", new long[]{256, 1}, 14, 0, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(14));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(256.0f, output[0], 0.0f);
        }
    }

    @Test
    void preparesAndCachesQ6KMatrixForGenericMatVec() {
        String previous = System.getProperty("gollek.gguf.q6k.cache_min_rows");
        System.setProperty("gollek.gguf.q6k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6_k", new long[]{256, 2}, 14, 0, 2L * 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(256);
            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q6KMatrixCacheBytes(model));

            GgufTensorOps.Q6KMatrix first = GgufTensorOps.q6KMatrixCached(model, tensor);
            GgufTensorOps.Q6KMatrix second = GgufTensorOps.q6KMatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ6KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q6k.cache_min_rows", previous);
        }
    }

    @Test
    void boundsPreparedQ6KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("gollek.gguf.q6k.cache_max_bytes");
        System.setProperty("gollek.gguf.q6k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q6.first", new long[]{256, 1}, 14, 0, 210);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q6.second", new long[]{256, 1}, 14, 210, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q6KMatrix first = GgufTensorOps.q6KMatrixCached(model, firstTensor);
            assertEquals(320L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q6KMatrixCacheBytes(model));

            GgufTensorOps.q6KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q6KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q6KMatrix firstAfterEviction = GgufTensorOps.q6KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ6KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q6k.cache_max_bytes", previous);
        }
    }

    @Test
    void usesGgufShapeZeroAsMatrixColumnsForF32MatVec() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Float.BYTES);
            float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            for (int i = 0; i < values.length; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, values[i]);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo("f32", new long[]{3, 2}, 0, 0, 6L * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVec(model, tensor, new float[]{1.0f, 1.0f, 1.0f}, output);

            assertEquals(6.0f, output[0], 0.0f);
            assertEquals(15.0f, output[1], 0.0f);

            float[] parallelOutput = new float[2];
            GgufTensorOps.matVecParallel(model, tensor, new float[]{1.0f, 1.0f, 1.0f}, parallelOutput);
            assertEquals(6.0f, parallelOutput[0], 0.0f);
            assertEquals(15.0f, parallelOutput[1], 0.0f);
        }
    }

    @Test
    void adaptsParallelMatVecToWorkSizeAndConfiguredChunks() {
        String previousMinOps = System.getProperty("gollek.gguf.parallel_min_ops");
        String previousThreads = System.getProperty("gollek.gguf.parallel_threads");
        String previousChunks = System.getProperty("gollek.gguf.parallel_chunks_per_thread");
        System.setProperty("gollek.gguf.parallel_min_ops", "1024");
        System.setProperty("gollek.gguf.parallel_threads", "3");
        System.setProperty("gollek.gguf.parallel_chunks_per_thread", "2");
        try {
            assertFalse(GgufTensorOps.shouldParallelize(false, 1024, 1024));
            assertFalse(GgufTensorOps.shouldParallelize(true, 1, 1024));
            assertFalse(GgufTensorOps.shouldParallelize(true, 2, 256));
            assertTrue(GgufTensorOps.shouldParallelize(true, 4, 256));
            assertEquals(6, GgufTensorOps.parallelChunkCount(64));
            assertEquals(4, GgufTensorOps.parallelChunkCount(4));
            assertEquals(1, GgufTensorOps.parallelChunkCount(1));
        } finally {
            restoreProperty("gollek.gguf.parallel_min_ops", previousMinOps);
            restoreProperty("gollek.gguf.parallel_threads", previousThreads);
            restoreProperty("gollek.gguf.parallel_chunks_per_thread", previousChunks);
        }
    }

    @Test
    void usesTwoParallelChunksPerThreadByDefault() {
        String previousThreads = System.getProperty("gollek.gguf.parallel_threads");
        String previousChunks = System.getProperty("gollek.gguf.parallel_chunks_per_thread");
        System.setProperty("gollek.gguf.parallel_threads", "3");
        System.clearProperty("gollek.gguf.parallel_chunks_per_thread");
        try {
            assertEquals(6, GgufTensorOps.parallelChunkCount(64));
            assertEquals(4, GgufTensorOps.parallelChunkCount(4));
            assertEquals(1, GgufTensorOps.parallelChunkCount(1));
        } finally {
            restoreProperty("gollek.gguf.parallel_threads", previousThreads);
            restoreProperty("gollek.gguf.parallel_chunks_per_thread", previousChunks);
        }
    }

    @Test
    void computesQ4KMatVecRowsInParallel() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeSimpleQ4KBlock(segment.asSlice(0, 144));
            writeSimpleQ4KBlock(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[256];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }
            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(384.0f, output[0], 0.0f);
            assertEquals(384.0f, output[1], 0.0f);
        }
    }

    @Test
    void preparesAndCachesQ8MatrixForGenericMatVec() {
        String previous = System.getProperty("gollek.gguf.q8.cache_min_rows");
        System.setProperty("gollek.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8", new long[]{32, 2}, 8, 0, 2L * 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[32];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }

            assertEquals(32.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(32.0f, preparedOutput[0], 0.0f);
            assertEquals(64.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void preparedQ8MatrixPreservesSignedQuantValues() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(34);
            writeQ8Block(segment, (short) 0x3c00, (byte) -2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8.signed", new long[]{32, 1}, 8, 0, 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ones(32);
            assertEquals(-64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(GgufTensorOps.q8Matrix(model, tensor), vector, output, 1, true);
            assertEquals(-64.0f, output[0], 0.0f);
        }
    }

    @Test
    void supportsQ8_1RowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q8.cache_min_rows");
        System.setProperty("gollek.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 36);
            writeQ8_1Block(segment.asSlice(0, 36), (short) 0x3c00, (byte) 1);
            writeQ8_1Block(segment.asSlice(36, 36), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8_1", new long[]{32, 2}, 9, 0, 2L * 36);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(9));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(32.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(64.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(32.0f, preparedOutput[0], 0.0f);
            assertEquals(64.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void boundsPreparedQ8MatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("gollek.gguf.q8.cache_max_bytes");
        System.setProperty("gollek.gguf.q8.cache_max_bytes", "36");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q8.first", new long[]{32, 1}, 8, 0, 34);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q8.second", new long[]{32, 1}, 8, 34, 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, firstTensor);
            assertEquals(36L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(36L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.q8MatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertTrue(GgufTensorOps.q8MatrixCacheBytes(model) <= 36L);

            GgufTensorOps.Q8Matrix firstAfterEviction = GgufTensorOps.q8MatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_max_bytes", previous);
        }
    }

    @Test
    void supportsQ8KRowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q8.cache_min_rows");
        System.setProperty("gollek.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 292);
            writeQ8KBlock(segment.asSlice(0, 292), 1.0f, (byte) 1);
            writeQ8KBlock(segment.asSlice(292, 292), 1.0f, (byte) 2);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q8_k", new long[]{256, 2}, 15, 0, 2L * 292);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(15));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(512.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(520L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(512.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void supportsIQ4NLRowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q8.cache_min_rows");
        System.setProperty("gollek.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeIQ4NLBlock(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x88);
            writeIQ4NLBlock(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0x99);
            GGUFTensorInfo tensor = new GGUFTensorInfo("iq4_nl", new long[]{32, 2}, 20, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(20));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(32.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(416.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(32.0f, output[0], 0.0f);
            assertEquals(416.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(32.0f, preparedOutput[0], 0.0f);
            assertEquals(416.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void supportsIQ4XSRowDotAndPreparedMatVec() {
        String previous = System.getProperty("gollek.gguf.q8.cache_min_rows");
        System.setProperty("gollek.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 136);
            writeIQ4XSBlock(segment.asSlice(0, 136), (short) 0x3c00, (byte) 0x88);
            writeIQ4XSBlock(segment.asSlice(136, 136), (short) 0x3c00, (byte) 0x99);
            GGUFTensorInfo tensor = new GGUFTensorInfo("iq4_xs", new long[]{256, 2}, 23, 0, 2L * 136);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(23));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(3328.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(3328.0f, output[1], 0.0f);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(576L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(256.0f, preparedOutput[0], 0.0f);
            assertEquals(3328.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void precomputesQ4KVectorGroupSumsForMatVecMinCorrection() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[256];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }
            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(128.0f, output[1], 0.0f);
            assertEquals(128.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            float[] cachedOutput = new float[2];
            GgufTensorOps.matVecRows(matrix, vector, cachedOutput, 2, true);
            assertEquals(128.0f, cachedOutput[0], 0.0f);
            assertEquals(128.0f, cachedOutput[1], 0.0f);
        }
    }

    @Test
    void genericQ4KMatVecCanUsePreparedCachePath() {
        String previous = System.getProperty("gollek.gguf.q4k.cache_min_rows");
        System.setProperty("gollek.gguf.q4k.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[256];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }
            float[] output = new float[2];

            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(128.0f, output[1], 0.0f);
        } finally {
            if (previous == null) {
                System.clearProperty("gollek.gguf.q4k.cache_min_rows");
            } else {
                System.setProperty("gollek.gguf.q4k.cache_min_rows", previous);
            }
        }
    }

    @Test
    void reusesPreparedQ4KMatrixPerModelAndTensor() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, tensor);
            GgufTensorOps.Q4KMatrix second = GgufTensorOps.q4KMatrixCached(model, tensor);

            assertSame(first, second);
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
            GgufTensorOps.Q4KMatrix afterClear = GgufTensorOps.q4KMatrixCached(model, tensor);
            assertNotSame(first, afterClear);
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
        }
    }

    @Test
    void boundsPreparedQ4KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("gollek.gguf.q4k.cache_max_bytes");
        System.setProperty("gollek.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, firstTensor);
            assertEquals(320L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q4KMatrixCacheBytes(model));

            GgufTensorOps.q4KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q4KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q4KMatrix firstAfterEviction = GgufTensorOps.q4KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
        } finally {
            restoreProperty("gollek.gguf.q4k.cache_max_bytes", previous);
        }
    }

    @Test
    void canDisablePreparedQ4KMatrixCache() {
        String previous = System.getProperty("gollek.gguf.q4k.cache_max_bytes");
        System.setProperty("gollek.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KBlockWithAllScalesAndMins(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, tensor);
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));

            System.setProperty("gollek.gguf.q4k.cache_max_bytes", "0");
            GgufTensorOps.Q4KMatrix second = GgufTensorOps.q4KMatrixCached(model, tensor);

            assertNotSame(first, second);
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(0L, GgufTensorOps.q4KMatrixCacheBytes(model));
        } finally {
            restoreProperty("gollek.gguf.q4k.cache_max_bytes", previous);
        }
    }

    @Test
    void supportsReusableQ4KWorkBufferForPreparedMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 2}, 12, 0, 2L * 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrix(model, tensor);
            GgufTensorOps.Q4KWorkBuffer workBuffer = new GgufTensorOps.Q4KWorkBuffer();

            float[] vector = new float[256];
            for (int i = 0; i < vector.length; i++) {
                vector[i] = 1.0f;
            }
            float[] output = new float[2];

            GgufTensorOps.matVecRows(matrix, vector, output, 2, true, workBuffer);
            int firstCapacity = workBuffer.vectorGroupSumCapacity();
            GgufTensorOps.matVecRows(matrix, vector, output, 2, true, workBuffer);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(128.0f, output[1], 0.0f);
            assertTrue(firstCapacity >= 8);
            assertEquals(firstCapacity, workBuffer.vectorGroupSumCapacity());
            workBuffer.clear();
            assertEquals(0, workBuffer.vectorGroupSumCapacity());
        }
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    private static void writeSimpleQ4KBlock(MemorySegment block) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        block.set(LE_SHORT, 2, (short) 0);

        byte[] scales = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x21);
        }
    }

    private static void writeQ4_0Block(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedQuant);
        }
    }

    private static void writeQ4_1Block(MemorySegment block, short scale, short min, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, packedQuant);
        }
    }

    private static void writeQ5_0Block(MemorySegment block, short scale, int highBits, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_INT, 2, highBits);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 6 + i, packedQuant);
        }
    }

    private static void writeQ5_1Block(MemorySegment block, short scale, short min, int highBits, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, min);
        block.set(LE_INT, 4, highBits);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 8 + i, packedQuant);
        }
    }

    private static void writeQ2KBlock(MemorySegment block, byte scale, byte packedQuant) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, scale);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, packedQuant);
        }
        block.set(LE_SHORT, 80, (short) 0x3c00);
        block.set(LE_SHORT, 82, (short) 0);
    }

    private static void writeQ3KBlock(MemorySegment block, int signedScale, byte packedQuant, byte highMask) {
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, highMask);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 32 + i, packedQuant);
        }
        writeQ3KScales(block, signedScale);
        block.set(LE_SHORT, 108, (short) 0x3c00);
    }

    private static void writeQ3KScales(MemorySegment block, int signedScale) {
        int encoded = Math.max(0, Math.min(63, signedScale + 32));
        for (int group = 0; group < 16; group++) {
            int low = encoded & 0x0F;
            int high = (encoded >>> 4) & 0x03;
            long lowOffset = 96L + (group < 8 ? group : group - 8);
            int lowByte = block.get(ValueLayout.JAVA_BYTE, lowOffset) & 0xFF;
            if (group < 8) {
                lowByte = (lowByte & 0xF0) | low;
            } else {
                lowByte = (lowByte & 0x0F) | (low << 4);
            }
            block.set(ValueLayout.JAVA_BYTE, lowOffset, (byte) lowByte);

            long highOffset = 96L + 8 + (group % 4);
            int highByte = block.get(ValueLayout.JAVA_BYTE, highOffset) & 0xFF;
            highByte |= high << (2 * (group / 4));
            block.set(ValueLayout.JAVA_BYTE, highOffset, (byte) highByte);
        }
    }

    private static void writeQ4KBlockWithAllScalesAndMins(MemorySegment block) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        block.set(LE_SHORT, 2, (short) 0x3800);

        byte[] scales = {1, 1, 1, 1, 2, 2, 2, 2, 0x21, 0x21, 0x21, 0x21};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, (byte) 0x21);
        }
    }

    private static void writeQ5KBlock(MemorySegment block) {
        writeQ5KBlock(block, (byte) 0xFF, (byte) 0);
    }

    private static void writeQ5KBlock(MemorySegment block, byte highBits, byte packedQuant) {
        block.set(LE_SHORT, 0, (short) 0x3c00);
        block.set(LE_SHORT, 2, (short) 0);

        byte[] scales = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
        for (int i = 0; i < scales.length; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, scales[i]);
        }
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 16 + i, highBits);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 48 + i, packedQuant);
        }
    }

    private static void writeQ6KBlock(MemorySegment block) {
        writeQ6KBlock(block, (byte) 0x11, (byte) 0xAA, (byte) 1);
    }

    private static void writeQ6KBlock(MemorySegment block, byte lowPacked, byte highPacked, byte scale) {
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, i, lowPacked);
        }
        for (int i = 0; i < 64; i++) {
            block.set(ValueLayout.JAVA_BYTE, 128 + i, highPacked);
        }
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 192 + i, scale);
        }
        block.set(LE_SHORT, 208, (short) 0x3c00);
    }

    private static void writeQ8Block(MemorySegment block, short scale, byte quant) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, quant);
        }
    }

    private static void writeQ8_1Block(MemorySegment block, short scale, byte quant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, (short) 0x7b00);
        for (int i = 0; i < 32; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, quant);
        }
    }

    private static void writeQ8KBlock(MemorySegment block, float scale, byte quant) {
        block.set(LE_FLOAT, 0, scale);
        for (int i = 0; i < 256; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, quant);
        }
        short groupSum = (short) (quant * 16);
        for (int group = 0; group < 16; group++) {
            block.set(LE_SHORT, 260 + group * 2L, groupSum);
        }
    }

    private static void writeIQ4NLBlock(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, 2 + i, packedQuant);
        }
    }

    private static void writeIQ4XSBlock(MemorySegment block, short scale, byte packedQuant) {
        block.set(LE_SHORT, 0, scale);
        block.set(LE_SHORT, 2, (short) 0xAAAA);
        for (int i = 0; i < 4; i++) {
            block.set(ValueLayout.JAVA_BYTE, 4 + i, (byte) 0x11);
        }
        for (int i = 0; i < 128; i++) {
            block.set(ValueLayout.JAVA_BYTE, 8 + i, packedQuant);
        }
    }

    private static float[] ones(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0f;
        }
        return values;
    }

    private static float sum(float[] values) {
        float total = 0.0f;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}
