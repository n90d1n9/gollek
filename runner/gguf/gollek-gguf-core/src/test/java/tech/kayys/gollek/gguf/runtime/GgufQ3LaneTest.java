package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.gollek.gguf.runtime.GgufFx.*;
import static tech.kayys.gollek.gguf.runtime.GgufKFx.*;

class GgufQ3LaneTest {
    @Test
    void rawQ3KMatVecPreservesPackedQuantAndHighMaskLaneOrder() {
        String previousMinRows = System.getProperty("gollek.gguf.q3k.cache_min_rows");
        String previousMaxBytes = System.getProperty("gollek.gguf.q3k.cache_max_bytes");
        System.setProperty("gollek.gguf.q3k.cache_min_rows", "1");
        System.setProperty("gollek.gguf.q3k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(110);
            writeQ3KLaneOrderBlock(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3.raw.lanes", new long[]{256, 1}, 11, 0, 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

            float expected = expectedQ3LaneOrderDot(vector);
            assertEquals(expected, output[0], 0.0f);
            assertArrayEquals(expectedQ3LaneOrderRow(), dequantizedRow(model, tensor), 0.0f);
            assertEquals(0, GgufTensorOps.q3KMatrixCacheSize(model));
        } finally {
            restoreProperty("gollek.gguf.q3k.cache_min_rows", previousMinRows);
            restoreProperty("gollek.gguf.q3k.cache_max_bytes", previousMaxBytes);
        }
    }
}
