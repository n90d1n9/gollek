package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.loader.GGUFModel;

import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static tech.kayys.gollek.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.gollek.gguf.runtime.GgufProbeFx.decoderCacheModel;

class GgufProbeSmallTest {
    @Test
    void autoSkipsDecoderCachesWhenPreparedMatricesCannotFitConfiguredCaches() {
        String previousQ32MaxBytes = System.getProperty("gollek.gguf.q32.cache_max_bytes");
        String previousQ8MaxBytes = System.getProperty("gollek.gguf.q8.cache_max_bytes");
        System.setProperty("gollek.gguf.q32.cache_max_bytes", "1");
        System.setProperty("gollek.gguf.q8.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GgufRuntimeProbe.PreparedMatrixCacheSelection selection =
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, 0, true, 1, 1024);

            assertEquals("auto-skipped-cache-too-small", selection.mode());
            assertFalse(selection.prepare());
            assertEquals(0, selection.plan().preparedCandidates());
            assertEquals(2, selection.plan().skippedCacheTooSmallTensors());
        } finally {
            restoreProperty("gollek.gguf.q32.cache_max_bytes", previousQ32MaxBytes);
            restoreProperty("gollek.gguf.q8.cache_max_bytes", previousQ8MaxBytes);
        }
    }
}
