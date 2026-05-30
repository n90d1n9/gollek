package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.gollek.gguf.runtime.GgufProbeFx.assertPreparedMatrixProbe;
import static tech.kayys.gollek.gguf.runtime.GgufProbeFx.oneTensorModel;
import static tech.kayys.gollek.gguf.runtime.GgufQFx.writeQ1_0Block;

class GgufProbeTinyTest {
    @Test
    void probesPreparedQ1_0MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(18);
            writeQ1_0Block(segment, (byte) 0xFF);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{128, 1}, 41, 18),
                    18,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "Q1_0", 1, 128, -2.1176472f);
        }
    }
}
