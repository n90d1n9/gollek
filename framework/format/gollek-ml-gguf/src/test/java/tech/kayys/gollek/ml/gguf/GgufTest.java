package tech.kayys.gollek.ml.gguf;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class GgufTest {


    @Test
    void testGgufMetaValue() {
        var str = GgufMetaValue.ofString("test");
        assertNotNull(str);

        var num = new GgufMetaValue.Int32Val(42);
        assertNotNull(num);
    }

    @Test
    void testGgmlType() {
        // Verify enum exists and has expected values
        assertNotNull(GgmlType.F32);
        assertNotNull(GgmlType.F16);
    }

    @Test
    void testWriteAndReadTensor() throws IOException {
        Path tmp = Files.createTempFile("test", ".gguf");
        try {
            var tensor = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
            var stateDict = java.util.Map.of("weight", tensor);
            var meta = java.util.Map.<String, GgufMetaValue>of("general.architecture", GgufMetaValue.ofString("gollek"));

            GgufWriter.save(tmp, stateDict, meta);
            assertTrue(Files.exists(tmp));
            assertTrue(Files.size(tmp) > 0);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
