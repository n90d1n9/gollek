package tech.kayys.gollek.ml.safetensors;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SafetensorTest {

    @Test
    void testWriteAndRead() throws IOException {
        Path tmp = Files.createTempFile("test", ".safetensors");
        try {
            var tensor = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
            var stateDict = java.util.Map.of("weight", tensor);
            SafetensorWriter.save(tmp, stateDict);

            assertTrue(Files.exists(tmp));
            assertTrue(Files.size(tmp) > 0);

            var loaded = SafetensorReader.read(tmp);
            assertTrue(loaded.containsKey("weight"));
            assertArrayEquals(new float[]{1, 2, 3, 4}, loaded.get("weight"), 1e-5f);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    void testRoundTripLargeTensor() throws IOException {
        Path tmp = Files.createTempFile("test_large", ".safetensors");
        try {
            var tensor = GradTensor.randn(64, 128);
            var stateDict = java.util.Map.of("large_weight", tensor);
            SafetensorWriter.save(tmp, stateDict);

            var loaded = SafetensorReader.read(tmp);
            assertArrayEquals(tensor.data(), loaded.get("large_weight"), 1e-5f);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
