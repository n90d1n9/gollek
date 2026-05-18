package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerModelCheckpointLoaderTest {
    @TempDir
    Path tempDir;

    @Test
    void loadsModelCheckpointUsingProvidedLoader() throws Exception {
        Path checkpoint = tempDir.resolve("model.safetensors");
        byte[] bytes = new byte[] {1, 2, 3, 4};
        Files.write(checkpoint, bytes);
        AtomicReference<byte[]> loadedBytes = new AtomicReference<>();

        TrainerModelCheckpointLoader.LoadResult result = TrainerModelCheckpointLoader.loadModel(
                checkpoint,
                path -> loadedBytes.set(Files.readAllBytes(path)));

        assertTrue(result.loaded());
        assertNull(result.error());
        assertNull(result.cause());
        assertArrayEquals(bytes, loadedBytes.get());
    }

    @Test
    void capturesModelLoadFailureAndCause() {
        Path checkpoint = tempDir.resolve("model.safetensors");
        IOException failure = new IOException("bad checkpoint");

        TrainerModelCheckpointLoader.LoadResult result = TrainerModelCheckpointLoader.loadModel(
                checkpoint,
                path -> {
                    throw failure;
                });

        assertFalse(result.loaded());
        assertEquals("bad checkpoint", result.error());
        assertSame(failure, result.cause());
    }

    @Test
    void skippedWhenTargetIsAbsentAndLoaderIsNotCalled() {
        AtomicBoolean called = new AtomicBoolean(false);

        TrainerModelCheckpointLoader.LoadResult result = TrainerModelCheckpointLoader.loadModel(
                null,
                path -> called.set(true));

        assertFalse(result.loaded());
        assertNull(result.error());
        assertNull(result.cause());
        assertFalse(called.get());
    }
}
