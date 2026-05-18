package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class TrainerCheckpointIOTest {

    @Test
    void writeStringAtomicallyCreatesParentAndRemovesTempFile() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-io").resolve("nested/state.txt");

        TrainerCheckpointIO.writeStringAtomically(target, "checkpoint-ready");

        assertEquals("checkpoint-ready", Files.readString(target, StandardCharsets.UTF_8));
        assertFalse(Files.exists(tempPath(target)));
    }

    @Test
    void writeMapAtomicallyRoundTripsStringKeyedState() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-map").resolve("optimizer.state");
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("step", 7);
        state.put("class", "SGD");
        state.put("enabled", Boolean.TRUE);

        TrainerCheckpointIO.writeMapAtomically(target, state);

        assertEquals(state, TrainerCheckpointIO.readMap(target, "optimizer"));
        assertFalse(Files.exists(tempPath(target)));
    }

    @Test
    void writePropertiesAtomicallyPersistsLoadableProperties() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-properties").resolve("manifest.properties");
        Properties properties = new Properties();
        properties.setProperty("formatVersion", "1");
        properties.setProperty("artifact.model.bytes", "42");

        TrainerCheckpointIO.writePropertiesAtomically(target, properties, "test manifest");

        Properties loaded = new Properties();
        try (var reader = Files.newBufferedReader(target, StandardCharsets.UTF_8)) {
            loaded.load(reader);
        }
        assertEquals("1", loaded.getProperty("formatVersion"));
        assertEquals("42", loaded.getProperty("artifact.model.bytes"));
        assertFalse(Files.exists(tempPath(target)));
    }

    @Test
    void readMapReportsArtifactNameForNonMapPayload() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-invalid-payload").resolve("scheduler.state");
        writeObject(target, List.of("not", "a", "map"));

        IOException error = assertThrows(
                IOException.class,
                () -> TrainerCheckpointIO.readMap(target, "scheduler"));

        assertTrue(error.getMessage().contains("Invalid scheduler checkpoint payload"));
        assertTrue(error.getMessage().contains("expected Map"));
    }

    @Test
    void readMapReportsArtifactNameForNonStringKeys() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-invalid-key").resolve("grad-scaler.state");
        writeObject(target, Map.of(1, "bad-key"));

        IOException error = assertThrows(
                IOException.class,
                () -> TrainerCheckpointIO.readMap(target, "GradScaler"));

        assertTrue(error.getMessage().contains("Invalid GradScaler checkpoint payload"));
        assertTrue(error.getMessage().contains("non-string key"));
    }

    @Test
    void writeAtomicallyDeletesTempFileAfterWriterFailure() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-failure").resolve("state.txt");

        IOException error = assertThrows(IOException.class, () -> TrainerCheckpointIO.writeAtomically(target, temp -> {
            Files.writeString(temp, "partial", StandardCharsets.UTF_8);
            throw new IOException("simulated failure");
        }));

        assertTrue(error.getMessage().contains("simulated failure"));
        assertFalse(Files.exists(target));
        assertFalse(Files.exists(tempPath(target)));
    }

    @Test
    void sha256HexMatchesKnownDigest() throws Exception {
        Path target = Files.createTempDirectory("gollek-checkpoint-sha").resolve("payload.bin");
        Files.writeString(target, "abc", StandardCharsets.UTF_8);

        assertEquals(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                TrainerCheckpointIO.sha256Hex(target));
    }

    private static Path tempPath(Path target) {
        return target.resolveSibling(target.getFileName() + ".tmp");
    }

    private static void writeObject(Path target, Object payload) throws IOException {
        try (ObjectOutputStream output = new ObjectOutputStream(Files.newOutputStream(target))) {
            output.writeObject(payload);
        }
    }
}
