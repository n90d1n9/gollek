package tech.kayys.gollek.ml.gguf;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Temporary Gradle-owned compatibility reader for the ML stack.
 */
public final class GgufReader implements AutoCloseable {

    private final GgufWriter.CompatGgufPayload payload;

    public GgufReader(Path path) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(path))) {
            Object read = in.readObject();
            this.payload = (GgufWriter.CompatGgufPayload) read;
        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to read compatibility GGUF payload", e);
        }
    }

    public Map<String, float[]> loadTensors() {
        return payload.tensors;
    }

    @Override
    public void close() {
    }
}
