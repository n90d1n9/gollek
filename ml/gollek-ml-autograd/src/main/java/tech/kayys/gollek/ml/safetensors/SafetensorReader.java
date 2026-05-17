package tech.kayys.gollek.ml.safetensors;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Temporary Gradle-owned compatibility reader.
 *
 * <p>Today this restores the ML API contract for Gradle builds. It does not
 * aim to be a full universal SafeTensor parser yet.</p>
 */
public final class SafetensorReader {

    private SafetensorReader() {
    }

    @SuppressWarnings("unchecked")
    public static Map<String, float[]> read(Path path) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(path))) {
            Object payload = in.readObject();
            return (Map<String, float[]>) payload;
        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to read compatibility safetensor payload", e);
        }
    }
}
