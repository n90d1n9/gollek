package tech.kayys.gollek.ml.gguf;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Temporary Gradle-owned compatibility writer for the ML stack.
 */
public final class GgufWriter {

    private GgufWriter() {
    }

    public static void save(Path path, Map<String, GradTensor> tensors, Map<String, GgufMetaValue> metadata)
            throws IOException {
        CompatGgufPayload payload = new CompatGgufPayload();
        for (Map.Entry<String, GradTensor> entry : tensors.entrySet()) {
            payload.tensors.put(entry.getKey(), entry.getValue().data().clone());
        }
        payload.metadata.putAll(metadata);
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(payload);
        }
    }

    static final class CompatGgufPayload implements java.io.Serializable {
        @java.io.Serial
        private static final long serialVersionUID = 1L;

        final Map<String, float[]> tensors = new LinkedHashMap<>();
        final Map<String, GgufMetaValue> metadata = new LinkedHashMap<>();
    }
}
