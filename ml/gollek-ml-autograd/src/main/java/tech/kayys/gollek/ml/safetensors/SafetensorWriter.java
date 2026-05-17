package tech.kayys.gollek.ml.safetensors;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Temporary Gradle-owned compatibility writer.
 *
 * <p>This preserves the old ML-side API surface while the maintained
 * safetensor runtime continues to live elsewhere in the repository.</p>
 */
public final class SafetensorWriter {

    private SafetensorWriter() {
    }

    public static void save(Path path, Map<String, GradTensor> tensors) throws IOException {
        Map<String, float[]> state = new LinkedHashMap<>();
        for (Map.Entry<String, GradTensor> entry : tensors.entrySet()) {
            state.put(entry.getKey(), entry.getValue().data().clone());
        }
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(path))) {
            out.writeObject(state);
        }
    }
}
