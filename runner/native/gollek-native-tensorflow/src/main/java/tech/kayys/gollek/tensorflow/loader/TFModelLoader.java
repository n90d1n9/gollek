package tech.kayys.gollek.tensorflow.loader;

import java.io.IOException;
import java.nio.file.Path;

/** High-level TFModel loader (skeleton). */
public final class TFModelLoader implements AutoCloseable {

    public TFModel loadModel(Path path) throws IOException {
        // TODO: support Frozen .pb and SavedModel directories
        throw new UnsupportedOperationException("TF model loading not implemented");
    }

    @Override
    public void close() {
        // cleanup
    }
}
