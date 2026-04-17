package tech.kayys.gollek.tensorflow.loader;

import java.nio.file.Path;

/** TF graph parser stub (skeleton). */
public final class TFGraphParser {

    public TFModel parseFrozenPb(Path pbPath) {
        // Minimal stub: return an empty TFModel descriptor
        return new TFModel(pbPath.getFileName().toString(), java.util.Collections.emptyMap(), java.util.List.of(), java.util.List.of(), java.util.Collections.emptyMap(), null);
    }
}
