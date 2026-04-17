package tech.kayys.gollek.onnx.loader;

import java.nio.file.Path;

/** Lightweight ONNX parser stub used by ONNXLoader. */
public final class ONNXParser {

    public ONNXModel parse(Path path) {
        // Minimal stub: return an empty model descriptor
        return new ONNXModel(path.getFileName().toString(), java.util.Collections.emptyMap());
    }
}
