package tech.kayys.gollek.onnx.loader;

import java.io.IOException;
import java.nio.file.Path;

/** High-level loader facade for ONNX models (skeleton). */
public final class ONNXLoader {

    public ONNXModel load(Path path) throws IOException {
        // TODO: implement parsing and optional runtime binding (ONNX Runtime JNI)
        throw new UnsupportedOperationException("ONNX loading not implemented");
    }
}
