package tech.kayys.gollek.onnx.loader;

import java.util.Map;

/** Minimal ONNX model descriptor for module skeleton. */
public record ONNXModel(String name, Map<String,Object> metadata) {
}
