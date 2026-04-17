# gollek-native-onnx

Purpose

Provide robust ONNX model support for Gollek: loading, validation, conversion to GGUF, and runtime execution. Support both single-file .onnx models and sharded/packaged artifacts.

Key features

- ONNX loader: inspect graph, types, ops, and initializers (zero-copy where possible).
- Runtimes: optional bindings for ONNX Runtime (JNI) and a lightweight pure-Java executor for common ops.
- Conversion: utilities to convert ONNX weights to GGUF for the Gollek native inference engine.
- Quantization and optimization pipeline (static/dynamic quantization, operator fusion).

Implementation notes

- Prefer using the official ONNX protobuf schema for parsing. For runtime, prefer ONNX Runtime JNI when available; fall back to a Java executor for small models or CI tests.
- Provide a SafeTensors/NPY intermediate stage when exporting weights for GGUF.
- Add unit tests that validate graph parsing, initializer extraction, and a small end-to-end conversion using a toy model.

Next steps

1. Implement ONNX parser module with Graph/Node/Initializer abstractions.
2. Add converter ONNX→GGUF, reusing SafeTensors converter code where useful.
3. Add optional onnxruntime binding and tests.

