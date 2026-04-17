# gollek-native-tensorflow

Purpose

Comprehensive TensorFlow support: load frozen .pb GraphDef, SavedModel directories, and convert/execute models. Provide parsing, extraction of signatures, TFLite conversion, and optional pure-Java inference engine or conversion to GGUF for native inference.

Highlights

- TFGraphParser: lightweight, zero-copy protobuf parser for GraphDef (varint support, field tags).
- TFSavedModelParser: extracts signatures, variables, and assets from SavedModel layout.
- TFModelLoader: high-level API to load models from path or byte arrays and produce TFModel records.
- TFToTFLiteConverter: conversion utilities to build TFLite flatbuffers with quantization options.
- TFInferenceEngine: pure-Java executor for common ops (builder pattern, execution plan, multithreaded stages).

Improvements implemented

- Signature extraction and SavedModel metadata parsing for easier integration with serving code.
- Utilities for graph pruning, constant folding, and extraction of an inference subgraph.
- Placeholder conversion path TF→GGUF for models where conversion is feasible.

Next steps

1. Harden TF protobuf parsing with real GraphDef samples and tests.
2. Implement conversion of common TF ops to GGUF-compatible tensors/layout.
3. Add integration tests for SavedModel and a demo inference flow.

