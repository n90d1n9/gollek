# LiteRT Runtime Release Notes (Template)

## Summary

- LiteRT runtime assets published for: macOS / Linux (arm64, x86_64)
- Compatible with LiteRT/TFLite C API: 2.16+

## Assets

- `litert-runtime-macos-arm64.tar.gz`
- `litert-runtime-macos-x86_64.tar.gz`
- `litert-runtime-linux-x86_64.tar.gz`
- `litert-runtime-linux-arm64.tar.gz`

## Verification

```bash
export GOLLEK_LITERT_RUNTIME_REPO="bhangun/gollek"
export GOLLEK_LITERT_RUNTIME_RELEASE="latest"
export GOLLEK_LITERT_RUNTIME_ASSET="litert-runtime-macos-arm64.tar.gz"
make litert-runtime
```

```bash
export LITERT_LIBRARY_PATH="$HOME/.gollek/libs/libtensorflowlite_c.dylib"
export GOLLEK_TFLITE_MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.litertlm"
mvn -f inference-gollek/extension/runner/litert/gollek-runner-litert/pom.xml clean test
```
