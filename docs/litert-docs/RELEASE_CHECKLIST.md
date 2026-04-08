# LiteRT Runtime Release Checklist

Use this checklist when publishing LiteRT runtime assets.

## 1. Build or locate the runtime library

- macOS: `libtensorflowlite_c.dylib`
- Linux: `libtensorflowlite_c.so`

## 2. Package the runtime

```bash
export LITERT_LIBRARY_PATH="/path/to/libtensorflowlite_c.dylib"
make litert-runtime-package
```

Output (default):

```
dist/litert-runtime-${platform}-${arch}.tar.gz
```

## 2.1 Draft release notes

- Use `docs/RELEASE_NOTES_TEMPLATE.md` as a starting point.

## 3. Upload to GitHub Releases

Attach the archive(s) to the release.

Recommended naming:
- `litert-runtime-macos-arm64.tar.gz`
- `litert-runtime-macos-x86_64.tar.gz`
- `litert-runtime-linux-x86_64.tar.gz`
- `litert-runtime-linux-arm64.tar.gz`

### Optional: GitHub Actions Upload

Use the `LiteRT Runtime Release` workflow and point it at your local `dist/` folder.

### Optional: GitHub Actions Build + Release

Use the `LiteRT Runtime Build + Release` workflow if you want CI to fetch the
runtime binary and package/upload it (requires `source_url_macos` / `source_url_linux`).

## 4. Verify downloader

```bash
export GOLLEK_LITERT_RUNTIME_REPO="bhangun/gollek"
export GOLLEK_LITERT_RUNTIME_RELEASE="latest"
export GOLLEK_LITERT_RUNTIME_ASSET="litert-runtime-macos-arm64.tar.gz"
make litert-runtime
```

## 5. Smoke test

```bash
export LITERT_LIBRARY_PATH="$HOME/.gollek/libs/libtensorflowlite_c.dylib"
export GOLLEK_TFLITE_MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.litertlm"
mvn -f inference-gollek/extension/runner/litert/gollek-runner-litert/pom.xml clean test
```
