# LiteRT Runtime Release Assets

This module ships prebuilt LiteRT/TFLite runtime archives for easy setup.

## Asset Naming

Use this convention so the download helper can auto-detect:

```
litert-runtime-${platform}-${arch}.tar.gz
```

Examples:
- `litert-runtime-macos-arm64.tar.gz`
- `litert-runtime-macos-x86_64.tar.gz`
- `litert-runtime-linux-x86_64.tar.gz`
- `litert-runtime-linux-arm64.tar.gz`

## Contents

Each archive should contain exactly one runtime library at the archive root:

- macOS: `libtensorflowlite_c.dylib`
- Linux: `libtensorflowlite_c.so`

## Release Checklist

- See `docs/RELEASE_CHECKLIST.md` for step-by-step packaging + upload.

## Build & Package (local)

```bash
# Copy the runtime to ~/.gollek/libs first, or pass an explicit path
export LITERT_LIBRARY_PATH="/path/to/libtensorflowlite_c.dylib"
make litert-runtime-package
```

## Publish (GitHub Releases)

Upload the generated archive to your GitHub release:

```
litert-runtime-macos-arm64.tar.gz
```

The downloader will look in:

- repo: `GOLLEK_LITERT_RUNTIME_REPO` (default: `bhangun/gollek`)
- release: `GOLLEK_LITERT_RUNTIME_RELEASE` (default: `latest`)
- asset: `GOLLEK_LITERT_RUNTIME_ASSET` (default: `litert-runtime-${platform}-${arch}.tar.gz`)
