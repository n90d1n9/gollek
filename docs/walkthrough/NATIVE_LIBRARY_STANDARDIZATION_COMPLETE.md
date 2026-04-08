# ✅ Native Library Standardization - COMPLETE

## Summary

Successfully configured Gollek to use **`~/.gollek/libs/`** as the standard location for all native inference libraries, eliminating the need to track large binary files in git.

## What Changed

### 1. Standard Library Location

**Before**: Native libraries scattered across:
- `plugins/runner/gguf/.../native-libs/`
- `plugins/runner/onnx/.../build/`
- `extension/kernel/libtorch/build/lib/`
- Various vendor directories

**After**: Centralized at `~/.gollek/libs/`:
```
~/.gollek/libs/
├── llama/           # GGUF/llama.cpp
├── onnxruntime/     # ONNX Runtime
├── libtorch/        # PyTorch/LibTorch
└── litert/          # TensorFlow Lite
```

### 2. Code Updates

#### Created Files
- ✅ `NativeLibraryManager.java` - Centralized library management utility
- ✅ `Makefile.native` - Easy installation via `make install-native-libs`
- ✅ `docs/NATIVE_LIBRARY_GUIDE.md` - Comprehensive documentation

#### Modified Files
- ✅ `LlamaCppBinding.java` - Search `~/.gollek/libs/llama/` first
- ✅ `OnnxRuntimeRunner.java` - Search `~/.gollek/libs/onnxruntime/` first
- ✅ `NativeLibraryLoader.java` (LibTorch) - Search `~/.gollek/libs/libtorch/` first
- ✅ `.gitignore` - Ignore `~/.gollek/libs/` and variants

### 3. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GOLLEK_NATIVE_LIB_DIR` | Override base directory | `~/.gollek/libs/` |
| `GOLLEK_LLAMA_LIB_PATH` | Explicit llama.cpp library path | - |
| `GOLLEK_LLAMA_LIB_DIR` | Explicit llama.cpp directory | `~/.gollek/libs/llama/` |
| `GOLLEK_ONNX_LIB_PATH` | Explicit ONNX Runtime library path | - |
| `GOLLEK_ONNX_LIB_DIR` | Explicit ONNX Runtime directory | `~/.gollek/libs/onnxruntime/` |

### 4. Loading Priority

All runners now use this priority order:

1. **Explicit configuration** (config file or command-line)
2. **Environment variables** (`GOLLEK_*_LIB_PATH`)
3. **Standard location** (`~/.gollek/libs/<runner>/`)
4. **Legacy locations** (source vendor, native-libs)
5. **System library path**
6. **Build directories**

## Quick Start

### Install Libraries

```bash
# Install all native libraries
make -f Makefile.native install-native-libs

# Install specific runners
make -f Makefile.native install-gguf-libs
make -f Makefile.native install-onnx-libs
make -f Makefile.native install-libtorch-libs
```

### Verify Installation

```bash
# Check installed libraries
ls -lh ~/.gollek/libs/*/

# Expected output:
# ~/.gollek/libs/llama/:
#   libllama.dylib
#   libggml.dylib
#   ...
#
# ~/.gollek/libs/onnxruntime/:
#   libonnxruntime.dylib
```

### Run Inference

```bash
# Libraries will be automatically loaded from ~/.gollek/libs/
java -jar gollek.jar chat --model <model> --prompt "Hello"
```

## Git Repository Impact

### Before
- 33M .git directory
- Large binary files tracked in history
- Multiple `.dylib`, `.so` files in commits

### After
- 23M .git directory (**30% reduction**)
- No binary files tracked
- Clean `.gitignore` prevents future accidents

## Migration Guide

### For Existing Users

If you have libraries in old locations:

```bash
# Option 1: Use Makefile to copy
make -f Makefile.native install-native-libs

# Option 2: Manual migration
mkdir -p ~/.gollek/libs/llama
cp plugins/runner/gguf/.../native-libs/cpu/*.dylib ~/.gollek/libs/llama/

mkdir -p ~/.gollek/libs/onnxruntime
cp plugins/runner/onnx/.../build/.../lib/*.dylib ~/.gollek/libs/onnxruntime/

# Set permissions
chmod +x ~/.gollek/libs/*/*.dylib

# Clear macOS quarantine
xattr -dr com.apple.quarantine ~/.gollek/libs/
```

### For New Users

1. Clone the repository
2. Download or build native libraries
3. Install to standard location:
   ```bash
   make -f Makefile.native install-native-libs
   ```
4. Start using Gollek!

## Documentation

- **[NATIVE_LIBRARY_GUIDE.md](docs/NATIVE_LIBRARY_GUIDE.md)** - Complete setup and troubleshooting guide
- **[GIT_CLEANUP_COMPLETE.md](GIT_CLEANUP_COMPLETE.md)** - Git repository cleanup details
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Detailed cleanup summary

## Troubleshooting

### Library Not Found

```bash
# Verify installation
ls -lh ~/.gollek/libs/llama/

# Set explicit path
export GOLLEK_LLAMA_LIB_PATH=~/.gollek/libs/llama/libllama.dylib
```

### Permission Issues

```bash
# Set executable permissions
chmod +x ~/.gollek/libs/*/*.dylib
```

### macOS Quarantine

```bash
# Clear quarantine attributes
xattr -dr com.apple.quarantine ~/.gollek/libs/llama
xattr -dr com.apple.quarantine ~/.gollek/libs/onnxruntime
```

## Benefits

### For Developers
- ✅ No large files in git history
- ✅ Faster clone and pull operations
- ✅ Easy library updates
- ✅ Consistent across platforms

### For CI/CD
- ✅ Pre-install libraries in pipeline
- ✅ Use `GOLLEK_NATIVE_LIB_DIR` for custom paths
- ✅ Cache `~/.gollek/libs/` between builds

### For Production
- ✅ Version-manage libraries separately
- ✅ Rollback library versions easily
- ✅ Audit library installations

## Testing

### Verify Library Loading

```bash
# Run with debug logging
java -Dgollek.logging.level=DEBUG -jar gollek.jar chat --model <model>

# Should see:
# [INFO] Loaded native library: /Users/you/.gollek/libs/llama/libllama.dylib
```

### Check Environment

```bash
# Verify environment variables
echo $GOLLEK_NATIVE_LIB_DIR
echo $GOLLEK_LLAMA_LIB_PATH

# List installed libraries
make -f Makefile.native verify-libs
```

## Next Steps

1. ✅ **Done**: Native library standardization complete
2. ⏭️ **Push changes**: `git push`
3. ⏭️ **Notify team**: Share migration guide
4. ⏭️ **Update CI/CD**: Configure library installation in pipelines

## Team Notification Template

```
📢 UPDATE: Native Library Location Standardized

The Gollek platform now uses ~/.gollek/libs/ for all native libraries.

WHAT CHANGED:
- Native libraries moved to ~/.gollek/libs/<runner>/
- No more large binary files in git repository
- Repository size reduced by 30%

ACTION REQUIRED:
1. Pull latest changes: git pull
2. Install libraries: make -f Makefile.native install-native-libs
3. Verify: ls -lh ~/.gollek/libs/*/

DOCUMENTATION:
- docs/NATIVE_LIBRARY_GUIDE.md - Complete guide
- Makefile.native - Installation helper

Questions? Let me know!
```

## Summary

**Status**: ✅ **COMPLETE**

All native library loading has been standardized to use `~/.gollek/libs/` with:
- Comprehensive documentation
- Easy installation via Makefile
- Environment variable overrides
- Backwards compatibility with legacy locations
- Proper gitignore rules

**Repository is ready for push!** 🚀
