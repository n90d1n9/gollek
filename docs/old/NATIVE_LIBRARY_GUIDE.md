# Native Library Management Guide

## Overview

Gollek now uses a **standardized native library location** at `~/.gollek/libs/` for all inference engine native libraries. This provides:

- ✅ Centralized library management
- ✅ No need to track large binaries in git
- ✅ Easy library updates and version management
- ✅ Consistent across all runners (GGUF, ONNX, LibTorch, TFLite)

## Directory Structure

```
~/.gollek/
└── libs/
    ├── llama/              # GGUF / llama.cpp libraries
    │   ├── libllama.dylib
    │   ├── libggml.dylib
    │   ├── libggml-base.dylib
    │   └── libggml-cpu.dylib
    │
    ├── onnxruntime/        # ONNX Runtime libraries
    │   └── libonnxruntime.dylib
    │
    ├── libtorch/           # PyTorch / LibTorch libraries
    │   ├── libtorch.dylib
    │   ├── libtorch_cpu.dylib
    │   ├── libc10.dylib
    │   └── libomp.dylib
    │
    └── litert/            # TensorFlow Lite libraries
        └── libtensorflowlite.dylib
```

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOLLEK_NATIVE_LIB_DIR` | Override default `~/.gollek/libs/` | `/opt/gollek/libs` |
| `GOLLEK_LLAMA_LIB_DIR` | Specific llama.cpp library directory | `~/.gollek/libs/llama` |
| `GOLLEK_LLAMA_LIB_PATH` | Specific llama.cpp library file | `~/.gollek/libs/llama/libllama.dylib` |
| `GOLLEK_ONNX_LIB_DIR` | Specific ONNX Runtime library directory | `~/.gollek/libs/onnxruntime` |
| `GOLLEK_ONNX_LIB_PATH` | Specific ONNX Runtime library file | `~/.gollek/libs/onnxruntime/libonnxruntime.dylib` |
| `GOLLEK_LIBTORCH_SOURCE_DIR` | LibTorch source/vendor directory | `~/.gollek/source/vendor/libtorch` |
| `LIBTORCH_PATH` | Legacy LibTorch path | `/opt/libtorch` |

### Priority Order

Libraries are loaded in this priority order:

1. **Explicit configuration** (`native.library-path` in config)
2. **Environment variables** (`GOLLEK_*_LIB_PATH`)
3. **Standard location** (`~/.gollek/libs/<runner>/`)
4. **Legacy locations** (`~/.gollek/native-libs/`, `~/.gollek/source/vendor/`)
5. **System library path** (`java.library.path`)
6. **Build directories** (project-local `build/lib`, `target/`)

## Setup Instructions

### Option 1: Copy from Build Directory

#### GGUF / llama.cpp

```bash
# Create standard directory
mkdir -p ~/.gollek/libs/llama

# Copy from build directory
cp plugins/runner/gguf/gollek-ext-runner-gguf/src/main/resources/native-libs/cpu/*.dylib ~/.gollek/libs/llama/
# OR from build output
cp extension/format/gguf/gollek-ext-runner-gguf/target/llama-cpp/lib/*.dylib ~/.gollek/libs/llama/

# Set executable (macOS/Linux)
chmod +x ~/.gollek/libs/llama/*.dylib

# Clear macOS quarantine (if needed)
xattr -dr com.apple.quarantine ~/.gollek/libs/llama
```

#### ONNX Runtime

```bash
# Create standard directory
mkdir -p ~/.gollek/libs/onnxruntime

# Copy from build directory
cp plugins/runner/onnx/gollek-runner-onnx/src/main/cpp/onnxruntime/build/onnxruntime-osx-arm64-1.19.2/lib/libonnxruntime.dylib ~/.gollek/libs/onnxruntime/

# Set executable
chmod +x ~/.gollek/libs/onnxruntime/*.dylib

# Clear macOS quarantine
xattr -dr com.apple.quarantine ~/.gollek/libs/onnxruntime
```

#### LibTorch

```bash
# Create standard directory
mkdir -p ~/.gollek/libs/libtorch

# Copy from LibTorch installation
cp /path/to/libtorch/lib/*.dylib ~/.gollek/libs/libtorch/

# Set executable
chmod +x ~/.gollek/libs/libtorch/*.dylib
```

### Option 2: Use Makefile Helper

```bash
# Install all native libraries to standard location
make install-native-libs

# Install specific runner
make install-gguf-libs
make install-onnx-libs
make install-libtorch-libs
```

### Option 3: Download Pre-built Binaries

#### GGUF / llama.cpp

```bash
# Download pre-built llama.cpp binaries
cd ~/.gollek/libs/llama
curl -L https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-macos-arm64.tar.gz | tar xz
```

#### ONNX Runtime

```bash
# Download ONNX Runtime
cd ~/.gollek/libs/onnxruntime
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-arm64-1.19.2.tgz | tar xz
cp lib/libonnxruntime.dylib .
```

## Verification

### Check Library Installation

```bash
# List installed libraries
ls -lh ~/.gollek/libs/*/

# Check specific library
file ~/.gollek/libs/llama/libllama.dylib
```

### Test Loading

```bash
# Run a simple inference test
java -jar gollek.jar chat --model <model> --prompt "Hello"

# Check logs for library loading messages
# Should see: "Loaded native library: /Users/you/.gollek/libs/llama/libllama.dylib"
```

## Migration from Old Structure

If you have libraries in the old locations:

```bash
# Migrate llama.cpp
if [ -d ~/.gollek/source/vendor/llama.cpp ]; then
    mkdir -p ~/.gollek/libs/llama
    cp ~/.gollek/source/vendor/llama.cpp/build/bin/*.dylib ~/.gollek/libs/llama/ 2>/dev/null || true
fi

# Migrate ONNX Runtime
if [ -d ~/.gollek/source/vendor/onnxruntime ]; then
    mkdir -p ~/.gollek/libs/onnxruntime
    cp ~/.gollek/source/vendor/onnxruntime/lib/*.dylib ~/.gollek/libs/onnxruntime/ 2>/dev/null || true
fi

# Migrate legacy native-libs
if [ -d ~/.gollek/native-libs ]; then
    cp ~/.gollek/native-libs/*.dylib ~/.gollek/libs/ 2>/dev/null || true
fi
```

## Troubleshooting

### Library Not Found

**Error**: `Native library not found at ~/.gollek/libs/llama/libllama.dylib`

**Solutions**:
1. Verify library exists: `ls -lh ~/.gollek/libs/llama/`
2. Check file permissions: `chmod +x ~/.gollek/libs/llama/*.dylib`
3. Clear macOS quarantine: `xattr -dr com.apple.quarantine ~/.gollek/libs/llama`
4. Set explicit path: `export GOLLEK_LLAMA_LIB_PATH=~/.gollek/libs/llama/libllama.dylib`

### UnsatisfiedLinkError

**Error**: `UnsatisfiedLinkError: dlopen: library not loaded`

**Solutions**:
1. Check dependencies: `otool -L ~/.gollek/libs/llama/libllama.dylib`
2. Ensure all dependencies are in same directory or system path
3. Try loading with verbose logging: add `-Dgollek.logging.level=DEBUG`

### macOS Quarantine Issues

**Error**: `Library not loaded: cannot load file`

**Solution**:
```bash
# Clear quarantine attributes
xattr -dr com.apple.quarantine ~/.gollek/libs/llama
xattr -dr com.apple.quarantine ~/.gollek/libs/onnxruntime
xattr -dr com.apple.quarantine ~/.gollek/libs/libtorch
```

## Best Practices

1. **Never commit native libraries to git** - Use `~/.gollek/libs/` which is in `.gitignore`
2. **Use environment variables for CI/CD** - Set `GOLLEK_NATIVE_LIB_DIR` in pipeline
3. **Keep libraries organized** - One subdirectory per runner type
4. **Document library versions** - Keep track of which version works with your models
5. **Test after updates** - Verify inference works after library updates

## Integration with Build Systems

### Maven

Add to your `pom.xml`:

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-antrun-plugin</artifactId>
            <executions>
                <execution>
                    <phase>process-resources</phase>
                    <configuration>
                        <target>
                            <mkdir dir="${user.home}/.gollek/libs/llama"/>
                            <copy todir="${user.home}/.gollek/libs/llama">
                                <fileset dir="${project.basedir}/native-libs"/>
                            </copy>
                        </target>
                    </configuration>
                    <goals>
                        <goal>run</goal>
                    </goals>
                </execution>
            </executions>
        </plugin>
    </plugins>
</build>
```

### Gradle

Add to your `build.gradle`:

```groovy
task installNativeLibs(type: Copy) {
    from 'native-libs'
    into "${System.getProperty('user.home')}/.gollek/libs/llama"
}

processResources.dependsOn installNativeLibs
```

## Related Documentation

- [GIT_CLEANUP_COMPLETE.md](GIT_CLEANUP_COMPLETE.md) - Git repository cleanup
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Detailed cleanup summary
