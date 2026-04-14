# Developer Setup Guide

Complete guide for setting up a Gollek development environment.

---

## System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Java** | 25 (Loom/Panama) | Required for virtual threads, FFM, vector API |
| **Maven** | 3.8+ | Build tool |
| **Git** | 2.30+ | Version control |
| **OS** | macOS, Linux, Windows | Apple Silicon, x86_64, aarch64 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/bhangun/gollek.git
cd gollek
```

---

## 2. Install Java 25

### macOS

```bash
# Using SDKMAN
sdk install java 25.ea.32-open

# Or using Homebrew
brew install --cask temurin@25
```

### Linux

```bash
# Using SDKMAN
sdk install java 25.ea.32-open

# Or download from Adoptium
wget https://github.com/adoptium/temurin25-binaries/releases/download/...
```

### Verify

```bash
java -version
# Expected: openjdk version "25.x.x"
```

---

## 3. Build the Project

### Full Build

```bash
# Clean build (skip tests for speed)
mvn clean install -DskipTests

# With tests
mvn clean install
```

### Build Specific Modules

```bash
# Build core framework
mvn clean install -pl framework/lib/gollek-ml-api,framework/lib/gollek-ml-autograd -am

# Build optimization suite
mvn clean install -pl framework/lib/gollek-ml-optimize -am

# Build ONNX integration
mvn clean install -pl framework/lib/gollek-ml-onnx -am

# Build inference engine
mvn clean install -pl inference/gollek-runtime-inference -am
```

### Build Native Image (Optional)

```bash
# Install GraalVM
sdk install java 25-graalce

# Build native image
mvn clean package -Pnative -DskipTests
```

---

## 4. Run Tests

### All Tests

```bash
mvn test
```

### Specific Module Tests

```bash
# Test tensor operations
mvn test -pl framework/lib/gollek-ml-tensor

# Test optimization
mvn test -pl framework/lib/gollek-ml-optimize

# Test inference
mvn test -pl inference/gollek-runtime-inference
```

---

## 5. Run Examples

### JBang Examples

```bash
# Hello world
jbang examples/jbang/common/hello_gollek.java

# Comprehensive demo
jbang examples/jbang/sdk/unified_framework_demo.java --demo all
```

### Maven Examples

```bash
# Run FFN example
mvn exec:java -pl framework/lib/gollek-ml-examples \
    -Dexec.mainClass=tech.kayys.gollek.ml.examples.SimpleFFNExample

# Run CNN example
mvn exec:java -pl framework/lib/gollek-ml-examples \
    -Dexec.mainClass=tech.kayys.gollek.ml.examples.MNISTCNNExample
```

---

## 6. IDE Setup

### IntelliJ IDEA

1. **Import Project**: File → Open → Select `gollek/pom.xml`
2. **Enable Preview Features**:
   - Settings → Build → Compiler → Java Compiler
   - Add `--enable-preview --add-modules jdk.incubator.vector` to "Additional command line parameters"
3. **Enable Native Access**:
   - Add `--enable-native-access=ALL-UNNAMED` to VM options

### VS Code

1. **Install Extensions**:
   - Java Extension Pack
   - Maven for Java
2. **Configure `settings.json`**:
   ```json
   {
     "java.compile.nullAnalysis.mode": "automatic",
     "java.jdt.ls.vmargs": "--enable-preview --add-modules jdk.incubator.vector"
   }
   ```

### Eclipse

1. **Import as Maven Project**: File → Import → Maven → Existing Maven Projects
2. **Enable Preview**: Project Properties → Java Compiler → Enable preview features

---

## 7. GPU Development

### CUDA Setup

```bash
# Install CUDA Toolkit (11.8+)
# macOS: Not supported (use Metal)
# Linux: https://developer.nvidia.com/cuda-downloads
# Windows: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi
```

### Metal Setup (Apple Silicon)

Works out of the box on M1/M2/M3. No additional setup needed.

---

## 8. Code Style

### Formatting

The project uses standard Java conventions. Format with:

```bash
# IntelliJ
mvn formatter:format

# Or use IDE auto-format
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Classes** | PascalCase | `ModelRunner`, `FusionEngine` |
| **Methods** | camelCase | `infer()`, `compile()` |
| **Constants** | UPPER_SNAKE_CASE | `MAX_BATCH_SIZE` |
| **Packages** | lowercase | `tech.kayys.gollek.ml.runner` |

---

## 9. Contributing

### Branch Strategy

- `main` - Stable releases
- `develop` - Development branch
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches

### Pull Request Process

1. Create feature branch from `develop`
2. Make changes with tests
3. Run full test suite: `mvn clean test`
4. Submit PR with description
5. Address review comments
6. Merge after approval

### Commit Messages

Follow conventional commits:

```
feat: add graph fusion engine
fix: resolve memory leak in pooling
docs: update installation guide
test: add quantization tests
```

---

## 10. Troubleshooting

### "java.lang.UnsupportedClassVersionError"

Ensure Java 25:

```bash
java -version
export JAVA_HOME=/path/to/java25
```

### "Preview features not enabled"

Add to Maven compiler config:

```xml
<compilerArgs>
    <arg>--enable-preview</arg>
    <arg>--add-modules</arg>
    <arg>jdk.incubator.vector</arg>
</compilerArgs>
```

### "Native memory access denied"

Add VM options:

```bash
--enable-native-access=ALL-UNNAMED
```

### Build fails on specific module

Clean and rebuild:

```bash
mvn clean
mvn install -pl <module> -am
```

---

## Next Steps

- **[Examples](examples/docs/SETUP.md) - Run example code
- **[QUICKSTART.md](QUICKSTART.md) - Quick start guide
- **[Architecture](docs/) - Internal architecture docs
- **[Contributing](CONTRIBUTING.md) - Contribution guidelines
