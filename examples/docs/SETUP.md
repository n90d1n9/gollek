# Gollek Examples Setup

Quick setup guide for running Gollek examples.

---

## Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| **Java** | 25+ | `java -version` |
| **JBang** | 0.120+ | `jbang --version` |
| **Jupyter** (optional) | Latest | `jupyter --version` |
| **Maven** (optional) | 3.8+ | `mvn --version` |

---

## Option 1: JBang (Easiest)

No build tools needed. JBang handles everything:

```bash
# Install JBang
curl -Ls https://sh.jbang.dev | bash -s -

# Run an example
cd gollek/examples/jbang/
jbang common/hello_gollek.java
```

**Available Examples:**
```bash
# SDK examples
jbang sdk/unified_framework_demo.java --demo all
jbang sdk/graph_fusion_example.java

# Edge inference
jbang edge/image_classification.java

# Quantization
jbang quantizer/gollek-quantizer-turboquant.java --mode demo
```

---

## Option 2: Jupyter Notebooks

Interactive Python-style experience for Java:

```bash
# Install Jupyter
pip install jupyter

# Install Java kernel
jbang app install --force jupyter-java@quarkusio

# Launch
cd gollek/examples/jupyter/
jupyter notebook
```

**Available Notebooks:**
- `01-getting-started.ipynb` - SDK basics
- `06-llm-integration.ipynb` - LLM integration

---

## Option 3: Maven Project

For full IDE support:

```bash
# Build the examples
cd gollek/framework/lib/gollek-ml-examples/
mvn clean compile

# Run an example
mvn exec:java -Dexec.mainClass=tech.kayys.gollek.ml.examples.SimpleFFNExample
```

---

## GPU Acceleration

### CUDA (NVIDIA)

```bash
# Verify CUDA availability
nvidia-smi

# Run with CUDA
jbang --java 25 sdk/unified_framework_demo.java --device cuda
```

### Metal (Apple Silicon)

Works automatically on M1/M2/M3:

```bash
# Run with Metal
jbang --java 25 sdk/unified_framework_demo.java --device metal
```

---

## Troubleshooting

### "Could not find or load main class"

Ensure Java 25:

```bash
java -version
# Should show: openjdk version "25.x.x"
```

### "Dependency resolution failed"

Clear JBang cache:

```bash
jbang cache clear
jbang sdk/unified_framework_demo.java
```

### "Native memory access denied"

Add native access flags to JBang script header:

```java
//COMPILE_OPTIONS --enable-native-access=ALL-UNNAMED
//RUNTIME_OPTIONS --enable-native-access=ALL-UNNAMED
```

### Jupyter kernel not found

Install the Java kernel:

```bash
jbang app install --force jupyter-java@quarkusio
jupyter kernelspec list  # Should show 'java' or 'jjava'
```

---

## Next Steps

- **[JBang Setup Guide](/docs/setup/jbang-setup) - Detailed JBang configuration
- **[SDK Installation](/docs/setup/sdk-installation) - Maven/Gradle setup
- **[Examples Index](/examples/jbang/INDEX.md) - Browse all examples
- **[Framework Guide](/docs/framework/) - Learn the ML framework
