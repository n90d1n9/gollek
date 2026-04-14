# 🎯 JBang Examples Index

Complete index of all JBang examples in the Gollek repository.

## 📂 Directory Structure

```
gollek/examples/jbang/
├── sdk/
│   ├── unified_framework_demo.java       (v0.3: Unified Runner, Batching, Fusion, Quantization)
│   ├── graph_fusion_example.java         (v0.3: Operation fusion benchmark)
│   ├── 01_tensor_operations.java         (700+ lines, TensorOps demo)
│   ├── 02_vision_transforms.java         (800+ lines, Vision pipeline)
│   ├── 03_tokenization.java              (900+ lines, NLP tokenization)
│   ├── 04_mnist_training.java            (600+ lines, Complete ML pipeline)
│   ├── 05_pytorch_comparison.java        (750+ lines, PyTorch migration)
│   ├── README.md                         (Comprehensive guide)
│   └── [Other examples...]
├── edge/                                 (Edge inference examples)
├── quantizer/                            (Quantization examples)
├── nlp/                                  (NLP pipeline examples)
├── multimodal/                           (Multimodal examples)
└── integration/                          (Third-party integration examples)
```

## 🆕 v0.3 New Examples

| Example | Description | Usage |
|---------|-------------|-------|
| **unified_framework_demo.java** | All v0.3 capabilities demo | `jbang unified_framework_demo.java --demo all` |
| **graph_fusion_example.java** | Operation fusion benchmark | `jbang graph_fusion_example.java --size 8192` |

### Capabilities Demonstrated

- ✅ **Unified Model Runner** - Single API for GGUF, ONNX, LiteRT, TensorRT
- ✅ **Dynamic Batching** - Automatic request batching with SLA
- ✅ **Graph Fusion** - Fuse operations into single kernels
- ✅ **Quantization** - INT4/INT8/FP8 with AWQ/GPTQ
- ✅ **Memory Pool** - Zero-copy tensor allocation

## 📚 Example Categories

### Core SDK Examples (sdk/)

#### 🔢 TensorOps Examples
- **01_tensor_operations.java** - Advanced tensor manipulation
  - Difficulty: ⭐ Beginner
  - Topics: Slicing, indexing, concatenation, gathering, masking
  - Time: 10 minutes
  - Best for: Understanding tensor operations

#### 👁️ Vision Examples
- **02_vision_transforms.java** - Image preprocessing pipeline
  - Difficulty: ⭐⭐ Intermediate
  - Topics: Resize, crop, normalize, augment, compose
  - Time: 15 minutes
  - Best for: Computer vision tasks

#### 📝 NLP Examples
- **03_tokenization.java** - Text tokenization
  - Difficulty: ⭐⭐ Intermediate
  - Topics: Encoding, decoding, special tokens, batch processing
  - Time: 20 minutes
  - Best for: NLP model preparation

#### 🤖 Full ML Pipeline Examples
- **04_mnist_training.java** - Complete training pipeline
  - Difficulty: ⭐⭐⭐ Advanced
  - Topics: Model definition, training loop, validation, evaluation
  - Time: 30 minutes
  - Best for: Learning complete workflows

#### 🔄 Migration Guides
- **05_pytorch_comparison.java** - PyTorch vs Gollek
  - Difficulty: ⭐ Beginner
  - Topics: API comparison, migration patterns, effort estimates
  - Time: 10 minutes
  - Best for: PyTorch developers

### Quick Start Examples

- **gollek-quickstart.java** - Get started quickly
- **gollek-sdk-core-example.java** - Core SDK basics
- **gollek-sdk-vision-example.java** - Vision tasks
- **gollek-sdk-train-example.java** - Training basics
- **gollek-sdk-augment-example.java** - Data augmentation
- **gollek-sdk-export-example.java** - Model export

---

## 🚀 Getting Started

### Step 1: Install JBang
```bash
# macOS
brew install jbang/tap/jbang

# Linux/Windows - See official docs
curl -Ls https://sh.jbang.dev | bash -s - app setup
```

### Step 2: Navigate to Examples
```bash
cd gollek/examples/jbang/sdk
```

### Step 3: Run Examples
```bash
# Beginner path
jbang 01_tensor_operations.java
jbang 05_pytorch_comparison.java

# Intermediate path
jbang 02_vision_transforms.java
jbang 03_tokenization.java

# Advanced path
jbang 04_mnist_training.java

# With parameters
jbang 04_mnist_training.java 10 32
```

---

## 📊 Quick Reference

| Example | File | Lines | Time | Level | Topics |
|---------|------|-------|------|-------|--------|
| TensorOps | 01_tensor_operations.java | 700+ | 10m | ⭐ | 7 ops |
| Native STT | `multimodal/native_whisper_stt.java` | 100+ | 5m | ⭐ | 🎙️ SIMD STT |
| SIMD Audio | `machine_learning/simd_audio_processing.java` | 50+ | 2m | ⭐ | ⚡ Vector API |
| Tokenization | 03_tokenization.java | 900+ | 20m | ⭐⭐ | 7 concepts |
| MNIST | 04_mnist_training.java | 600+ | 30m | ⭐⭐⭐ | 6 stages |
| PyTorch | 05_pytorch_comparison.java | 750+ | 10m | ⭐ | 6 areas |

---

## 🎓 Learning Paths

### Path 1: Basics (60 min)
1. Quickstart example (5 min)
2. 01_tensor_operations (10 min)
3. 05_pytorch_comparison (10 min)
4. 02_vision_transforms (15 min)
5. Quick review (10 min)

### Path 2: Deep Learning (90 min)
1. 05_pytorch_comparison (10 min)
2. 01_tensor_operations (10 min)
3. 02_vision_transforms (15 min)
4. 03_tokenization (20 min)
5. 04_mnist_training (35 min)

### Path 3: Production Ready (120 min)
1. All core examples (90 min)
2. Code modifications (15 min)
3. Custom script creation (15 min)

---

## 🔍 Find What You Need

### By Task
- **Create tensors** → 01_tensor_operations
- **Preprocess images** → 02_vision_transforms
- **Process text** → 03_tokenization
- **Train models** → 04_mnist_training
- **Migrate from PyTorch** → 05_pytorch_comparison

### By Difficulty
- **Beginner** → 01, 05, gollek-quickstart
- **Intermediate** → 02, 03
- **Advanced** → 04, custom scripts

### By Domain
- **Computer Vision** → 02_vision_transforms
- **NLP** → 03_tokenization
- **Deep Learning** → 04_mnist_training
- **General ML** → 01_tensor_operations

---

## 📖 Documentation Map

```
gollek/
├── docs/
│   ├── api/
│   │   ├── tensorops/          (01_tensor_operations reference)
│   │   ├── vision-transforms/  (02_vision_transforms reference)
│   │   ├── tokenizer/          (03_tokenization reference)
│   │   └── nn/                 (04_mnist_training reference)
│   ├── migration/
│   │   └── pytorch.md          (05_pytorch_comparison reference)
│   └── guides/
│       ├── getting-started/
│       └── jbang-examples/
│
├── examples/
│   ├── jbang/
│   │   ├── sdk/                (This directory)
│   │   ├── advanced/           (Future)
│   │   └── tutorials/          (Future)
│   │
│   └── maven/                  (Spring into Maven projects)
```

---

## 💡 Advanced Usage

### Run in IDE
```bash
jbang edit --open=code 01_tensor_operations.java
```

### Run with Custom JVM Options
```bash
jbang --java-options="-Xmx4g" 04_mnist_training.java
```

### Create Your Own Script
```bash
cat > my_example.java << 'EOF'
#!/usr/bin/env jbang
//DEPS tech.kayys.gollek:gollek-sdk-parent:0.2.0:pom

public class my_example {
    public static void main(String[] args) {
        System.out.println("Hello Gollek!");
    }
}
EOF

jbang my_example.java
```

### Share as Gist
```bash
jbang https://gist.github.com/.../my_script.java
```

---

## 🐛 Troubleshooting

### Issue: JBang not found
**Solution:** Add to PATH
```bash
export PATH="$PATH:$HOME/.jbang/bin"
```

### Issue: Dependencies not downloading
**Solution:** Clear cache
```bash
rm -rf ~/.jbang
jbang 01_tensor_operations.java
```

### Issue: Out of memory
**Solution:** Increase JVM heap
```bash
jbang --java-options="-Xmx8g" 04_mnist_training.java
```

### Issue: Slow first run
**Solution:** Normal behavior (dependency download)
- First run: 2-3 minutes (downloads SDK)
- Subsequent runs: 1-2 seconds startup

---

## 📞 Support

- **GitHub Issues:** https://github.com/bhangun/gollek/issues
- **Discussions:** https://github.com/bhangun/gollek/discussions
- **Documentation:** https://gollek.io/docs
- **Examples Index:** This file

---

## ✨ Example Features

All examples include:
- ✅ Production-grade code (no placeholders)
- ✅ Comprehensive documentation
- ✅ Clear output formatting
- ✅ Best practices
- ✅ Real-world patterns
- ✅ Error handling
- ✅ Performance tips
- ✅ Migration guides

---

## 🎯 Next Steps After Examples

1. **Modify examples** - Change parameters, try new patterns
2. **Create custom scripts** - Build your own JBang scripts
3. **Graduate to Maven** - Create full projects with pom.xml
4. **Explore docs** - Read API documentation for each module
5. **Join community** - Share examples, ask questions, contribute

---

**Last Updated:** April 8, 2026  
**Gollek SDK Version:** v0.2.0  
**Status:** ✅ Ready to Use

[📖 See Full Guide →](./sdk/README.md)
