# 📚 Gollek SDK v0.2 - JBang Examples

Complete set of examples demonstrating Gollek SDK v0.2 features using JBang (no build required).

## 🚀 Quick Start

### Install JBang

```bash
# macOS (Homebrew)
brew install jbang/tap/jbang

# Linux/Windows
curl -Ls https://sh.jbang.dev | bash -s - app setup

# Or download: https://jbang.dev
```

### Run Examples

```bash
# Navigate to examples directory
cd gollek/examples/jbang/sdk

# Run any example
jbang 01_tensor_operations.java
jbang 02_vision_transforms.java
jbang 03_tokenization.java
jbang 04_mnist_training.java 10 32
jbang 05_pytorch_comparison.java
```

---

## 📋 Examples Overview

### 1️⃣ **TensorOps - Advanced Tensor Operations**

**File:** `01_tensor_operations.java`

**Topics:**
- Tensor creation and manipulation
- Slicing and indexing
- Concatenation and stacking
- Gathering and scattering
- Boolean masking operations
- Element-wise comparisons
- Top-K selection pattern

**Key Concepts:**
```java
// Slicing
Tensor sliced = TensorOps.slice(x, 1, 5, 15);

// Concatenation
Tensor cat = TensorOps.cat(1, List.of(x, y, z));

// Masking
Tensor mask = TensorOps.gt(x, 0);
Tensor positive = TensorOps.maskedSelect(x, mask);
```

**Learning Outcomes:**
- Understand stride-based tensor indexing
- Master data selection patterns
- Learn PyTorch equivalents in Java

**Difficulty:** ⭐ Beginner
**Time:** ~10 minutes

---

### 2️⃣ **Vision Transforms - Image Preprocessing**

**File:** `02_vision_transforms.java`

**Topics:**
- Image resizing (bilinear interpolation)
- Center and random cropping
- Image normalization
- Data augmentation techniques
- Composable pipelines
- Batch processing

**Key Concepts:**
```java
// Build pipeline
var pipeline = new VisionTransforms.Compose(
    new VisionTransforms.Resize(224, 224),
    new VisionTransforms.CenterCrop(224, 224),
    new VisionTransforms.Normalize(mean, std),
    new VisionTransforms.RandomFlip()
);

// Apply to image
Tensor processed = pipeline.apply(image);
```

**Learning Outcomes:**
- Build efficient preprocessing pipelines
- Understand image augmentation strategies
- Learn computer vision best practices

**Difficulty:** ⭐⭐ Intermediate
**Time:** ~15 minutes

---

### 3️⃣ **Tokenization - NLP Text Processing**

**File:** `03_tokenization.java`

**Topics:**
- Text encoding to token IDs
- Token decoding
- Special token handling
- Batch processing with padding
- Attention masks and token types
- Real-world NLP tasks

**Key Concepts:**
```java
// Encode text
String text = "Hello world";
EncodedTokens encoded = tokenizer.encode(text);

// Get token IDs
List<Integer> tokenIds = encoded.tokenIds;

// Get attention mask for padding
int[] attentionMask = encoded.attentionMask;
```

**Learning Outcomes:**
- Understand text tokenization
- Prepare data for NLP models
- Master batch processing strategies
- Learn transformer input formats

**Difficulty:** ⭐⭐ Intermediate
**Time:** ~20 minutes

---

### 4️⃣ **MNIST Training - Complete ML Pipeline**

**File:** `04_mnist_training.java`

**Usage:**
```bash
# Default: 10 epochs, batch size 32
jbang 04_mnist_training.java

# Custom configuration
jbang 04_mnist_training.java 5 16    # 5 epochs, batch 16
jbang 04_mnist_training.java 20 64   # 20 epochs, batch 64
```

**Topics:**
- Model architecture definition
- Data loading and preprocessing
- Full training loop
- Loss computation and backpropagation
- Validation and evaluation
- Learning rate scheduling
- Model persistence

**Key Concepts:**
```java
// Model definition
class MNISTNet extends NNModule {
    Conv2d conv1 = new Conv2d(1, 32, 3, 1, 1);
    Linear fc1 = new Linear(3136, 256);
    Linear fc2 = new Linear(256, 10);
    
    @Override
    public GradTensor forward(GradTensor input) {
        // Forward pass implementation
    }
}

// Training
for (int epoch = 0; epoch < epochs; epoch++) {
    for (Batch batch : dataLoader) {
        GradTensor output = model.forward(input);
        GradTensor loss = lossFunction.forward(output, target);
        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();
    }
}
```

**Expected Output:**
```
Epoch  1: Loss: 0.287, Train Acc: 91.2%, Val Acc: 96.5%, Time: 45000ms
Epoch  5: Loss: 0.045, Train Acc: 98.5%, Val Acc: 98.2%, Time: 44000ms
Epoch 10: Loss: 0.012, Train Acc: 99.6%, Val Acc: 99.1%, Time: 43000ms

Training completed in 450.0 seconds
Final Evaluation:
  Test set accuracy: 99.0%
```

**Learning Outcomes:**
- Build complete ML pipelines
- Implement training loops
- Understand model architecture
- Master optimization techniques
- Learn evaluation strategies

**Difficulty:** ⭐⭐⭐ Advanced
**Time:** ~30 minutes (+ 7 minutes running)

---

### 5️⃣ **PyTorch Comparison - Migration Guide**

**File:** `05_pytorch_comparison.java`

**Topics:**
- Side-by-side code comparisons
- API differences and similarities
- Migration patterns
- Type system differences
- Performance characteristics

**Key Comparisons:**
| Aspect | PyTorch | Gollek | Effort |
|--------|---------|--------|--------|
| Tensor creation | `torch.randn()` | `Tensor.randn()` | 🟢 Same |
| Tensor slicing | `x[1:3]` | `TensorOps.slice()` | 🟡 Method call |
| Model definition | `nn.Module` | `NNModule` | 🟢 Similar |
| Training loop | Python syntax | Java syntax | 🟡 Minor |
| Data loading | `DataLoader` | `DataLoader` | 🟢 Same |

**Learning Outcomes:**
- Understand API correspondence
- Learn migration strategy
- Estimate effort for projects
- Understand implementation differences

**Difficulty:** ⭐ Beginner
**Time:** ~10 minutes

---

## 🎯 Learning Paths

### Path 1: Complete Beginner
1. **01_tensor_operations** - Learn tensor basics (10 min)
2. **05_pytorch_comparison** - Understand API (10 min)
3. **02_vision_transforms** - Try practical example (15 min)
4. **04_mnist_training** - Run complete example (30 min)
   
**Total:** ~65 minutes

### Path 2: PyTorch Developer
1. **05_pytorch_comparison** - See similarities (10 min)
2. **01_tensor_operations** - Learn differences (10 min)
3. **04_mnist_training** - Build complete model (30 min)
4. **03_tokenization** - Explore NLP (20 min)

**Total:** ~70 minutes

### Path 3: ML Practitioner
1. **02_vision_transforms** - Vision preprocessing (15 min)
2. **03_tokenization** - NLP preprocessing (20 min)
3. **04_mnist_training** - Training pipeline (30 min)
4. **01_tensor_operations** - Advanced patterns (10 min)

**Total:** ~75 minutes

### Path 4: Deep Dive (All)
All 5 examples in sequence (~100 minutes)

---

## 🔧 Example Features

### Code Quality
- ✅ Production-grade code (no placeholders)
- ✅ Comprehensive documentation
- ✅ Proper error handling
- ✅ Clear output formatting
- ✅ Best practices demonstrated

### Educational Value
- ✅ Commented code sections
- ✅ Conceptual explanations
- ✅ Multiple patterns shown
- ✅ Real-world use cases
- ✅ Common pitfalls addressed

### Runability
- ✅ Single-file JBang scripts
- ✅ No external dependencies (except SDK)
- ✅ No compilation needed
- ✅ Self-contained examples
- ✅ Configurable parameters

---

## 📊 Statistics

| Example | Lines | Time | Topics | Difficulty |
|---------|-------|------|--------|------------|
| 01_tensor_operations | 400+ | 10 min | 7 | ⭐ |
| 02_vision_transforms | 500+ | 15 min | 7 | ⭐⭐ |
| 03_tokenization | 600+ | 20 min | 7 | ⭐⭐ |
| 04_mnist_training | 300+ | 30 min | 6 | ⭐⭐⭐ |
| 05_pytorch_comparison | 400+ | 10 min | 6 | ⭐ |
| **Total** | **2,200+** | **~85 min** | **Many** | **Varied** |

---

## 🚀 Next Steps

### After Running Examples

1. **Modify and Experiment**
   ```bash
   # Edit example
   nano 01_tensor_operations.java
   
   # Run modified version
   jbang 01_tensor_operations.java
   ```

2. **Create Your Own Scripts**
   ```bash
   # Create new script
   cat > my_example.java << 'EOF'
   ///usr/bin/env jbang
   //DEPS tech.kayys.gollek:gollek-sdk-tensor:0.2.0
   
   public class my_example {
       public static void main(String[] args) {
           System.out.println("Hello Gollek!");
       }
   }
   EOF
   
   jbang my_example.java
   ```

3. **Build Full Projects**
   - Graduate from JBang to Maven projects
   - See: `gollek/gollek-sdk/lib` structure
   - Create `pom.xml` with Gollek dependencies

4. **Explore More Examples**
   - Check `gollek/examples/jbang/` for more
   - Browse documentation at gollek.io/docs
   - Join community discussions

---

## 📚 Related Documentation

- [SDK v0.2 Release Notes](/docs/release-notes-v0.2)
- [TensorOps API Reference](/docs/api-tensorops)
- [Vision Transforms Guide](/docs/api-vision-transforms)
- [Tokenizer API](/docs/api-tokenizer)
- [PyTorch Migration Guide](/docs/migration-pytorch)
- [Framework Overview](/docs/framework)

---

## 🆘 Troubleshooting

### JBang Not Found
```bash
# Add to PATH
export PATH="$PATH:$HOME/.jbang/bin"
```

### Dependencies Not Downloading
```bash
# Clear cache
rm -rf ~/.jbang

# Try again
jbang 01_tensor_operations.java
```

### Memory Issues
```bash
# Set JVM memory
jbang --java-options="-Xmx4g" 04_mnist_training.java
```

### Slow First Run
- First run downloads dependencies (~2-3 min)
- Subsequent runs are fast (~1-2 sec startup)
- Network latency may affect initial load

---

## 💡 Tips & Tricks

### Run with Arguments
```bash
# MNIST example with custom parameters
jbang 04_mnist_training.java 5 64

# First arg: epochs (default 10)
# Second arg: batch size (default 32)
```

### Edit in IDE
```bash
# Open in VS Code
jbang edit --open=code 01_tensor_operations.java

# Edit and save, then run
jbang 01_tensor_operations.java
```

### Share Scripts
- JBang scripts are portable
- Single .java file contains everything
- Share via gist, GitHub, etc
- Run directly: `jbang https://gist.github.com/.../my_script.java`

---

## 🎓 Key Takeaways

✅ **Ease:** Single-file JBang scripts are easy to learn
✅ **Practical:** Real-world patterns demonstrated
✅ **Complete:** Full ML pipeline examples included
✅ **Accessible:** No build complexity required
✅ **Progressive:** Difficulty increases gradually
✅ **Executable:** All examples runnable as-is

---

## 📞 Get Help

- **GitHub Issues:** https://github.com/bhangun/gollek/issues
- **Discussions:** https://github.com/bhangun/gollek/discussions
- **Documentation:** https://gollek.io/docs
- **Email:** team@gollek.io

---

**Version:** Gollek SDK v0.2.0  
**Last Updated:** April 8, 2026  
**Status:** ✅ Ready to Use
