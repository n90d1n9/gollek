# Gollek LiteRT Edge Examples

Enhanced examples demonstrating how to use LiteRT for edge device inference with JBang. These examples cover vision models (`.litert`) and language models (`.litertlm`) with comprehensive error handling, validation, and performance benchmarking.

## 🚀 Quick Start

### Prerequisites

- **Java 21 or higher** - [Download Adoptium](https://adoptium.net/)
- **JBang** - Install with: `curl -Ls https://sh.jbang.dev | bash -s -`
- **Maven** - For building the SDK modules
- **LiteRT Models** - `.litertlm` for language models, `.litert` for vision models

### Building the SDK

Before running the examples, build and install the LiteRT SDK modules:

```bash
# Navigate to the gollek directory
cd gollek

# Build and install the LiteRT SDK module
mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests

# Verify installation
ls ~/.m2/repository/tech/kayys/gollek/gollek-sdk-litert/
```

### Running Examples

**Using the enhanced runner script (recommended):**

```bash
# List available examples
./run.sh

# Run an example
./run.sh edge
./run.sh image --model mobilenet.litert --image cat.jpg
./run.sh text --model gemma.litertlm --prompt "Hello world"
./run.sh benchmark --model model.litert --iterations 100
```

**Using JBang directly:**

```bash
# Basic edge inference (demo mode)
jbang LiteRTEdgeExample.java

# With a real model
jbang -Dmodel=/path/to/model.litert LiteRTEdgeExample.java

# Image classification
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --topk 5

# Object detection
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --threshold 0.5

# Text generation
jbang TextGenerationExample.java \
  --model gemma.litertlm \
  --prompt "Once upon a time" \
  --maxtokens 100 \
  --temperature 0.7

# Performance benchmark
jbang BenchmarkExample.java \
  --model model.litert \
  --iterations 100 \
  --batch 8 \
  --output results.csv
```

## 📚 Available Examples

### 1. LiteRTEdgeExample.java - Basic Edge Inference

**Purpose**: Demonstrates core LiteRT SDK capabilities including model loading, synchronous/asynchronous inference, batch processing, and performance metrics.

**Features**:
- ✅ Model loading and management
- ✅ Synchronous and asynchronous inference
- ✅ Batch processing
- ✅ Performance metrics collection
- ✅ Hardware delegate selection
- ✅ Comprehensive error handling

**Usage**:

```bash
# Demo mode (no model required)
jbang LiteRTEdgeExample.java

# With a language model
jbang -Dmodel=gemma-4-E4B-it-litert-lm.litertlm LiteRTEdgeExample.java

# With a vision model
jbang -Dmodel=mobilenet_v2.litert LiteRTEdgeExample.java

# Use GPU delegate
jbang -Dmodel=model.litert -Ddelegate=GPU LiteRTEdgeExample.java

# Custom configuration
jbang -Dmodel=model.litert -Dthreads=8 -Dxnnpack=true LiteRTEdgeExample.java
```

**Example Output**:
```
Configuration:
  Delegate:    AUTO
  Threads:     4
  XNNPACK:     true
  Memory Pool: true

✓ LiteRT SDK initialized
  Version: 0.1.0-SNAPSHOT
  Available: true

Loading model: mobilenet_v2.litert
✓ Model loaded

Model Information:
  ID:    edge-model
  Size:  13.45 MB
  Inputs: 1
  Outputs: 1

Running synchronous inference...
✓ Inference completed in 45ms
  Output size: 4000 bytes

Performance Metrics:
  Total inferences:    5
  Failed inferences:   0
  Avg latency:         42.35 ms
  P50 latency:         41.20 ms
  P95 latency:         48.50 ms
  P99 latency:         49.10 ms
  Peak memory:         45.67 MB
  Current memory:      32.12 MB
  Active delegate:     CPU
```

---

### 2. ImageClassificationExample.java - Image Classification

**Purpose**: Image classification using models like MobileNet, EfficientNet, ResNet, etc. with ImageNet 1000-class labels.

**Features**:
- ✅ Image preprocessing and resizing
- ✅ Single image classification
- ✅ Top-K predictions with ImageNet labels
- ✅ Batch classification with multiple crops
- ✅ Comprehensive error handling and validation

**Usage**:

```bash
# Basic classification
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg

# Top-10 predictions
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --topk 10

# Use GPU acceleration
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --delegate GPU
```

**Example Output**:
```
Top 5 Predictions:
  1. Egyptian cat                  87.45%
  2. tabby                         8.32%
  3. tiger cat                     2.15%
  4. Siamese cat                   1.08%
  5. lynx                          0.52%
```

**Supported Models**:
- MobileNet V2/V3
- EfficientNet B0-B7
- ResNet-50/101
- Inception V3
- Any ImageNet-trained `.litert` classification model

---

### 3. ObjectDetectionExample.java - Object Detection

**Purpose**: Object detection using SSD MobileNet or YOLO models with bounding box visualization.

**Features**:
- ✅ Object detection with bounding boxes
- ✅ Confidence threshold filtering
- ✅ Visual output with drawn detections
- ✅ Batch detection with multiple thresholds
- ✅ COCO dataset labels (80 classes)
- ✅ Automatic output image generation

**Usage**:

```bash
# Basic detection
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg

# Custom confidence threshold
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --threshold 0.6

# Save output to specific file
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --output detected.jpg

# Use CoreML on iOS/macOS
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --delegate COREML
```

**Example Output**:
```
Detections:
  1. Class 3 (car) (92.34%) at [120, 80, 200, 150]
  2. Class 0 (person) (87.12%) at [300, 100, 100, 250]
  3. Class 16 (dog) (76.45%) at [50, 200, 80, 100]

Drawing detections on image...
  Saved to: street_detected.jpg
```

**Supported Models**:
- SSD MobileNet V1/V2
- YOLO V5/V8 (converted to LiteRT)
- Any COCO-trained `.litert` detection model

---

### 4. TextGenerationExample.java - Text Generation ⭐ NEW

**Purpose**: Text generation using language models like Gemma, Llama, Phi, etc.

**Features**:
- ✅ Text tokenization and preprocessing
- ✅ Single-turn text generation
- ✅ Interactive chat mode
- ✅ Configurable generation parameters (temperature, top-k, max tokens)
- ✅ Streaming generation simulation

**Usage**:

```bash
# Basic text generation
jbang TextGenerationExample.java \
  --model gemma-4-E4B-it-litert-lm.litertlm \
  --prompt "Hello, how are you?"

# With custom parameters
jbang TextGenerationExample.java \
  --model model.litertlm \
  --prompt "Write a story about" \
  --maxtokens 100 \
  --temperature 0.8 \
  --topk 50

# Interactive chat mode
jbang TextGenerationExample.java \
  --model model.litertlm \
  --interactive
```

**Example Output**:
```
Configuration:
  Model:       gemma-4-E4B-it-litert-lm.litertlm
  Prompt:      Hello, how are you?
  Max Tokens:  50
  Temperature: 0.7
  Top-K:       40
  Delegate:    AUTO

Loading model: gemma-4-E4B-it-litert-lm.litertlm
✓ Model loaded

Generating text...
Prompt: "Hello, how are you?"

Generated text:
I'm doing well, thank you for asking! How can I help you today?

✓ Generation completed in 234ms
  Tokens/sec: 213.68
```

**Supported Models**:
- Gemma 2B/7B
- Llama 2/3
- Microsoft Phi-2/3
- Any conversational `.litertlm` language model

---

### 5. BenchmarkExample.java - Performance Benchmarking ⭐ NEW

**Purpose**: Comprehensive performance benchmarking tool for measuring LiteRT inference performance.

**Features**:
- ✅ Single inference latency benchmark
- ✅ Batch inference throughput benchmark
- ✅ Async inference concurrency benchmark
- ✅ Sustained load testing
- ✅ Statistical analysis (mean, median, percentiles)
- ✅ CSV report generation
- ✅ Memory usage tracking

**Usage**:

```bash
# Basic benchmark
jbang BenchmarkExample.java \
  --model model.litert

# Custom benchmark parameters
jbang BenchmarkExample.java \
  --model model.litert \
  --iterations 100 \
  --batch 8 \
  --warmup 10

# With GPU delegate
jbang BenchmarkExample.java \
  --model model.litert \
  --delegate GPU \
  --threads 8

# Generate CSV report
jbang BenchmarkExample.java \
  --model model.litert \
  --output results.csv
```

**Example Output**:
```
Benchmark Configuration:
  Model:         model.litert
  Iterations:    50
  Batch Size:    1
  Warmup Runs:   10
  Delegate:      AUTO
  Threads:       4
  XNNPACK:       true

══════════════════════════════════════════════════════════
Benchmark 1: Single Inference Latency
══════════════════════════════════════════════════════════
  Test:            Single Inference
  Mean Latency:    42.35 ms
  Median Latency:  41.20 ms
  P50 Latency:     41.20 ms
  P95 Latency:     48.50 ms
  P99 Latency:     49.10 ms
  Min Latency:     38 ms
  Max Latency:     52 ms
  Std Deviation:   3.45 ms
  Throughput:      23.61 inf/sec
  Peak Memory:     45.67 MB

══════════════════════════════════════════════════════════
Benchmark Summary
══════════════════════════════════════════════════════════
  Single Inference                        42.35 ms (mean)     23.61 inf/sec
  Batch Inference (batch=8)               156.78 ms (mean)    51.03 inf/sec
  Async Inference (concurrency=8)         89.45 ms (mean)     11.18 inf/sec
  Sustained Load (60s)                    43.12 ms (mean)     23.19 inf/sec
```

**CSV Report Format**:
```csv
Test,Mean_ms,Median_ms,P50_ms,P95_ms,P99_ms,Min_ms,Max_ms,StdDev_ms,Throughput_inf_sec,Peak_Memory_B
Single Inference,42.35,41.20,41.20,48.50,49.10,38,52,3.45,23.61,47890432
Batch Inference (batch=8),156.78,154.30,154.30,168.90,172.40,145,178,8.92,51.03,52428800
```

---

## ⚙️ Configuration Options

### Global Options (All Examples)

| Property | Description | Default | Example |
|----------|-------------|---------|---------|
| `model` | Path to model file (`.litertlm` or `.litert`) | - | `-Dmodel=gemma.litertlm` |
| `delegate` | Hardware delegate (NONE, CPU, GPU, NNAPI, COREML, AUTO) | AUTO | `-Ddelegate=GPU` |
| `threads` | Number of CPU threads | 4 | `-Dthreads=8` |
| `xnnpack` | Enable XNNPACK optimization | true | `-Dxnnpack=false` |

### Example-Specific Options

#### ImageClassificationExample

| Property | Description | Default | Example |
|----------|-------------|---------|---------|
| `image` | Input image path (required) | - | `--image cat.jpg` |
| `topk` | Number of top predictions | 5 | `--topk 10` |

#### ObjectDetectionExample

| Property | Description | Default | Example |
|----------|-------------|---------|---------|
| `image` | Input image path (required) | - | `--image street.jpg` |
| `threshold` | Confidence threshold | 0.5 | `--threshold 0.6` |
| `output` | Output image path | auto-generated | `--output detected.jpg` |

#### TextGenerationExample

| Property | Description | Default | Example |
|----------|-------------|---------|---------|
| `prompt` | Input prompt | "Hello, how are you?" | `--prompt "Write a story"` |
| `maxtokens` | Max tokens to generate | 50 | `--maxtokens 100` |
| `temperature` | Sampling temperature (0.0-2.0) | 0.7 | `--temperature 0.8` |
| `topk` | Top-K sampling | 40 | `--topk 50` |
| `interactive` | Enable chat mode | false | `--interactive` |

#### BenchmarkExample

| Property | Description | Default | Example |
|----------|-------------|---------|---------|
| `iterations` | Number of iterations | 50 | `--iterations 100` |
| `batch` | Batch size | 1 | `--batch 8` |
| `warmup` | Warmup runs | 10 | `--warmup 20` |
| `output` | CSV output file | - | `--output results.csv` |

---

## 🔧 Hardware Delegates

### CPU (XNNPACK)
- **Optimized for**: ARM and x86 CPUs
- **Uses**: SIMD instructions (NEON, AVX)
- **Best for**: General inference on CPU

```bash
jbang -Ddelegate=CPU -Dmodel=model.litert LiteRTEdgeExample.java
```

### GPU (Metal/OpenCL)
- **iOS/macOS**: Metal delegate
- **Android**: OpenCL delegate
- **Best for**: High-throughput inference

```bash
jbang -Ddelegate=GPU -Dmodel=model.litert LiteRTEdgeExample.java
```

### NNAPI (Android only)
- **Android Neural Networks API**
- **Automatically selects**: Best accelerator (DSP, GPU, NPU)
- **Best for**: Android devices with hardware accelerators

```bash
jbang -Ddelegate=NNAPI -Dmodel=model.litert LiteRTEdgeExample.java
```

### CoreML (iOS/macOS only)
- **Apple Core ML framework**
- **Uses**: Apple Neural Engine (ANE)
- **Best for**: iOS/macOS devices with ANE

```bash
jbang -Ddelegate=COREML -Dmodel=model.litert LiteRTEdgeExample.java
```

---

## 📦 Model Preparation

### Converting TensorFlow Models to LiteRT

```python
import tensorflow as tf

# Convert SavedModel to LiteRT
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
litert_model = converter.convert()

# Save with .litert extension
with open('model.litert', 'wb') as f:
    f.write(litert_model)
```

### Downloading Pre-trained Models

```bash
# Gemma 4B for text generation (language model)
# Available from litert-community repository
model_name="gemma-4-E4B-it-litert-lm.litertlm"

# MobileNet V2 for image classification (vision model)
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v2_1.0_224.litert

# SSD MobileNet for object detection (vision model)
wget https://storage.googleapis.com/download.tensorflow.org/models/ssd_mobilenet_v2.litert
```

---

## 🚀 Performance Tips

1. **Use Hardware Delegates**: GPU or CoreML for 5-10x speedup
2. **Enable XNNPACK**: Optimized CPU inference (enabled by default)
3. **Use Memory Pool**: Reduces allocation overhead (enabled by default)
4. **Batch Inference**: Process multiple inputs together for better throughput
5. **Warm Up Models**: Run a dummy inference to initialize caches
6. **Monitor Metrics**: Track latency and memory usage with benchmark example
7. **Thread Count**: Match threads to available CPU cores (avoid over-subscription)
8. **Model Quantization**: Use quantized models for faster inference on edge devices

---

## 🐛 Troubleshooting

### JBang Dependency Resolution

**Problem**: JBang cannot resolve SNAPSHOT dependencies

**Solution**:
```bash
# Ensure SDK is built
cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests

# Clear JBang cache
jbang cache clear

# Run with fresh dependency resolution
jbang --fresh LiteRTEdgeExample.java
```

### Model Loading Fails

**Problem**: Model loading error

**Solutions**:
- ✅ Verify model file exists and has correct extension (`.litertlm` for language models, `.litert` for vision)
- ✅ Check file permissions: `ls -l model.litert`
- ✅ Ensure model is compatible with your LiteRT version
- ✅ Try loading with CLI first: `gollek litert load model.litert`

```bash
# Debug model file
file model.litert
ls -lh model.litert
```

### Inference Fails

**Problem**: Inference runtime error

**Solutions**:
- ✅ Verify input tensor size matches model input shape
- ✅ Check input data type (float32, uint8, etc.)
- ✅ Ensure output buffer is large enough
- ✅ Run with verbose logging: `jbang -Dverbose=true LiteRTEdgeExample.java`

```bash
# Check model input/output shapes
python3 -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='model.litert')
interpreter.allocate_tensors()
print('Input details:', interpreter.get_input_details())
print('Output details:', interpreter.get_output_details())
"
```

### Performance Issues

**Problem**: Slow inference times

**Solutions**:
- ✅ Enable hardware delegate: `-Ddelegate=GPU` or `-Ddelegate=COREML`
- ✅ Increase thread count: `-Dthreads=8` (match your CPU cores)
- ✅ Enable XNNPACK: `-Dxnnpack=true`
- ✅ Use batch inference for multiple inputs
- ✅ Run benchmark to identify bottlenecks: `./run.sh benchmark`

```bash
# Compare delegates
./run.sh edge --model model.litert --delegate CPU
./run.sh edge --model model.litert --delegate GPU
./run.sh edge --model model.litert --delegate COREML

# Run comprehensive benchmark
./run.sh benchmark --model model.litert --iterations 100
```

### Out of Memory

**Problem**: Memory allocation errors

**Solutions**:
- ✅ Reduce batch size
- ✅ Use smaller model (e.g., MobileNet V2 0.5 instead of 1.0)
- ✅ Enable memory pool with appropriate size
- ✅ Monitor memory usage with benchmark example

```bash
# Monitor memory usage
./run.sh benchmark --model model.litert --output memory_report.csv
```

### Java Version Issues

**Problem**: Java version errors

**Solutions**:
```bash
# Check Java version
java -version

# Must be Java 21 or higher
# Install with SDKMAN:
sdk install java 21.0.2-tem

# Or use Adoptium:
# https://adoptium.net/
```

---

## 📁 Project Structure

```
edge/
├── LiteRTEdgeExample.java          # Basic edge inference example
├── ImageClassificationExample.java # Image classification with ImageNet labels
├── ObjectDetectionExample.java     # Object detection with COCO labels
├── TextGenerationExample.java      # Text generation with language models ⭐ NEW
├── BenchmarkExample.java           # Performance benchmarking ⭐ NEW
├── run.sh                          # Enhanced runner script
└── README.md                       # This file
```

---

## 🧪 Running Tests

The examples include built-in validation and error handling. To test:

```bash
# Test demo modes (no models required)
./run.sh edge
./run.sh text
./run.sh benchmark

# Test with invalid inputs (should show error messages)
./run.sh image --model nonexistent.litert --image cat.jpg
./run.sh object --image street.jpg  # Missing model

# Test with real models
./run.sh image --model mobilenet_v2.litert --image test_images/cat.jpg
./run.sh benchmark --model mobilenet_v2.litert --iterations 10
```

---

## 📊 Example Comparison

| Example | Use Case | Model Type | Input | Output | Complexity |
|---------|----------|------------|-------|--------|------------|
| LiteRTEdgeExample | Core SDK demo | Any | Synthetic | Metrics | ⭐ |
| ImageClassificationExample | Image classification | Vision (.litert) | Image | Labels + probabilities | ⭐⭐ |
| ObjectDetectionExample | Object detection | Vision (.litert) | Image | Bounding boxes + labels | ⭐⭐⭐ |
| TextGenerationExample | Text generation | Language (.litertlm) | Text prompt | Generated text | ⭐⭐⭐ |
| BenchmarkExample | Performance testing | Any | Synthetic | Metrics + CSV | ⭐⭐⭐⭐ |

---

## 🤝 Contributing

When contributing new examples:

1. Follow the existing code style and structure
2. Add comprehensive error handling and validation
3. Include JBang dependency directives at the top
4. Update this README with usage instructions
5. Add the example to `run.sh`
6. Test both demo mode and real model mode

---

## 📄 Dependencies

All examples use JBang to automatically fetch dependencies:

- `tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT`
- `tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT`
- `tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT`
- Java 21+

---

## 📜 License

MIT License - Copyright (c) 2026 Kayys.tech

---

## 🆘 Support

- **GitHub Issues**: https://github.com/wayang-platform/gollek/issues
- **Documentation**: https://wayang-platform.github.io/gollek
- **Email**: team@wayang.dev
- **Discord**: https://discord.gg/wayang-platform (if available)

---

## 🗺️ Roadmap

Planned enhancements:

- [ ] Add semantic segmentation example
- [ ] Add pose estimation example
- [ ] Add image-to-image translation example
- [ ] Add multi-model pipeline example
- [ ] Add real-time video processing example
- [ ] Add model quantization example
- [ ] Add custom delegate integration
- [ ] Add ONNX model conversion example

---

**Happy Edge Computing! 🚀**
