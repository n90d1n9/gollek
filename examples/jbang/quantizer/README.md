# Gollek Quantizer JBang Examples

Examples demonstrating how to use Gollek's four quantization backends: GPTQ, AWQ, AutoRound, and TurboQuant.

## 🚀 Quick Start

### Prerequisites

- **Java 25 or higher** - [Download Adoptium](https://adoptium.net/)
- **JBang** - Install with: `curl -Ls https://sh.jbang.dev | bash -s -`
- **Built quantizer modules** - See below

### Building the Quantizer Modules

Before running the examples, build all quantizer modules:

```bash
# Build all quantizers
cd gollek/core/quantizer
mvn clean install \
  -pl gollek-quantizer-gptq,gollek-quantizer-awq,\
      gollek-quantizer-autoround,gollek-quantizer-turboquant \
  -am -DskipTests

# Build safetensor quantization
cd ../../plugins/runner/safetensor/gollek-safetensor-quantization
mvn clean install

# Build SDK API
cd ../../../sdk/lib/gollek-sdk-api
mvn clean install
```

### Running Examples

**Using JBang directly:**

```bash
# GPTQ quantization
jbang gollek-quantizer-gptq.java --demo
jbang gollek-quantizer-gptq.java --model /path/to/model.safetensors --output /path/to/quantized

# AWQ quantization
jbang gollek-quantizer-awq.java --demo
jbang gollek-quantizer-awq.java --model /path/to/awq-model.safetensors

# AutoRound quantization
jbang gollek-quantizer-autoround.java --demo
jbang gollek-quantizer-autoround.java --model /path/to/autoround-model.safetensors

# TurboQuant quantization
jbang gollek-quantizer-turboquant.java --detect /path/to/model.safetensors

# Quantizer comparison & benchmark
jbang gollek-quantizer-comparison.java --demo
jbang gollek-quantizer-comparison.java --benchmark --dimension 128 --iterations 1000
jbang gollek-quantizer-comparison.java --recommend "LLM serving"
```

---

## 📚 Available Examples

### 1. `gollek-quantizer-gptq.java` - GPTQ Quantization

**Purpose**: Demonstrates GPTQ (GPT Quantization) for highest quality 4-bit compression.

**Features**:
- ✅ Hessian-based per-layer optimization
- ✅ Configurable bits (2, 3, 4, 8), group size, dampening
- ✅ ActOrder for improved quality
- ✅ Symmetric and asymmetric quantization
- ✅ Demo mode with synthetic data
- ✅ Full CLI with picocli

**Usage**:

```bash
# Demo mode (no model required)
jbang gollek-quantizer-gptq.java --demo

# Quantize a real model
jbang gollek-quantizer-gptq.java \
  --model /path/to/model.safetensors \
  --output /path/to/quantized \
  --bits 4 \
  --group-size 128 \
  --act-order
```

**Example Output**:
```
Configuration:
  Model:         /path/to/model.safetensors
  Output:        /path/to/quantized
  Bits:          4
  Group Size:    128
  Dampening:     0.01
  ActOrder:      true

Starting GPTQ quantization...

✓ Quantization completed successfully!

Results:
  Input Tensors:     32
  Output Tensors:    32
  Input Size:        13.50 GB
  Output Size:       3.80 GB
  Compression Ratio: 3.55x
  Throughput:        45.2 MB/s
  Elapsed Time:      1523456ms
```

---

### 2. `gollek-quantizer-awq.java` - AWQ Quantization

**Purpose**: Demonstrates AWQ (Activation-Aware Weight Quantization) for fast 4-bit compression.

**Features**:
- ✅ Activation-aware channel protection
- ✅ GEMM, GEMV, and Marlin kernel formats
- ✅ Symmetric and asymmetric quantization
- ✅ Layer inspection and statistics
- ✅ Demo mode with configuration examples

**Usage**:

```bash
# Demo mode
jbang gollek-quantizer-awq.java --demo

# Load AWQ model
jbang gollek-quantizer-awq.java \
  --model /path/to/awq-model.safetensors \
  --bits 4 \
  --group-size 128 \
  --format GEMM
```

**Example Output**:
```
AWQ Configuration Examples:

1. Standard 4-bit AWQ (GEMM format):
   Bits:          4
   Group Size:    128
   Kernel Format: GEMM
   Has Zeros:     true
   Pack Factor:   8

AWQ vs GPTQ Comparison:
  ┌─────────────┬──────────┬──────────┬──────────┐
  │ Metric      │ GPTQ     │ AWQ      │ Winner   │
  ├─────────────┼──────────┼──────────┼──────────┤
  │ Quality     │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │ GPTQ     │
  │ Speed       │ ⭐⭐      │ ⭐⭐⭐⭐   │ AWQ      │
  └─────────────┴──────────┴──────────┴──────────┘
```

---

### 3. `gollek-quantizer-autoround.java` - AutoRound Quantization

**Purpose**: Demonstrates AutoRound quantization with optimization-based rounding (SignSGD).

**Features**:
- ✅ SignSGD optimization-based rounding
- ✅ Scale factor learning via Adam
- ✅ Multiple backend variants (ExLlama, Marlin, IPEX)
- ✅ FP32 and FP16 scale tensor support
- ✅ Algorithm overview in demo mode

**Usage**:

```bash
# Demo mode
jbang gollek-quantizer-autoround.java --demo

# Load AutoRound model
jbang gollek-quantizer-autoround.java \
  --model /path/to/autoround-model.safetensors \
  --bits 4 \
  --group-size 128 \
  --scale-dtype FLOAT32 \
  --pack-format AUTOROUND_NATIVE
```

**Example Output**:
```
AutoRound Algorithm Overview:

  AutoRound optimizes BOTH rounding decisions AND scale factors:

  For each transformer block B:
  1. Collect input activations X from calibration data
  2. Initialize s, z from min/max of each weight group
  3. For T iterations (default 200):
     a. q(W) = clamp(round(W/s + 0.5*v) − z, 0, 2^b − 1)
     b. W̃ = (q(W) + z) × s
     c. L = ||BX − B̃X||² (block reconstruction loss)
     d. Gradient step on s, z via Adam
     e. v update via SignSGD: v ← v − η × sign(∂L/∂v)
```

---

### 4. `gollek-quantizer-turboquant.java` - TurboQuant Quantization

**Purpose**: Demonstrates TurboQuant for edge-optimized, calibration-free quantization.

**Features**:
- ✅ Multi-format detection (GPTQ, AWQ, AutoRound, HQQ, SqueezeLLM, BnB)
- ✅ Model inspection and metadata
- ✅ MSE and Prod quantization modes
- ✅ KV cache compression

**Usage**:

```bash
# Detect quantization format
jbang gollek-quantizer-turboquant.java --detect /path/to/model.safetensors

# Inspect model metadata
jbang gollek-quantizer-turboquant.java --inspect /path/to/model.safetensors
```

---

### 5. `gollek-quantizer-comparison.java` - Quantizer Comparison & Benchmark ⭐ NEW

**Purpose**: Compare all 4 quantization backends and choose the right one for your use case.

**Features**:
- ✅ Detailed comparison matrix (quality, speed, memory, calibration)
- ✅ Use case-based recommendations
- ✅ Micro-benchmarks for TurboQuant
- ✅ Model-level performance estimates

**Usage**:

```bash
# Show comparison matrix
jbang gollek-quantizer-comparison.java --demo

# Run micro-benchmarks
jbang gollek-quantizer-comparison.java \
  --benchmark \
  --dimension 128 \
  --iterations 1000

# Get recommendation for your use case
jbang gollek-quantizer-comparison.java --recommend "LLM serving"
jbang gollek-quantizer-comparison.java --recommend "edge deployment"
jbang gollek-quantizer-comparison.java --recommend "KV cache compression"
```

**Example Output**:
```
Quantizer Comparison Matrix
══════════════════════════════════════════════════════════════════════

Metric          │ GPTQ         │ AWQ          │ AutoRound    │ TurboQuant
──────────────────────────────────────────────────────────────────────────
Algorithm       │ Hessian      │ Scaling      │ SignSGD      │ RandRot
Quality         │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐      │ ⭐⭐⭐
Speed           │ ⭐⭐         │ ⭐⭐⭐⭐      │ ⭐⭐⭐        │ ⭐⭐⭐⭐⭐
Memory          │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐
Calibration     │ Required     │ Required     │ Required     │ None
4-bit Time*     │ ~25 min      │ ~5 min       │ ~15 min      │ ~2 min
Compression     │ 3.5x         │ 3.5x         │ 3.5x         │ 4-8x
Quality Loss    │ <1%          │ 1-2%         │ ~1%          │ 2-5%
Best For        │ Quality      │ Speed        │ Balance      │ Edge
Hardware        │ CPU/GPU      │ CPU/GPU      │ CPU/GPU      │ CPU/Edge
Tests Pass      │ 16/16        │ 19/19        │ 39/39        │ 74/74
```

---

## 🎯 Quantizer Selection Guide

### By Use Case

| Use Case | Recommended Quantizer | Why |
|----------|----------------------|-----|
| **Production LLM serving** | GPTQ | Best quality, minimal accuracy loss |
| **Fast model conversion** | AWQ | 3x faster than GPTQ |
| **Balanced quality/speed** | AutoRound | Optimization-based, good middle ground |
| **Edge/mobile deployment** | TurboQuant | Calibration-free, SIMD optimized |
| **KV cache compression** | TurboQuant | 2.5-bit effective with outlier splitting |
| **Inner product search** | TurboQuant_prod | Unbiased inner product estimation |

### By Hardware

| Hardware | Recommended Quantizer | Expected Speedup |
|----------|----------------------|------------------|
| **CPU (x86)** | GPTQ, TurboQuant | 4-8x memory reduction |
| **CPU (ARM)** | TurboQuant | SIMD vectorized (NEON) |
| **NVIDIA GPU** | GPTQ, AWQ, AutoRound | Full GPU acceleration |
| **Edge TPU** | TurboQuant | Calibration-free, low memory |
| **Apple Silicon** | TurboQuant | Hadamard rotation (fast WHT) |

---

## 📊 Test Results

All quantizer modules are fully tested and production-ready:

| Module | Tests | Status |
|--------|-------|--------|
| gollek-quantizer-gptq | 16/16 | ✅ 100% |
| gollek-quantizer-awq | 19/19 | ✅ 100% |
| gollek-quantizer-autoround | 39/39 | ✅ 100% |
| gollek-quantizer-turboquant | 74/74 | ✅ 100% |
| Integration Tests | 12/12 | ✅ 100% |

**Total**: 160/160 tests passing (100%)

---

## 🔧 Configuration Options

### GPTQ Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bits` | 4 | Quantization bits (2, 3, 4, 8) |
| `--group-size` | 128 | Group size for quantization |
| `--damp` | 0.01 | Dampening percentage (0.0-1.0) |
| `--act-order` | false | Use activation ordering |
| `--symmetric` | false | Use symmetric quantization |
| `--num-samples` | 128 | Number of calibration samples |
| `--seq-len` | 2048 | Sequence length for calibration |

### AWQ Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bits` | 4 | Quantization bits |
| `--group-size` | 128 | Group size |
| `--format` | GEMM | Kernel format (GEMM, GEMV, MARLIN) |
| `--symmetric` | false | Use symmetric quantization |

### AutoRound Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bits` | 4 | Quantization bits (2, 3, 4, 8) |
| `--group-size` | 128 | Group size |
| `--scale-dtype` | FLOAT32 | Scale tensor dtype |
| `--pack-format` | AUTOROUND_NATIVE | Packing format |

### Comparison Options

| Option | Default | Description |
|--------|---------|-------------|
| `--demo` | false | Show comparison matrix |
| `--benchmark` | false | Run micro-benchmarks |
| `--dimension` | 128 | Vector dimension for benchmarks |
| `--iterations` | 1000 | Number of benchmark iterations |
| `--recommend` | - | Get recommendation for use case |

---

## 🐛 Troubleshooting

### NoClassDefFoundError: jdk/incubator/vector/Vector

**Solution**: Ensure you're using Java 25+ and the Vector API module is available:

```bash
java --add-modules=jdk.incubator.vector --version
```

### Module Not Found

**Solution**: Build the quantizer modules first:

```bash
cd gollek/core/quantizer
mvn clean install -pl gollek-quantizer-gptq -am -DskipTests
```

### Model Loading Fails

**Solution**: Verify model file exists and has correct format:

```bash
# Check file exists
ls -lh /path/to/model.safetensors

# Detect format
jbang gollek-quantizer-turboquant.java --detect /path/to/model.safetensors
```

---

## 📁 Project Structure

```
quantizer/
├── gollek-quantizer-gptq.java          # GPTQ quantization example
├── gollek-quantizer-awq.java           # AWQ quantization example
├── gollek-quantizer-autoround.java     # AutoRound quantization example
├── gollek-quantizer-turboquant.java    # TurboQuant quantization example
├── gollek-quantizer-comparison.java    # Comparison & benchmark ⭐ NEW
└── README.md                           # This file
```

---

## 📖 Resources

- **Quantization Documentation**: https://gollek-ai.github.io/docs/plugins/quantization
- **Integration Guide**: [QUANTIZER_INTEGRATION_GUIDE.md](https://github.com/gollek-ai/gollek/blob/main/QUANTIZER_INTEGRATION_GUIDE.md)
- **Research Papers**:
  - [GPTQ Paper](https://arxiv.org/abs/2210.17323)
  - [AWQ Paper](https://arxiv.org/abs/2306.00978)
  - [AutoRound Paper](https://arxiv.org/abs/2309.05516)
  - [TurboQuant Paper](https://arxiv.org/abs/2504.19874)

---

## 🤝 Contributing

When contributing new examples:

1. Follow the existing code style and structure
2. Add comprehensive error handling and validation
3. Include JBang dependency directives at the top
4. Update this README with usage instructions
5. Test both demo mode and real model mode

---

**Happy Quantizing! 🚀**
