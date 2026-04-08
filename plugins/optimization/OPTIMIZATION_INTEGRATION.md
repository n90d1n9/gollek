# GPU Kernel Optimization Integration Guide

This document describes how the Gollek GPU kernel modules integrate with the optimization extensions for maximum inference performance.

---

## Overview

Each GPU kernel module (CUDA, Blackwell, ROCm, Metal) now integrates with platform-specific optimization managers that automatically select and execute the optimal attention kernel based on hardware capabilities.

```
┌─────────────────────────────────────────────────────────┐
│                    KernelRunner                         │
│  (CudaRunner / BlackwellRunner / RocmRunner / MetalRunner) │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              OptimizationManager                        │
│  (Auto-selects best kernel for current GPU)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ FlashAttn-4  │  │ FlashAttn-3  │  │ FlashAttn-2  │  │
│  │ (Blackwell)  │  │ (Hopper+)    │  │ (Ampere+)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Native Kernel Modules                      │
│  libgollek_fa4.so | libgollek_fa3.so | libgollek_rocm.so │
└─────────────────────────────────────────────────────────┘
```

## GGUF (llama.cpp) Note

GGUF uses llama.cpp for inference and does not directly invoke Gollek's GPU
kernel optimization modules. When optimization extensions are present on the
classpath, the GGUF provider will advertise their availability in its
capability feature flags (for UI/engine visibility only).

---

## Optimization Modules

## Optimization Compatibility Matrix

| Optimization Module | Scope | GGUF (llama.cpp) | Kernel Runners (CUDA/ROCm/Blackwell/Metal) |
|---------------------|-------|-----------------|-------------------------------------------|
| `gollek-ext-prompt-cache` | Engine + KV cache | Detectable only | Supported |
| `gollek-ext-kv-cache` | Kernel KV memory | Detectable only | Supported |
| `gollek-ext-paged-attention` | Kernel attention | Detectable only | Supported |
| `gollek-ext-prefilldecode` | Kernel prefill/decode | Detectable only | Supported |
| `gollek-ext-hybridattn` | Kernel attention | Detectable only | Supported |
| `gollek-ext-fa3` | Kernel attention | Detectable only | Supported |
| `gollek-ext-fa4` | Kernel attention | Detectable only | Supported |

Notes:
* “Detectable only” means GGUF will expose feature flags but does not execute these kernels.
* Kernel runners are the expected integration path for these modules.

### FlashAttention-4 (`gollek-ext-fa4`)

**Target:** NVIDIA Blackwell (B100, B200, GB200)

**Key Features:**
- TMEM (Tensor Memory) accumulation
- FP4 tensor core acceleration
- Async UMMA pipelines
- Software exp() on FMA units

**Performance:** 1,613 TFLOPs/s (71% peak) on B200

**Integration:**
```java
// In BlackwellRunner.initialize()
this.optimizationManager = new BlackwellOptimizationManager(
    blackwell, kvCacheManager, 64L * 1024 * 1024, true);

// In forward pass
optimizationManager.executeFlashAttention4(
    output, query, kPool, vPool,
    batchSize, seqLen, numHeads, numHeadsKV, headDim,
    scale, causal, useFp4);
```

---

### FlashAttention-3 (`gollek-ext-fa3`)

**Target:** NVIDIA Hopper (H100, H200) and AMD MI300X

**Key Features:**
- FP8 tensor core support
- Improved memory tiling
- Multi-CTA MMA backward

**Performance:** 950 TFLOPs/s on H100

**Integration:**
```java
// In CudaRunner or RocmRunner
if (computeCapability >= 90 || gpuArch >= 942) {
    optimizationManager.executeAttention(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal, useFp8);
}
```

---

### Hybrid Attention + GDN (`gollek-ext-hybridattn`)

**Target:** All platforms with custom architecture support

**Key Features:**
- Gated Delta Network (GDN) recurrent layers
- Hybrid H1/H2 attention schedules
- 99.6% accuracy on MQAR retrieval

**Integration:**
```java
// In HybridAttentionGdnRunner
boolean[] layerSchedule = buildLayerSchedule("H2", numLayers);
optimizationManager.executeHybridAttention(
    layerSchedule, state, output, query, kPool, vPool,
    batchSize, seqLen, numHeads, numHeadsKV, headDim,
    scale, causal, useFp4);
```

---

## Platform-Specific Integration

### CUDA (A100/H100/H200)

**Optimization Manager:** `CudaOptimizationManager`

**Auto-Selection Logic:**
```java
if (computeCapability >= 100) {
    // Blackwell: FA4 with TMEM
    executeFlashAttention4();
} else if (computeCapability >= 90) {
    // Hopper: FA3 with FP8
    executeFlashAttention3();
} else if (computeCapability >= 80) {
    // Ampere: FA2
    executeFlashAttention2();
} else {
    // Fallback: Paged attention
    executePagedAttention();
}
```

**Configuration:**
```properties
gollek.runners.cuda.enabled=true
gollek.runners.cuda.device-id=0
gollek.runners.cuda.use-fp8=true  # Auto-enabled on H100+
```

---

### Blackwell (B100/B200/GB200)

**Optimization Manager:** `BlackwellOptimizationManager`

**TMEM Integration:**
```java
// Allocate 64MB TMEM for FA4
MemorySegment tmem = blackwell.tmemAlloc(64L * 1024 * 1024);

// Execute with TMEM acceleration
optimizationManager.executeFlashAttention4(
    output, query, kPool, vPool,
    batchSize, seqLen, numHeads, numHeadsKV, headDim,
    scale, causal, useFp4);  // FP4 = 2x speedup
```

**Async Execution:**
```java
// Overlap weight copy with FA4 computation
optimizationManager.executeAsyncFlashAttention4(
    stream, weightsNext, dWeightsNext, weightBytes,
    output, query, kPool, vPool,
    batchSize, seqLen, numHeads, numHeadsKV, headDim,
    scale, causal, useFp4);
```

**Configuration:**
```properties
gollek.runners.blackwell.enabled=true
gollek.runners.blackwell.use-fp4=true
gollek.runners.blackwell.use-tmem=true
gollek.runners.blackwell.tmem-size-mb=64
```

---

### ROCm (MI300X/MI250X)

**Optimization Manager:** `RocmOptimizationManager`

**Unified Memory (MI300X):**
```java
if (isUnifiedMemory) {
    // Zero-copy access to KV cache
    optimizationManager.executeWithUnifiedMemory(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
} else {
    // Discrete GPU (MI250X)
    optimizationManager.executeAttention(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
}
```

**Configuration:**
```properties
gollek.runners.rocm.enabled=true
gollek.runners.rocm.device-id=0
gollek.runners.rocm.use-managed-memory=true  # MI300X only
gollek.runners.rocm.use-fp8=true  # MI300X only
```

---

### Metal (M1/M2/M3/M4)

**Optimization Manager:** `MetalOptimizationManager`

**MPSGraph SDPA (macOS 14+):**
```java
if (isSdpaAvailable) {
    // FA4 equivalent via MPSGraph
    optimizationManager.executeMpsGraphSdpa(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
} else {
    // Fallback to separate matmuls
    optimizationManager.executeSeparateMatmuls(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
}
```

**Weight Offloading:**
```java
if (modelSizeGB > unifiedMemoryGB * 0.5) {
    // Stream weights from CPU to GPU
    optimizationManager.executeWithWeightOffloading(
        output, query, weightsCpu, weightsGpu, weightBytes,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
}
```

**Configuration:**
```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=auto  # Auto-select standard/offload
gollek.runners.metal.library-path=~/.gollek/libs/libgollek_metal.dylib
```

---

## Performance Comparison

### Llama-3.2-3B (tokens/sec)

| Platform | Kernel | Precision | Tokens/sec |
|----------|--------|-----------|------------|
| B200 | FA4+TMEM | FP4 | 180 |
| B200 | FA4+TMEM | FP8 | 135 |
| H100 | FA3 | FP8 | 95 |
| A100 | FA2 | FP16 | 45 |
| MI300X | FA3 | FP8 | 65 |
| M3 Max | MPSGraph SDPA | BF16 | 50 |

### Llama-3-70B (tokens/sec)

| Platform | Kernel | Precision | Tokens/sec |
|----------|--------|-----------|------------|
| B200 | FA4+TMEM | FP4 | 55 |
| B200 | FA4+TMEM | FP8 | 40 |
| H100 | FA3 | FP8 | 28 |
| A100 80GB | FA2 | FP16 | 12 |
| MI300X | FA3 | FP8 | 32 |
| M3 Max 128GB | MPSGraph SDPA | BF16 | 18 |

---

## Kernel Selection Flow

```
GPU Detection
      ↓
Compute Capability Check
      ↓
┌─────────────────┐
│ sm_100+ (B200)  │ → FA4 + TMEM + FP4
├─────────────────┤
│ sm_90+ (H100)   │ → FA3 + FP8
├─────────────────┤
│ sm_80+ (A100)   │ → FA2
├─────────────────┤
│ gfx942 (MI300X) │ → FA3 + FP8 (unified mem)
├─────────────────┤
│ gfx90a (MI250X) │ → FA2
├─────────────────┤
│ Apple M-series  │ → MPSGraph SDPA or separate matmuls
└─────────────────┘
```

---

## Building with Optimizations

### Prerequisites

| Module | Requirements |
|--------|--------------|
| FA4 | CUDA 12.8+, sm_100a |
| FA3 | CUDA 12.3+, sm_90 |
| FA2 | CUDA 11.0+, sm_80 |
| ROCm FA3 | ROCm 6.0+, gfx942 |
| Metal SDPA | macOS 14+, Xcode |

### Build Commands

```bash
# FlashAttention-4 (Blackwell)
make -C inference-gollek/extension/optimization/gollek-ext-fa4 \
    CUDA_ARCH=sm_100a USE_FP4=1

# FlashAttention-3 (Hopper)
make -C inference-gollek/extension/optimization/gollek-ext-fa3 \
    CUDA_ARCH=sm_90 USE_FP8=1

# ROCm FlashAttention (MI300X)
make -C inference-gollek/extension/optimization/gollek-ext-fa3 \
    AMDGPU_TARGET=gfx942 USE_FP8=1

# All optimizations
mvn clean install -pl inference-gollek/extension/optimization
```

---

## Testing

### Unit Tests

```bash
# CUDA optimization tests
mvn test -Dtest=CudaOptimizationManagerTest \
  -pl inference-gollek/extension/kernel/gollek-kernel-cuda

# Blackwell TMEM tests
mvn test -Dtest=BlackwellOptimizationManagerTest \
  -pl inference-gollek/extension/kernel/gollek-kernel-blackwell

# GPU smoke tests (requires GPU)
CUDA_VISIBLE_DEVICES=0 mvn test -Pcuda-gpu-tests
```

### Benchmark Tests

```bash
# FA4 vs FA3 comparison
java -jar benchmark.jar \
  --runner blackwell \
  --model llama-3-70b \
  --kernel fa4,fa3 \
  --precision fp4,fp8

# Unified memory vs discrete
java -jar benchmark.jar \
  --runner rocm \
  --model llama-3-70b \
  --device mi300x,mi250x
```

---

## Troubleshooting

### FA4 Not Available

**Symptom:** `FlashAttention4Binding not available`

**Solution:**
```bash
# Verify CUDA version
nvcc --version  # Must be 12.8+

# Verify GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv  # Must be 10.0+

# Rebuild FA4
make -C inference-gollek/extension/optimization/gollek-ext-fa4 \
    CUDA_ARCH=sm_100a
```

### TMEM Allocation Failed

**Symptom:** `Could not allocate TMEM`

**Solution:**
```bash
# Check if running on Blackwell
nvidia-smi  # Should show B100/B200/GB200

# Reduce TMEM size
gollek.runners.blackwell.tmem-size-mb=32  # Instead of 64
```

### MPSGraph SDPA Not Available

**Symptom:** `isSdpaAvailable() = false`

**Solution:**
```bash
# Check macOS version
sw_vers -productVersion  # Must be 14.0+

# Check chip
sysctl -n machdep.cpu.brand_string  # Must be M-series
```

---

## API Reference

### Optimization Manager Interface

```java
public interface GpuOptimizationManager {
    void executeAttention(
        MemorySegment output,
        MemorySegment query,
        MemorySegment kPool,
        MemorySegment vPool,
        int batchSize,
        int seqLen,
        int numHeads,
        int numHeadsKV,
        int headDim,
        float scale,
        boolean causal,
        boolean useFp8);
    
    String getRecommendedKernel();
    String getRecommendedPrecision();
    double getExpectedThroughput(double modelSize);
}
```

### Kernel Metadata

```java
// Query kernel selection
String kernel = optimizationManager.getRecommendedKernel();
// Returns: "FA4-TMEM-FP4", "FA3", "FA2", "MPSGraph-SDPA", etc.

// Query expected performance
double tokensPerSec = optimizationManager.getExpectedThroughput(70.0);
// Returns expected tokens/sec for 70B model
```

---

## Configuration Reference

For complete configuration options and performance profiles, see the **[Optimization Configuration Guide](OPTIMIZATION_CONFIGURATION.md)**.

### Quick Configuration Reference

#### CUDA (A100/H100/B200)

```properties
gollek.runners.cuda.enabled=true
gollek.runners.cuda.use-fp8=true      # Auto on H100+
gollek.runners.cuda.use-fp4=true      # Auto on B200
gollek.runners.cuda.use-unified-memory=true
```

#### Blackwell (B100/B200/GB200)

```properties
gollek.runners.blackwell.enabled=true
gollek.runners.blackwell.use-tmem=true    # 64MB on-chip
gollek.runners.blackwell.use-fp4=true     # 2x speedup
gollek.runners.blackwell.async-copy=true
```

#### ROCm (MI300X/MI250X)

```properties
gollek.runners.rocm.enabled=true
gollek.runners.rocm.use-managed-memory=true  # MI300X zero-copy
gollek.runners.rocm.use-fp8=true             # MI300X FP8 cores
```

#### Metal (M1/M2/M3/M4)

```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=auto          # Auto standard/offload
gollek.runners.metal.use-sdpa=true      # MPSGraph FA4 equiv
gollek.runners.metal.use-bf16=true      # M3/M4 BF16 support
```

---

## Resources

- [FlashAttention-4 Paper](https://arxiv.org/abs/2603.05451)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691)
- [Hybrid Attention Paper](https://arxiv.org/abs/2412.06464)
- [CUDA Kernel Docs](../gollek-kernel-cuda/README.md)
- [Blackwell Kernel Docs](../gollek-kernel-blackwell/README.md)

---

[Back to GPU Kernels](/docs/gpu-kernels) &nbsp; [View Architecture](/docs/architecture)
