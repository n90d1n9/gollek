# Gollek Optimization Configuration Guide

Complete reference for configuring GPU kernel optimizations across all platforms.

---

## Quick Reference

| Platform | Config File | Key Properties |
|----------|-------------|----------------|
| **CUDA** | `application.properties` | `gollek.runners.cuda.*` |
| **Blackwell** | `application.properties` | `gollek.runners.blackwell.*` |
| **ROCm** | `application.properties` | `gollek.runners.rocm.*` |
| **Metal** | `application.properties` | `gollek.runners.metal.*` |

> GGUF note: these optimization configs apply to kernel runners. GGUF uses
> llama.cpp and only advertises optimization availability when extensions are
> present.

---

## CUDA Optimization Configuration

### Basic Configuration

```properties
# Enable CUDA runner
gollek.runners.cuda.enabled=true

# Auto-select optimal kernel based on GPU
gollek.runners.cuda.mode=auto

# CUDA library path
gollek.runners.cuda.library-path=/usr/local/cuda/lib64/libgollek_cuda.so

# Device selection (0-based index)
gollek.runners.cuda.device-id=0
```

### Precision Configuration

```properties
# Enable FP8 on H100+ (auto-detected)
gollek.runners.cuda.use-fp8=true

# Enable FP4 on Blackwell (auto-detected)
gollek.runners.cuda.use-fp4=true

# Force BF16 precision (fallback if not supported)
gollek.runners.cuda.prefer-bf16=true
```

### Memory Configuration

```properties
# Enable unified memory on A100/H100/B200 (zero-copy)
gollek.runners.cuda.use-unified-memory=true

# Maximum GPU memory usage (percentage)
gollek.runners.cuda.max-memory-percent=90

# Pre-allocate KV cache blocks
gollek.runners.cuda.kv-cache-blocks=1024

# KV cache block size (tokens per block)
gollek.runners.cuda.block-size=64
```

### Attention Kernel Configuration

```properties
# FlashAttention configuration
gollek.runners.cuda.use-flash-attention=true

# Attention softmax scale (auto-calculated if not set)
# gollek.runners.cuda.attn-scale=0.088  # 1/sqrt(128)

# Causal masking (true for decoder-only models)
gollek.runners.cuda.causal=true
```

### Performance Tuning

```properties
# Maximum batch size
gollek.runners.cuda.max-batch-size=32

# Number of concurrent streams
gollek.runners.cuda.num-streams=4

# Enable async memory copies
gollek.runners.cuda.async-copy=true

# Enable kernel fusion (where supported)
gollek.runners.cuda.kernel-fusion=true
```

### Model-Specific Configuration

#### Llama-3.2-3B (A100/H100)

```properties
gollek.runners.cuda.enabled=true
gollek.runners.cuda.use-fp8=true
gollek.runners.cuda.use-unified-memory=true
gollek.runners.cuda.num-layers=28
gollek.runners.cuda.num-heads=16
gollek.runners.cuda.num-heads-kv=8
gollek.runners.cuda.head-dim=128
gollek.runners.cuda.model-dim=3072
gollek.runners.cuda.ffn-dim=8192
gollek.runners.cuda.vocab-size=128256
```

#### Llama-3-70B (H100/B200)

```properties
gollek.runners.cuda.enabled=true
gollek.runners.cuda.use-fp8=true
gollek.runners.cuda.use-unified-memory=true
gollek.runners.cuda.num-layers=80
gollek.runners.cuda.num-heads=64
gollek.runners.cuda.num-heads-kv=8
gollek.runners.cuda.head-dim=128
gollek.runners.cuda.model-dim=8192
gollek.runners.cuda.ffn-dim=28672
gollek.runners.cuda.vocab-size=128256
```

---

## Blackwell Optimization Configuration

### Basic Configuration

```properties
# Enable Blackwell runner
gollek.runners.blackwell.enabled=true

# Auto-select optimal configuration
gollek.runners.blackwell.mode=auto

# CUDA library path
gollek.runners.blackwell.library-path=/usr/local/cuda/lib64/libgollek_blackwell.so

# Device selection
gollek.runners.blackwell.device-id=0
```

### TMEM Configuration

```properties
# Enable TMEM for FlashAttention-4 (64MB on-chip)
gollek.runners.blackwell.use-tmem=true

# TMEM size in MB (default: 64, max: 64)
gollek.runners.blackwell.tmem-size-mb=64

# TMEM block size for attention tiling
gollek.runners.blackwell.tmem-block-size=128
```

### FP4 Configuration

```properties
# Enable FP4 tensor cores (2x speedup over FP8)
gollek.runners.blackwell.use-fp4=true

# FP4 calibration (auto-calibrated if not set)
# gollek.runners.blackwell.fp4-scale=1.0

# Fallback to FP8 if FP4 not available
gollek.runners.blackwell.fp8-fallback=true
```

### Async Execution Configuration

```properties
# Enable async copy/compute overlap
gollek.runners.blackwell.async-copy=true

# Number of async streams
gollek.runners.blackwell.num-async-streams=4

# Async semaphore timeout (ms)
gollek.runners.blackwell.semaphore-timeout-ms=1000
```

### Memory Configuration

```properties
# Enable unified memory (B200 has 180GB shared)
gollek.runners.blackwell.use-unified-memory=true

# Pre-fetch next layer weights
gollek.runners.blackwell.prefetch-layers=2

# Weight offloading threshold (GB)
gollek.runners.blackwell.offload-threshold-gb=100
```

### Performance Tuning

```properties
# Maximum batch size (B200 can handle larger batches)
gollek.runners.blackwell.max-batch-size=64

# Enable tensor core acceleration
gollek.runners.blackwell.enable-tensor-cores=true

# Enable TMEM accumulation for attention
gollek.runners.blackwell.tmem-accumulation=true

# Kernel launch coalescing
gollek.runners.blackwell.kernel-coalescing=true
```

### Model-Specific Configuration

#### Llama-3-70B (B200 Optimal)

```properties
gollek.runners.blackwell.enabled=true
gollek.runners.blackwell.use-tmem=true
gollek.runners.blackwell.use-fp4=true
gollek.runners.blackwell.async-copy=true
gollek.runners.blackwell.num-layers=80
gollek.runners.blackwell.num-heads=64
gollek.runners.blackwell.num-heads-kv=8
gollek.runners.blackwell.head-dim=128
gollek.runners.blackwell.model-dim=8192
gollek.runners.blackwell.ffn-dim=28672
```

---

## ROCm Optimization Configuration

### Basic Configuration

```properties
# Enable ROCm runner
gollek.runners.rocm.enabled=true

# HIP library path
gollek.runners.rocm.library-path=/opt/rocm/lib/libamdhip64.so

# Kernel path (architecture-specific)
gollek.runners.rocm.kernel-path=/opt/gollek/lib/gollek_rocm_gfx942.hsaco

# Device selection
gollek.runners.rocm.device-id=0
```

### Architecture Configuration

```properties
# GPU architecture (auto-detected if not set)
# gfx942 = MI300X, gfx90a = MI250X, gfx1100 = RX 7900 XTX
gollek.runners.rocm.gpu-arch=gfx942

# Build kernel for specific architecture
gollek.runners.rocm.build-target=gfx942
```

### Unified Memory Configuration (MI300X)

```properties
# Enable managed memory (zero-copy on MI300X)
gollek.runners.rocm.use-managed-memory=true

# Pre-fetch data to GPU
gollek.runners.rocm.prefetch-to-gpu=true

# Enable CPU atomic access to GPU memory
gollek.runners.rocm.cpu-atomic-access=true
```

### Precision Configuration

```properties
# Enable FP8 on MI300X
gollek.runners.rocm.use-fp8=true

# FP8 scaling factor (auto-calculated if not set)
# gollek.runners.rocm.fp8-scale=1.0

# Use BF16 for accumulation
gollek.runners.rocm.bf16-accumulation=true
```

### Attention Kernel Configuration

```properties
# FlashAttention version (auto-selected)
# fa3 = MI300X, fa2 = MI250X, paged = fallback
gollek.runners.rocm.flash-attention=auto

# Attention softmax scale
# gollek.runners.rocm.attn-scale=0.088
```

### Performance Tuning

```properties
# Maximum batch size
gollek.runners.rocm.max-batch-size=32

# Number of compute units to use
gollek.runners.rocm.num-compute-units=0  # 0 = all available

# Wavefront size (64 for AMD)
gollek.runners.rocm.wavefront-size=64

# Enable LDS (Local Data Share) optimization
gollek.runners.rocm.lds-optimization=true
```

### Model-Specific Configuration

#### Llama-3-70B (MI300X Optimal)

```properties
gollek.runners.rocm.enabled=true
gollek.runners.rocm.use-managed-memory=true
gollek.runners.rocm.use-fp8=true
gollek.runners.rocm.gpu-arch=gfx942
gollek.runners.rocm.num-layers=80
gollek.runners.rocm.num-heads=64
gollek.runners.rocm.num-heads-kv=8
gollek.runners.rocm.head-dim=128
gollek.runners.rocm.model-dim=8192
```

---

## Metal Optimization Configuration

### Basic Configuration

```properties
# Enable Metal runner
gollek.runners.metal.enabled=true

# Runner mode (auto-selects based on model size)
# auto | standard | offload | force | disabled
gollek.runners.metal.mode=auto

# Library path
gollek.runners.metal.library-path=~/.gollek/libs/libgollek_metal.dylib
```

### Runner Mode Configuration

```properties
# AUTO: Automatically select based on model size vs memory
# - Model < 50% of RAM → standard
# - Model ≥ 50% of RAM → offload
gollek.runners.metal.mode=auto

# STANDARD: Load full model to unified memory
# Best for: M2 Max+ with 32GB+ RAM
gollek.runners.metal.mode=standard

# OFFLOAD: Stream weights from CPU to GPU
# Best for: M1/M2 with <32GB RAM, large models
gollek.runners.metal.mode=offload
```

### MPSGraph Configuration (macOS 14+)

```properties
# Enable MPSGraph SDPA (FlashAttention-4 equivalent)
gollek.runners.metal.use-sdpa=true

# Enable BF16 on M3/M4 chips
gollek.runners.metal.use-bf16=true

# MPSGraph optimization level (0-3)
gollek.runners.metal.mps-opt-level=3
```

### Weight Offloading Configuration

```properties
# Offload batch size (layers loaded at once)
gollek.runners.metal.offload-batch-size=4

# Enable async weight streaming
gollek.runners.metal.async-streaming=true

# CPU-GPU transfer buffer size (MB)
gollek.runners.metal.transfer-buffer-mb=256
```

### Memory Configuration

```properties
# Reserve system memory (GB)
gollek.runners.metal.reserve-memory-gb=2

# Enable purgeable memory for weights
gollek.runners.metal.purgeable-memory=true

# Memory pressure threshold (0.0-1.0)
gollek.runners.metal.memory-pressure-threshold=0.8
```

### Performance Tuning

```properties
# Maximum batch size
gollek.runners.metal.max-batch-size=16

# Enable AMX acceleration (M3/M4)
gollek.runners.metal.enable-amx=true

# Tile size for matrix operations
gollek.runners.metal.tile-size=32

# Enable kernel fusion
gollek.runners.metal.kernel-fusion=true
```

### Chip-Specific Configuration

#### M3 Max (40 GPU cores, 128GB)

```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=standard
gollek.runners.metal.use-sdpa=true
gollek.runners.metal.use-bf16=true
gollek.runners.metal.enable-amx=true
gollek.runners.metal.max-batch-size=32
```

#### M2 (8GB, Memory Constrained)

```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=offload
gollek.runners.metal.offload-batch-size=2
gollek.runners.metal.async-streaming=true
gollek.runners.metal.reserve-memory-gb=1
```

---

## Cross-Platform Optimization

### Environment Variables

```bash
# CUDA/Blackwell
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ROCm
export ROCR_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Metal
export METAL_DEVICE_WRAPPER_TYPE=1  # Enable Metal validation
```

### YAML Configuration (application.yaml)

```yaml
gollek:
  runners:
    cuda:
      enabled: true
      mode: auto
      device-id: 0
      use-fp8: true
      use-unified-memory: true
      max-batch-size: 32
      
    blackwell:
      enabled: true
      mode: auto
      device-id: 0
      use-tmem: true
      use-fp4: true
      async-copy: true
      
    rocm:
      enabled: true
      device-id: 0
      gpu-arch: gfx942
      use-managed-memory: true
      use-fp8: true
      
    metal:
      enabled: true
      mode: auto
      library-path: ~/.gollek/libs/libgollek_metal.dylib
      use-sdpa: true
      use-bf16: true
```

---

## Performance Profiles

### Maximum Throughput (B200)

```properties
gollek.runners.blackwell.enabled=true
gollek.runners.blackwell.use-tmem=true
gollek.runners.blackwell.use-fp4=true
gollek.runners.blackwell.async-copy=true
gollek.runners.blackwell.max-batch-size=64
gollek.runners.blackwell.kernel-fusion=true
gollek.runners.blackwell.tmem-accumulation=true
```

**Expected:** 180 tokens/sec (7B), 55 tokens/sec (70B)

### Balanced Performance (H100)

```properties
gollek.runners.cuda.enabled=true
gollek.runners.cuda.use-fp8=true
gollek.runners.cuda.use-unified-memory=true
gollek.runners.cuda.max-batch-size=32
gollek.runners.cuda.async-copy=true
```

**Expected:** 95 tokens/sec (7B), 28 tokens/sec (70B)

### Memory Efficient (MI300X)

```properties
gollek.runners.rocm.enabled=true
gollek.runners.rocm.use-managed-memory=true
gollek.runners.rocm.use-fp8=true
gollek.runners.rocm.gpu-arch=gfx942
gollek.runners.rocm.max-batch-size=32
```

**Expected:** 65 tokens/sec (7B), 32 tokens/sec (70B)

### Apple Silicon (M3 Max)

```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=standard
gollek.runners.metal.use-sdpa=true
gollek.runners.metal.use-bf16=true
gollek.runners.metal.enable-amx=true
```

**Expected:** 55 tokens/sec (7B), 20 tokens/sec (70B)

### Low Memory (8GB Mac)

```properties
gollek.runners.metal.enabled=true
gollek.runners.metal.mode=offload
gollek.runners.metal.offload-batch-size=2
gollek.runners.metal.async-streaming=true
gollek.runners.metal.reserve-memory-gb=1
```

**Expected:** 18 tokens/sec (7B), 3 tokens/sec (70B, offload)

---

## Troubleshooting Configuration

### Configuration Not Applied

```bash
# Verify properties are loaded
java -jar app.jar --debug | grep "gollek.runners"

# Check active configuration
curl http://localhost:8080/q/config
```

### Out of Memory

```properties
# Reduce batch size
gollek.runners.*.max-batch-size=1

# Enable offload mode (Metal)
gollek.runners.metal.mode=offload

# Reduce KV cache blocks
gollek.runners.*.kv-cache-blocks=256
```

### Suboptimal Performance

```properties
# Verify GPU is being used
nvidia-smi  # CUDA/Blackwell
rocminfo    # ROCm
system_profiler SPDisplaysDataType  # Metal

# Check kernel selection
# Look for log messages like:
# "[CUDA] Optimization: FA4-TMEM-FP4"
# "[Metal] Using MPSGraph SDPA"
```

---

## Validation

### Test Configuration

```bash
# Run with verbose logging
java -jar app.jar -Dquarkus.log.level=DEBUG

# Check optimization manager initialization
# Look for:
# "[CUDA Opt] Initialized — compute=10.0 TMEM=true FP4=true"
# "[Blackwell Opt] TMEM allocated: 64.0 MB"
```

### Benchmark Configuration

```bash
# Run benchmark with current config
java -jar benchmark.jar \
  --runner cuda \
  --model llama-3-70b \
  --duration 60s \
  --report benchmark-report.md

# Compare with expected throughput
# See performance tables in each README
```

---

## Resources

- [CUDA Kernel README](../inference-gollek/extension/kernel/gollek-kernel-cuda/README.md)
- [Blackwell Kernel README](../inference-gollek/extension/kernel/gollek-kernel-blackwell/README.md)
- [ROCm Kernel README](../inference-gollek/extension/kernel/gollek-kernel-rocm/README.md)
- [Metal Kernel README](../inference-gollek/extension/kernel/gollek-kernel-metal/README.md)
- [Optimization Integration Guide](./OPTIMIZATION_INTEGRATION.md)

---

[Back to GPU Kernels](/docs/gpu-kernels) &nbsp; [View Architecture](/docs/architecture)
