# gollek-quantizer-awq

AWQ implementation for Gollek — activation-aware weight quantization that protects salient channels for accurate 4-bit LLM inference.

## Citation

```bibtex
@article{lin2023awq,
  title   = {AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author  = {Ji Lin and Jiaming Tang and Haotian Tang and Shang Yang and Wei-Ming Chen and Wei-Chen Wang and Guangxuan Xiao and Xingyu Dang and Chuang Gan and Song Han},
  journal = {arXiv preprint arXiv:2306.00978},
  year    = {2023},
  url     = {https://arxiv.org/abs/2306.00978}
}
```

## Algorithm

### Key Insight (§3)

Not all weights are equally important. Weights corresponding to **high-magnitude activation channels** cause disproportionate quantization error. AWQ identifies the ~1% of "salient" input channels from a calibration dataset and applies a per-channel scale factor before quantization — giving those channels more effective precision without storing them at higher bit-width.

### Activation-Aware Scaling (§3.2)

For each linear layer with weight `W ∈ [d_out × d_in]` and input `X`:

```
# Find salient channels from calibration activations
s[c] = mean(|X[:, c]|)^α        # α ∈ [0, 1], typically 0.5

# Pre-quantization transform (mathematically equivalent to original)
W' = W · diag(s)^(-1)           # scale down weights
X' = X · diag(s)                # scale up inputs

# Quantize the scaled weights
Q  = quantize(W')

# At inference: Q · X' = W · diag(s)^(-1) · diag(s) · X = W · X  ✓
```

The scale `s` is chosen to minimise quantization error on the salient channels while keeping the overall output unchanged.

### Dequantization (§4)

Groups are along the **input** dimension (unlike GPTQ which groups along output):

```
w[i, j] = (q[i, j] − zero[g_in, j]) × scale[g_in, j]
           where g_in = i / groupSize
```

**AWQ vs GPTQ grouping direction:**

| | GPTQ | AWQ |
|---|---|---|
| Group axis | output (`d_out`) | input (`d_in`) |
| `qweight` shape | `[d_out/8, d_in]` | `[d_in/8, d_out]` (transposed) |
| `scales` shape | `[d_out/groupSize, d_in]` | `[d_in/groupSize, d_out]` |

### INT4 Packing (AutoAWQ GEMM format)

Each `INT32` in `qweight[pi, j]` packs 8 consecutive input-feature weights for the same output feature `j`:

```
qweight[pi, j] = w[pi*8+0, j] | (w[pi*8+1, j] << 4) | ... | (w[pi*8+7, j] << 28)
```

Zero-points in `qzeros[g, pj]` pack 8 consecutive output-feature zeros for the same group:

```
qzeros[g, pj] = zero[g, pj*8+0] | (zero[g, pj*8+1] << 4) | ... | (zero[g, pj*8+7] << 28)
```

## Tensor Layout

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `layer.qweight` | `[d_in/8, d_out]` | INT32 | Packed INT4 weights (input-grouped) |
| `layer.scales`  | `[d_in/groupSize, d_out]` | FP16 | Per-group scale factors |
| `layer.qzeros`  | `[d_in/groupSize, d_out/8]` | INT32 | Packed INT4 zero-points (optional) |

## Kernel Formats

| Format | Use case | Packing |
|--------|----------|---------|
| `GEMM` | Batch inference (default, AutoAWQ) | Column-major: consecutive output features packed |
| `GEMV` | Single-token generation | Row-major: consecutive input features packed |
| `MARLIN` | GPU (exllama v2 / vLLM) | Interleaved for GPU memory access patterns |

## Usage

```java
AWQQuantizerService service = new AWQQuantizerService();

// Load AutoAWQ-quantized model
AWQLoader loader = service.loadQuantized(Path.of("model-awq/"));

// Dequantize to FP32
AWQSafetensorConverter converter = new AWQSafetensorConverter(loader,
    new AWQSafetensorConverter.ConversionConfig("float32"));
converter.convert(Path.of("model-fp32/"));

// Custom config
AWQConfig cfg = AWQConfig.builder()
    .bits(4)
    .groupSize(128)
    .kernelFormat(AWQConfig.KernelFormat.GEMM)
    .hasZeros(true)
    .activationAware(true)
    .build();
```

## Module Structure

| Class | Role |
|-------|------|
| `AWQConfig` | Configuration record — bits, groupSize, kernelFormat, hasZeros, activationAware |
| `AWQQuantizerService` | High-level load / dequantize service |
| `AWQLoader` | Load AWQ safetensors, auto-detect config |
| `AWQSafetensorFileLoader` | Low-level safetensor shard loading |
| `AWQSafetensorShard` | Single shard metadata and tensor access |
| `AWQSafetensorConverter` | AWQ → FP32/FP16 dequantization converter |
| `AWQDequantizer` | JDK 25 Vector API INT4 unpack + scale (GEMM/GEMV/Marlin) |
| `AWQLayer` | Per-layer quantized weight container |

## Related Quantizers

| Module | Algorithm | Paper |
|--------|-----------|-------|
| `gollek-quantizer-gptq` | GPTQ (Hessian one-shot) | [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| **`gollek-quantizer-awq`** | **AWQ (activation-aware)** | [arxiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| `gollek-quantizer-autoround` | AutoRound (sign-SGD) | [arxiv 2309.05516](https://arxiv.org/abs/2309.05516) |
| `gollek-quantizer-quip` | QuIP# (E8 lattice, 2-bit) | [arxiv 2406.11235](https://arxiv.org/abs/2406.11235) |
| `gollek-quantizer-turboquant` | TurboQuant (rotation + Lloyd-Max) | [arxiv 2504.19874](https://arxiv.org/abs/2504.19874) |
