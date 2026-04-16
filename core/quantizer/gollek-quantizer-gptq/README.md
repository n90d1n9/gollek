# gollek-quantizer-gptq

GPTQ implementation for Gollek — Hessian-based one-shot weight quantization for LLMs at 2/3/4/8 bits.

## Citation

```bibtex
@article{frantar2022gptq,
  title   = {GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author  = {Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  journal = {arXiv preprint arXiv:2210.17323},
  year    = {2022},
  url     = {https://arxiv.org/abs/2210.17323}
}
```

## Algorithm

GPTQ quantizes each weight matrix W layer-by-layer using second-order information from the Hessian of the layer's input activations.

### Core Algorithm (§3)

For each weight matrix W of shape `[d_out × d_in]`, GPTQ solves:

```
min_{Q}  ‖WX − QX‖²_F
```

where X is the calibration input and Q is the quantized weight matrix.

**Key insight:** quantize columns of W one at a time, updating remaining columns to compensate for the error introduced:

```
for each column j = 0..d_in:
  q_j ← quantize(w_j)                          # round to nearest grid point
  e_j ← (w_j − q_j) / [H^{-1}]_{jj}           # quantization error
  W_{j+1:} ← W_{j+1:} − e_j · [H^{-1}]_{j, j+1:}  # error compensation
```

where `H = 2 X Xᵀ` is the Hessian and `H^{-1}` is computed once via Cholesky decomposition.

**Dampening** for numerical stability: `H ← H + λI`, `λ = dampPercent · mean(diag(H))`.

### Group Quantization (§4.1)

Weights are quantized in groups of `groupSize` (default 128). Each group has its own scale and zero-point:

```
q = clamp(round(w / scale) + zero, 0, 2^bits − 1)
w̃ = (q − zero) · scale
```

Tensor naming convention (compatible with AutoGPTQ / HuggingFace):

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `layer.qweight` | `[d_in/8 × d_out]` | INT32 | Packed quantized weights (8 INT4 per INT32) |
| `layer.scales`  | `[d_in/groupSize × d_out]` | FP16 | Per-group scale factors |
| `layer.qzeros`  | `[d_in/groupSize × d_out/8]` | INT32 | Packed per-group zero points |
| `layer.g_idx`   | `[d_in]` | INT32 | Group index per column (act-order only) |
| `layer.bias`    | `[d_out]` | FP16 | Optional bias |

### Activation Order (§4.2)

With `actOrder=true`, columns are quantized in decreasing order of Hessian diagonal magnitude (most important weights first). Requires `g_idx` for correct dequantization.

### Dequantization

```
w̃ = (unpack(qweight) − unpack(qzeros)) · scales
```

For act-order: reorder columns using `g_idx` before applying scales.

## Compression

| bits | Elements per INT32 | Compression vs F32 |
|------|-------------------|-------------------|
| 2    | 16                | 16×               |
| 3    | 10 (+ 2 padding)  | ~10.7×            |
| 4    | 8                 | 8×                |
| 8    | 4                 | 4×                |

## Usage

```java
GPTQQuantizerService service = new GPTQQuantizerService();

// Quantize FP32 model → GPTQ INT4
QuantizationResult result = service.quantize(
    Path.of("model/"),
    Path.of("model-gptq/"),
    GPTQConfig.gptq4bit()
);

// Load and dequantize
GPTQLoader loader = service.loadQuantized(Path.of("model-gptq/"));

// Custom config
GPTQConfig cfg = GPTQConfig.builder()
    .bits(4)
    .groupSize(128)
    .actOrder(true)
    .dampPercent(0.01)
    .numSamples(128)
    .seqLen(2048)
    .build();
```

## Module Structure

| Class | Role |
|-------|------|
| `GPTQConfig` | Configuration record — bits, groupSize, actOrder, symmetric, exllamaV2 |
| `GPTQQuantizerService` | High-level quantize / dequantize / inspect service |
| `GPTQLoader` | Load GPTQ safetensors, auto-detect config |
| `GPTQSafetensorFileLoader` | Low-level safetensor shard loading |
| `GPTQSafetensorShard` | Single shard metadata and tensor access |
| `GPTQSafetensorHeader` | Safetensor header parsing |
| `GPTQSafetensorConverter` | GPTQ → FP32/FP16 dequantization converter |
| `GPTQQuantizerService.QuantizationResult` | Result record with compression stats |
| `QuantizedLayer` | Per-layer quantized weight container |
| `VectorDequantizer` | JDK Vector API dequantization (INT4 unpack + scale) |
| `MemoryAllocator` | Off-heap memory management for large models |

## Related Quantizers

| Module | Algorithm | Paper |
|--------|-----------|-------|
| **`gollek-quantizer-gptq`** | **GPTQ (Hessian one-shot)** | [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| `gollek-quantizer-awq` | AWQ (activation-aware) | [arxiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| `gollek-quantizer-autoround` | AutoRound (sign-SGD) | [arxiv 2309.05516](https://arxiv.org/abs/2309.05516) |
| `gollek-quantizer-quip` | QuIP# (E8 lattice, 2-bit) | [arxiv 2406.11235](https://arxiv.org/abs/2406.11235) |
| `gollek-quantizer-turboquant` | TurboQuant (rotation + Lloyd-Max) | [arxiv 2504.19874](https://arxiv.org/abs/2504.19874) |
