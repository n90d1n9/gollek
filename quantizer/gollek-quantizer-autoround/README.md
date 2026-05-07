# gollek-quantizer-autoround

AutoRound implementation for Gollek — sign-SGD optimization of rounding decisions and scales for accurate 2/3/4/8-bit LLM quantization.

## Citations

```bibtex
@article{cheng2023autoround,
  title   = {Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author  = {Wenhua Cheng and Weiwei Zhang and Haihao Shen and Yiyang Cai and Xin He and Kaokao Lv},
  journal = {arXiv preprint arXiv:2309.05516},
  year    = {2023},
  url     = {https://arxiv.org/abs/2309.05516}
}
```

Blog: [AutoRound v0.4 — LMSYS, 2025-11-13](https://www.lmsys.org/blog/2025-11-13-AutoRound/)

## Algorithm

### Key Insight (§3)

AutoRound jointly optimizes **rounding decisions** (`v`), **scales** (`s`), and **zero-points** (`z`) using a block-wise reconstruction objective with SignSGD — unlike GPTQ (Hessian, no scale optimization) and AWQ (scale only, no rounding optimization).

### Optimization Loop (§3.2)

For each transformer block `B` with calibration inputs `X`:

```
Initialize:  s, z  ← min/max of each weight group
             v     ← 0  (rounding perturbation ∈ {0, 1})

For t = 1..T (default T=200):
  q(W) = clamp(round(W/s + 0.5·v) − z,  0,  2^b − 1)   # quantize
  W̃    = (q(W) + z) × s                                  # dequantize
  L    = ‖B(X) − B̃(X)‖²                                  # block reconstruction loss
  s, z ← Adam step on ∂L/∂s, ∂L/∂z                      # optimize scales
  v    ← v − η · sign(∂L/∂v)                             # SignSGD on rounding
```

The `v` update is the core contribution: instead of always rounding to nearest, AutoRound learns whether each weight should round up or down to minimise the block output error.

### Dequantization

```
w[j, i] = (q[j, i] − zp[j, g]) × scale[j, g]
           where g = i / groupSize
```

Groups are along the **input** dimension. Zero-points are stored as plain `INT32` (not packed), scales as `FP32` — both differ from GPTQ/AWQ.

**Dequantization formula comparison:**

| Quantizer | Formula | Group axis | Scale dtype | ZP storage |
|-----------|---------|------------|-------------|------------|
| GPTQ | `(q − zero_packed) × scale` | output | FP16 | packed INT32 |
| AWQ | `(q − zero_packed) × scale` | input | FP16 | packed INT32 |
| **AutoRound (native)** | **`(q − zp_plain) × scale`** | **input** | **FP32** | **plain INT32** |
| AutoRound (GPTQ-compat) | `(q − zero_packed + 1) × scale` | input | FP16 | packed INT32 |

### INT4 Packing (AUTOROUND_NATIVE / GPTQ_COMPAT)

`qweight` shape: `[d_out/8, d_in]` — packed along the **output** dimension (same as GPTQ):

```
qweight[pr, c] = q[pr*8+0, c] | (q[pr*8+1, c] << 4) | ... | (q[pr*8+7, c] << 28)
```

ITREX (Intel Extension for PyTorch) uses row-major packing with reversed bit order within INT32.

## Tensor Layout

### AutoRound Native Format

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `layer.weight` | `[d_out/pack, d_in]` | INT32 | Packed quantized weights |
| `layer.scale`  | `[d_out, d_in/groupSize]` | FP32 | Per-group scales |
| `layer.zp`     | `[d_out, d_in/groupSize]` | INT32 | Per-group zero-points (plain, not packed) |

### GPTQ-Compatible Export

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `layer.qweight` | `[d_out/pack, d_in]` | INT32 | Same packing as GPTQ |
| `layer.scales`  | `[d_out, d_in/groupSize]` | FP16 | FP16 scales (GPTQ naming) |
| `layer.qzeros`  | `[d_out, d_in/groupSize/pack]` | INT32 | Packed zero-points (GPTQ naming) |

The loader auto-detects format from tensor names.

## Pack Formats

| Format | Use case | Scale dtype | ZP storage |
|--------|----------|-------------|------------|
| `AUTOROUND_NATIVE` | Intel AutoRound default | FP32 | plain INT32 |
| `GPTQ_COMPAT` | llama.cpp / vLLM / HuggingFace | FP16 | packed INT32 |
| `ITREX` | Intel Extension for PyTorch | FP32 | plain INT32, reversed bit order |

## Backend Targets

| Target | Description |
|--------|-------------|
| `exllamav2` | Default — exllama v2 kernel (GPU) |
| `gptq` | GPTQ-compatible export for llama.cpp / vLLM |
| `marlin` | Marlin kernel (high-throughput GPU) |
| `ipex` | Intel Extension for PyTorch (CPU/Xeon) |

## Usage

```java
AutoRoundQuantizerService service = new AutoRoundQuantizerService();

// Load AutoRound-quantized model
AutoRoundLoader loader = service.loadQuantized(Path.of("model-autoround/"));

// Dequantize to FP32
AutoRoundSafetensorConverter converter = new AutoRoundSafetensorConverter(loader, config);
converter.convert(Path.of("model-fp32/"));

// Custom config
AutoRoundConfig cfg = AutoRoundConfig.builder()
    .bits(4)
    .groupSize(128)
    .numIters(200)
    .learningRate(0.001)
    .useAdam(false)          // SignSGD for rounding (paper default)
    .backendTarget("gptq")
    .build();
```

## Module Structure

| Class | Role |
|-------|------|
| `AutoRoundConfig` | Configuration record — bits, groupSize, packFormat, scaleDtype, numIters, SignSGD/Adam |
| `AutoRoundQuantizerService` | High-level load / dequantize service |
| `AutoRoundLoader` | Load AutoRound safetensors, auto-detect format |
| `AutoRoundDequantizer` | JDK 25 Vector API dequantization (native + GPTQ-compat + ITREX) |
| `AutoRoundSafetensorConverter` | AutoRound → FP32/FP16 converter |
| `AutoRoundLayer` | Per-layer quantized weight container |
| `SafetensorParser` | Low-level safetensor header + shard parsing |

## Related Quantizers

| Module | Algorithm | Paper |
|--------|-----------|-------|
| `gollek-quantizer-gptq` | GPTQ (Hessian one-shot) | [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| `gollek-quantizer-awq` | AWQ (activation-aware scales) | [arxiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| **`gollek-quantizer-autoround`** | **AutoRound (SignSGD rounding + scale opt.)** | [arxiv 2309.05516](https://arxiv.org/abs/2309.05516) |
| `gollek-quantizer-quip` | QuIP# (E8 lattice, 2-bit) | [arxiv 2406.11235](https://arxiv.org/abs/2406.11235) |
| `gollek-quantizer-turboquant` | TurboQuant (rotation + Lloyd-Max) | [arxiv 2504.19874](https://arxiv.org/abs/2504.19874) |
