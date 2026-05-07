# gollek-quantizer-quip

QuIP# implementation for Gollek — 2-bit LLM weight quantization using incoherence processing and E8 lattice vector quantization.

## Citation

```bibtex
@article{tseng2024quip,
  title   = {QuIP\#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks},
  author  = {Albert Tseng and Jerry Chee and Qingyao Sun and Volodymyr Kuleshov and Christopher De Sa},
  journal = {arXiv preprint arXiv:2406.11235},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.11235}
}
```

## Algorithm

QuIP# quantizes LLM weights to 2 bits/weight in two steps:

### 1. Incoherence Processing (§2)

Apply random orthogonal transforms to the weight matrix before quantization:

```
W̃ = U W V^T
```

where `U` and `V` are randomized Hadamard matrices (`H_n · D`, D = random ±1 diagonal).

This spreads the weight distribution uniformly, reducing worst-case quantization error from `O(‖W‖_∞)` to `O(‖W‖_F / √n)`.

Implementation: `RandomHadamard` — Fast Walsh-Hadamard Transform in O(n log n).

### 2. E8 Lattice Vector Quantization (§3)

Partition the transformed weights into 8-dimensional blocks and find the nearest codeword in the E8 lattice codebook:

```
code[b] = argmin_{c ∈ E8} ‖W̃_block_b / scale_b − c‖²
```

The E8 codebook has 256 entries (2 bits/dim × 8 dims = 16 bits = 256 codewords), built from the 240 minimal vectors of the E8 lattice shell (norm² = 2) plus 16 shell-2 vectors.

Implementation: `E8Codebook` — exhaustive nearest-neighbor search (256 × 8 = 2048 ops/block).

### Dequantization

```
W ≈ U^T W̃ V    where W̃_block = scale_b · E8[code_b]
```

## Compression

| Format | bits/weight | Compression vs F32 |
|--------|------------|-------------------|
| F32    | 32         | 1×                |
| INT8   | 8          | 4×                |
| INT4   | 4          | 8×                |
| QuIP# (E8, 2-bit) | 2 | ~6.4× |

At 2-bit: each 8-dim block → 1 byte (code index) + 4 bytes (scale) = 5 bytes for 32 bytes of F32.

## Usage

```java
// 2-bit quantization
QuipQuantizerService svc = new QuipQuantizerService(QuipConfig.quip2bit());

// Quantize named weight tensors
Map<String, QuipTensor> quantized = svc.quantize(weights, shapes);

// Dequantize
Map<String, float[]> approx = svc.dequantize(quantized);

// From a .safetensors file
svc.quantizeFile(Path.of("model.safetensors"), Path.of("model-quip/"));
```

## Module Structure

| Class | Role |
|-------|------|
| `E8Codebook` | 256-entry E8 lattice codebook, nearest-index search |
| `RandomHadamard` | Fast Walsh-Hadamard Transform + random ±1 diagonal |
| `QuipQuantizer` | Core quantize/dequantize algorithm |
| `QuipTensor` | Compressed tensor record (codes + scales + seeds) |
| `QuipConfig` | Configuration (bits, Hadamard seeds, fallback) |
| `QuipQuantizerService` | High-level service + `.safetensors` file I/O |

## Related Quantizers

| Module | Algorithm | Paper |
|--------|-----------|-------|
| `gollek-quantizer-gptq` | GPTQ (Hessian-based INT4) | [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| `gollek-quantizer-awq` | AWQ (activation-aware INT4) | [arxiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| `gollek-quantizer-autoround` | AutoRound (sign gradient descent) | [arxiv 2309.05516](https://arxiv.org/abs/2309.05516) |
| `gollek-quantizer-quip` | **QuIP# (E8 lattice, 2-bit)** | [arxiv 2406.11235](https://arxiv.org/abs/2406.11235) |
