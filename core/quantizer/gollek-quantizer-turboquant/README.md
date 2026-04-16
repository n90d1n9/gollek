# gollek-quantizer-turboquant

TurboQuant implementation for Gollek — online vector quantization with near-optimal distortion rate for LLM weights and KV cache.

## Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author  = {Amir Zandieh and Majid Daliri and Insu Han},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.19874}
}
```

## Algorithm

TurboQuant provides two variants with provable distortion bounds.

### TurboQuant_mse — Algorithm 1 (§3.1)

MSE-optimal scalar quantization after random rotation:

```
Quant(x):
  y   ← Π · x                        # rotate to near-spherical distribution
  idx ← argmin_k |y[j] − c_k|  ∀j   # nearest Lloyd-Max centroid per coord

DeQuant(idx):
  ỹ ← c_{idx[j]}  ∀j                 # centroid lookup
  x̃ ← Πᵀ · ỹ                        # rotate back
```

**Distortion bound (Theorem 1):**
```
Dmse ≤ (√(3π)/2) · 4^(-b)  ≈  2.72 · 4^(-b)
```

Information-theoretic lower bound: `Dmse ≥ 4^(-b)`, so TurboQuant is within a factor of `√(3π)/2 ≈ 2.72` of optimal for all bit-widths.

| b | TurboQuant MSE | Lower bound | Gap |
|---|---------------|-------------|-----|
| 1 | 0.36          | 0.25        | 1.44× |
| 2 | 0.117         | 0.0625      | 1.87× |
| 3 | 0.030         | 0.0156      | 1.92× |
| 4 | 0.009         | 0.0039      | 2.31× |

### TurboQuant_prod — Algorithm 2 (§3.2)

Unbiased inner product estimator — adds a 1-bit QJL residual correction:

```
Quant(x):
  idx ← Quant_mse(x)          # stage 1: MSE quant at (b-1) bits
  r   ← x − DeQuant_mse(idx)  # residual
  γ   ← ‖r‖₂                  # residual norm
  qjl ← sign(S · r)           # stage 2: 1-bit QJL, S ~ N(0,1)^(d×d)
  output: (idx, qjl, γ)

DeQuant(idx, qjl, γ):
  x̃_mse ← DeQuant_mse(idx)
  x̃_qjl ← (√(π/2) / d) · γ · Sᵀ · qjl
  output: x̃_mse + x̃_qjl
```

**Inner product bound (Theorem 2):**
```
E[⟨y, x̃⟩] = ⟨y, x⟩   (unbiased)
Var[⟨y, x̃⟩] ≤ (√(3π)/2) · ‖y‖² / d · 4^(-b)
```

### Rotation Strategies

| Strategy | Cost | Notes |
|----------|------|-------|
| `HADAMARD` | O(d log d) | Fast Walsh-Hadamard + random ±1 diagonal. Preferred for large d. |
| `RANDOM_ORTHOGONAL` | O(d²) | Exact uniform random rotation via QR. Highest quality. |
| `RANDOM_SVD` | O(d·k) | Structured random projection. Intermediate tradeoff. |

### KV Cache Mode (§4.2–4.3)

Outlier-channel splitting for attention KV cache quantization:

```
32 outlier channels × (b+1) bits  +  (d−32) normal channels × b bits
─────────────────────────────────────────────────────────────────────
                    d total channels
```

Example: `d=128`, `b=2`, 32 outliers → `(32×3 + 96×2)/128 = 2.5` effective bits/channel.

## Usage

```java
// Weight quantization — MSE variant, 4-bit, Hadamard rotation
TurboQuantConfig cfg = TurboQuantConfig.mse4bit(768);
TurboQuantEngine engine = new TurboQuantEngine(cfg);

// Quantize a vector
byte[] codes = engine.quantize(weightVector);

// Dequantize
float[] approx = engine.dequantize(codes);

// KV cache — inner product variant, 2-bit, outlier split (2.5-bit effective)
TurboQuantConfig kvCfg = TurboQuantConfig.prod2bitKvCache(128);
TurboQuantKVCache kvCache = new TurboQuantKVCache(kvCfg);
kvCache.store(keyVector, valueVector);
float[] estimatedDot = kvCache.innerProduct(queryVector);
```

## Module Structure

| Class | Role |
|-------|------|
| `TurboQuantConfig` | Configuration record — variant, bits, rotation, outlier split, distortion bounds |
| `TurboQuantEngine` | Core quantize/dequantize (JDK 25 Vector API) |
| `TurboQuantEngineSIMD` | SIMD-optimised path for batch quantization |
| `TurboQuantKVCache` | Online KV cache quantization with outlier splitting |
| `RandomRotation` | Hadamard / random orthogonal / random SVD rotation matrices |
| `LloydMaxCodebook` | Optimal scalar codebook for N(0, 1/d) source |
| `TurboQuantDequantizer` | Dequantization with QJL residual correction |
| `TurboQuantService` | High-level service + safetensor file I/O |
| `QuantizerRegistry` | Plugin registry for multi-format quantizer dispatch |
| `BnBDequantizer` | bitsandbytes-compatible dequantization |
| `HQQDequantizer` | HQQ-compatible dequantization |
| `GGUFDequantizer` | GGUF-compatible dequantization |
| `SqueezeLLMDequantizer` | SqueezeLLM sparse dequantization |
| `StreamingSafetensorWriter` | Streaming output to `.safetensors` |
| `ParallelLoader` | Parallel tensor loading |

## Comparison with Related Quantizers

| Module | Algorithm | Bits | Distortion bound | Unbiased IP |
|--------|-----------|------|-----------------|-------------|
| `gollek-quantizer-gptq` | GPTQ (Hessian) | 4 | empirical | ✗ |
| `gollek-quantizer-awq` | AWQ (activation-aware) | 4 | empirical | ✗ |
| `gollek-quantizer-autoround` | AutoRound (sign-SGD) | 4 | empirical | ✗ |
| `gollek-quantizer-quip` | QuIP# (E8 lattice) | 2 | empirical | ✗ |
| **`gollek-quantizer-turboquant`** | **TurboQuant (rotation + Lloyd-Max)** | **1–4** | **≤ 2.72 · 4^(-b) (proven)** | **✓ (_prod variant)** |

TurboQuant is the only quantizer in this suite with a **provable near-optimal distortion bound** and an **unbiased inner product estimator**, making it particularly suited for attention KV cache quantization where inner product accuracy is critical.
