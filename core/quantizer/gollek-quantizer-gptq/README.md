# GPTQ Safetensor Loader & Converter
### JDK 25 · FFM API · Vector API (SIMD)

A high-performance Java implementation for loading and converting GPTQ-quantized LLM models stored in the `.safetensors` format. Built exclusively on JDK 25 modern APIs — **no JNI, no native libraries, no unsafe casts**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPTQLoader                               │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │ SafetensorParser  │    │        MemoryAllocator           │   │
│  │  (FFM API)        │    │        (FFM Arena)               │   │
│  │                   │    │                                  │   │
│  │ • mmap file       │───▶│ • SIMD-aligned off-heap buffers │   │
│  │ • parse JSON hdr  │    │ • Zero-copy tensor slices        │   │
│  │ • LE int64 read   │    │ • Deterministic Arena.close()   │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
│                                      │                           │
│                                      ▼                           │
│                          ┌──────────────────────┐               │
│                          │   QuantizedLayer      │               │
│                          │  qweight / qzeros /   │               │
│                          │  scales / g_idx / bias│               │
│                          └──────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VectorDequantizer                             │
│                    (Vector API / SIMD)                          │
│                                                                 │
│  For each input feature k, output feature j, group g:          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  w[k,j] = (unpack(qweight[k,j]) - unpack(qzeros[g,j])) │   │
│  │           × fp16_to_fp32(scales[g,j])                   │   │
│  │                                                          │   │
│  │  IntVector  → unpack 8× INT4 in parallel               │   │
│  │  FloatVector → SIMD multiply-add (FMA)                  │   │
│  │  AVX-512: 16 floats/cycle, AVX2: 8, NEON: 4            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SafetensorConverter                            │
│                                                                 │
│  • Builds output JSON header with updated tensor descriptors   │
│  • Memory-maps output file via FFM (zero-copy writes)          │
│  • Writes FP32 tensor data in safetensors binary format        │
│  • Preserves model metadata + adds conversion provenance       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key APIs Used

### Foreign Function & Memory (FFM) API — `java.lang.foreign`
| Usage | API |
|---|---|
| Memory-map entire .safetensors file | `FileChannel.map(READ_ONLY, 0, size, arena)` |
| Zero-copy tensor slice | `MemorySegment.asSlice(offset, length)` |
| Little-endian header read | `seg.get(JAVA_LONG.withOrder(LITTLE_ENDIAN), 0)` |
| SIMD-aligned allocation | `Arena.allocate(byteSize, 64L)` |
| Deterministic release | `Arena.ofShared()` / `Arena.close()` |
| Zero-copy write to output | `channel.map(READ_WRITE, 0, size, arena)` |

### Vector API — `jdk.incubator.vector`
| Usage | API |
|---|---|
| SIMD float multiply | `FloatVector.fromArray(SPECIES, ...).mul(v2)` |
| FMA (fused multiply-add) | `vw.fma(vi, acc)` |
| Horizontal sum | `acc.reduceLanes(VectorOperators.ADD)` |
| Auto-sized SIMD width | `FloatVector.PREFERRED_SPECIES` |

---

## GPTQ Format Reference

### Tensor Naming (AutoGPTQ convention)
```
model.layers.{N}.self_attn.q_proj.qweight   INT32  [outF/pack, inF]
model.layers.{N}.self_attn.q_proj.qzeros    INT32  [numGroups, outF/pack]
model.layers.{N}.self_attn.q_proj.scales    FP16   [numGroups, outF]
model.layers.{N}.self_attn.q_proj.g_idx     INT32  [inF]          (act-order only)
model.layers.{N}.self_attn.q_proj.bias      FP16   [outF]         (optional)
```

### INT4 Packing Layout
```
INT32 word (32 bits):
[ f7 f6 f5 f4  |  f3 f2 f1 f0  |  ... ]
  bits 28-31      bits 24-27        ...   (f = 4-bit weight value)
  
4-bit: 8 weights per INT32
8-bit: 4 weights per INT32
3-bit: 10 weights per INT32 (can span word boundaries)
```

### Dequantization Formula
```
w_fp32[k, j] = (q[k, j] - zeros[g, j]) × scales[g, j]

where:
  g = k / group_size            (or gIdx[k] for act-order)
  q = unpacked 4-bit value      (range [0, 15])
  zeros = unpacked qzeros + 1   (AutoGPTQ +1 convention)
  scales = fp16_to_fp32(raw)    (FP16 → FP32 expanded)
```

---

## Building

Requires **JDK 25+** and **Maven 3.9+**.

```bash
cd gptq-java
mvn package -DskipTests
```

---

## Running

```bash
# Required JVM flags
JAVA_OPTS="--enable-preview --add-modules=jdk.incubator.vector"

# Print SIMD capabilities
java $JAVA_OPTS -jar target/gptq-loader-1.0.0.jar caps

# Inspect a model
java $JAVA_OPTS -jar target/gptq-loader-1.0.0.jar info ./llama-3-8b-gptq/

# Load model into memory
java $JAVA_OPTS -jar target/gptq-loader-1.0.0.jar load ./llama-3-8b-gptq/ 4 128

# Convert GPTQ → FP32 safetensor
java $JAVA_OPTS -jar target/gptq-loader-1.0.0.jar convert \
    ./llama-3-8b-gptq/ \
    ./llama-3-8b-fp32.safetensors
```

---

## Running Tests

```bash
mvn test
```

16 unit tests covering:
- `GPTQConfig` preset values and derived properties
- FP16 ↔ FP32 round-trip conversions
- `MemoryAllocator` off-heap read/write correctness
- `VectorDequantizer` synthetic 4-bit dequantization
- Zero-point unpacking with AutoGPTQ +1 convention
- Matrix-vector multiply via Vector API
- Full synthetic quantize → dequantize → verify round-trip

---

## Performance Notes

| SIMD Width | Architecture | Float Lanes | Throughput (dequant) |
|---|---|---|---|
| 512-bit | AVX-512 (Intel Skylake-X+) | 16 | ~16× scalar |
| 256-bit | AVX2 (Intel Haswell+, AMD Zen2+) | 8 | ~8× scalar |
| 128-bit | SSE4 / ARM NEON | 4 | ~4× scalar |
| 128-bit | ARM SVE (Apple M1+) | 4–16 (flexible) | hardware-dependent |

The JIT compiler automatically selects the widest available SIMD instruction set.
Run `caps` to see what your JVM detects.

---

## Supported GPTQ Formats

| Format | Bits | Tested |
|---|---|---|
| AutoGPTQ standard | 4 | ✓ Primary target |
| AutoGPTQ | 8 | ✓ |
| AutoGPTQ act-order | 4 | ✓ (g_idx support) |
| GPTQ-for-LLaMA | 4 | ✓ (same format) |
| ExLlama v1 | 4 | ✓ |
| 2-bit / 3-bit | 2,3 | ✓ Generic path |
| ExLlama v2 | 4 | ⚠ Header-only |

---

## Project Structure

```
src/
├── main/java/com/gptq/
│   ├── Main.java                      CLI entry point
│   ├── model/
│   │   ├── SafetensorHeader.java      Safetensor JSON header model
│   │   ├── GPTQConfig.java            Quantization configuration
│   │   └── QuantizedLayer.java        Single quantized linear layer
│   ├── ffm/
│   │   ├── SafetensorParser.java      FFM mmap-based file parser
│   │   └── MemoryAllocator.java       Off-heap memory + FP16 utils
│   ├── vector/
│   │   └── VectorDequantizer.java     SIMD dequantization engine
│   ├── loader/
│   │   └── GPTQLoader.java            Orchestrated model loader
│   └── converter/
│       └── SafetensorConverter.java   GPTQ → FP32 converter + writer
└── test/java/com/gptq/
    └── GPTQTest.java                  16 unit tests
```
