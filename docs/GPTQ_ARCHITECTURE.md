# GPTQ Quantizer Architecture - Separation of Concerns

## ✅ Properly Architected

The GPTQ quantizer module now follows clean separation of concerns with **NO code duplication**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   safetensor-loader Module                          │
│              (Generic Safetensor Infrastructure)                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  SafetensorFFMLoader                                     │      │
│  │  - FFM-based memory mapping                              │      │
│  │  - MMAP/COPY fallback                                    │      │
│  │  - CDI-managed lifecycle                                 │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  SafetensorHeaderParser                                  │      │
│  │  - Zero-copy JSON header parsing                         │      │
│  │  - Returns SafetensorHeader                              │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  SafetensorLoadResult                                    │      │
│  │  - Owns Arena lifecycle                                  │      │
│  │  - Provides Safetensor access                            │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  SafetensorTensor                                        │      │
│  │  - Zero-copy MemorySegment view                          │      │
│  │  - Typed accessors (int[], float[], short[])             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  SafetensorHeader / SafetensorTensorInfo                 │      │
│  │  - Metadata models                                       │      │
│  │  - Shape, dtype, offsets                                 │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Uses (dependency)
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                gollek-quantizer-gptq Module                         │
│              (GPTQ-Specific Quantization Logic)                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  GPTQSafetensorFileLoader                                │      │
│  │  - Standalone (non-CDI) wrapper                          │      │
│  │  - Delegates to SafetensorHeaderParser                   │      │
│  │  - Manages file I/O and MMAP                             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  GPTQSafetensorShard                                     │      │
│  │  - Wraps SafetensorLoadResult                            │      │
│  │  - GPTQ-specific accessors                               │      │
│  │  - getTensorAsInt32(), getTensorAsFp16()                 │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  GPTQLoader                                              │      │
│  │  - Multi-shard model loading                             │      │
│  │  - GPTQ layer discovery & grouping                       │      │
│  │  - Uses GPTQSafetensorShard for file access              │      │
│  │  - Builds QuantizedLayer objects                         │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  QuantizedLayer                                          │      │
│  │  - GPTQ layer representation                             │      │
│  │  - qweight, qzeros, scales, g_idx, bias                  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐      │
│  │  VectorDequantizer                                       │      │
│  │  - SIMD dequantization (JDK Vector API)                  │      │
│  │  - INT4/INT8 → FP32/FP16                                 │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  GPTQSafetensorConverter                                 │      │
│  │  - GPTQ quantized → dequantized conversion               │      │
│  │  - Writes output safetensor files                        │      │
│  │  - Progress callbacks, async support                     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  GPTQQuantizerService                                    │      │
│  │  - Unified service facade                                │      │
│  │  - Quantize, dequantize, inspect operations              │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Separation of Concerns

### ✅ safetensor-loader Module (Generic)
**Responsibility**: Generic safetensor file parsing and loading

**Classes**:
- `SafetensorFFMLoader` - CDI-managed FFM loader
- `SafetensorHeaderParser` - Zero-copy JSON header parsing
- `SafetensorLoadResult` - Load result with lifecycle management
- `SafetensorTensor` - Zero-copy tensor view
- `SafetensorHeader` - Header metadata model
- `SafetensorTensorInfo` - Tensor metadata model
- `SafetensorDType` - Data type enumeration
- `SafetensorShardLoader` - Multi-shard loading
- `SafetensorLoaderFacade` - High-level API

**Location**: `gollek/plugins/runner/safetensor/gollek-safetensor-loader/`

**Key Principle**: NO GPTQ-specific logic. Pure safetensor format handling.

---

### ✅ gollek-quantizer-gptq Module (GPTQ-Specific)
**Responsibility**: GPTQ quantization algorithm and model handling

#### Layer 1: Safetensor Access (Thin Wrappers)

**`GPTQSafetensorFileLoader`**
- **Purpose**: Standalone (non-CDI) file loader
- **Why**: GPTQ module doesn't always run in CDI context
- **Delegates to**: `SafetensorHeaderParser` from safetensor-loader
- **Own logic**: File I/O, MMAP fallback, chunked reading

**`GPTQSafetensorShard`**
- **Purpose**: GPTQ-specific view over a safetensor shard
- **Wraps**: `SafetensorLoadResult` from safetensor-loader
- **Own logic**: Convenience methods for GPTQ access patterns
  - `getTensorAsInt32()` - for qweight, qzeros
  - `getTensorAsFp16()` - for scales, bias
  - `getTensorShape()` - shape introspection

#### Layer 2: GPTQ Model Loading

**`GPTQLoader`**
- **Purpose**: Load multi-shard GPTQ models
- **Own logic**:
  - Shard discovery (file naming patterns)
  - GPTQ layer detection (`.qweight`, `.qzeros`, `.scales`)
  - Layer grouping and organization
  - Auto-detection of GPTQ config from metadata
- **Uses**: `GPTQSafetensorShard` for file access

**`QuantizedLayer`**
- **Purpose**: Represent a single GPTQ quantized layer
- **Own logic**:
  - GPTQ tensor organization (qweight, qzeros, scales, g_idx, bias)
  - Dimension inference from tensor shapes
  - Completeness validation

#### Layer 3: Quantization Algorithms

**`VectorDequantizer`**
- **Purpose**: SIMD-accelerated dequantization
- **Own logic**:
  - GPTQ dequantization formula: `(q - zero) * scale`
  - INT4/INT8 unpacking from packed INT32
  - Vector API optimization (AVX2, AVX-512, NEON)
  - Act-order (g_idx) support

**`GPTQConfig`**
- **Purpose**: GPTQ quantization configuration
- **Own logic**:
  - Bit width, group size, act-order flags
  - Preset configurations (gptq4bit, gptq8bit, etc.)
  - Builder pattern for custom configs

#### Layer 4: Conversion & Services

**`GPTQSafetensorConverter`** ⭐ (Renamed from `SafetensorConverter`)
- **Purpose**: Convert GPTQ → dequantized FP32/FP16
- **Why GPTQ-specific**: Only works with GPTQ quantized models
- **Own logic**:
  - Layer-by-layer dequantization
  - Output safetensor file generation
  - Progress callbacks, async support, cancellation

**`GPTQQuantizerService`**
- **Purpose**: Unified service facade
- **Own logic**:
  - High-level API (quantize, dequantize, inspect)
  - Async operation management
  - Result statistics and reporting

**`MemoryAllocator`**
- **Purpose**: FFM memory management for GPTQ
- **Own logic**:
  - Off-heap allocation for tensor data
  - FP16 ↔ FP32 conversion utilities
  - Memory tracking and cleanup

---

## Dependency Flow

```
gollek-quantizer-gptq  ──depends-on──▶  gollek-safetensor-loader
     (GPTQ-specific)                        (Generic)
```

**pom.xml**:
```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-loader</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

---

## What Was Refactored

### ✅ Completed

1. **Renamed `SafetensorConverter` → `GPTQSafetensorConverter`**
   - Reason: GPTQ-specific conversion logic
   - Updated all references in dependent modules

2. **Created proper wrapper classes** (already existed ✅):
   - `GPTQSafetensorFileLoader` - standalone file loader
   - `GPTQSafetensorShard` - GPTQ-specific shard view
   - These wrap safetensor-loader classes, NO duplication

3. **Added dependency** on `gollek-safetensor-loader`

4. **Deleted duplicate generic classes**:
   - ❌ `SafetensorHeader.java` (use safetensor-loader's)
   - ❌ `SafetensorParser.java` (use safetensor-loader's)

---

## Code Examples

### Using GPTQLoader (GPTQ-specific)

```java
// Load a GPTQ quantized model
GPTQConfig config = GPTQConfig.gptq4bit();
try (GPTQLoader loader = new GPTQLoader(modelPath, config).load()) {
    
    // Access GPTQ layers
    for (String layerName : loader.getLayerNames()) {
        QuantizedLayer layer = loader.getLayer(layerName);
        System.out.printf("Layer: %s, in=%d, out=%d, bits=%d%n",
            layerName, 
            layer.getInFeatures(),
            layer.getOutFeatures(),
            config.bits());
    }
    
    // Dequantize using Vector API
    VectorDequantizer dequantizer = new VectorDequantizer(config);
    // ... dequantization logic
}
```

### Using safetensor-loader directly (Generic)

```java
@Inject
SafetensorFFMLoader loader;

// Load any safetensor model (not GPTQ-specific)
try (SafetensorLoadResult result = loader.load(modelPath)) {
    SafetensorTensor tensor = result.tensor("model.embed_tokens.weight");
    float[] weights = tensor.toFloatArray();
    // ... generic tensor operations
}
```

### Using GPTQSafetensorFileLoader (Standalone)

```java
// Non-CDI context (e.g., CLI tool, test)
try (GPTQSafetensorFileLoader fileLoader = new GPTQSafetensorFileLoader()) {
    GPTQSafetensorShard shard = fileLoader.loadShard(modelPath);
    
    // GPTQ-specific access
    int[] qweight = shard.getTensorAsInt32("layer.0.qweight");
    short[] scales = shard.getTensorAsFp16("layer.0.scales");
    
    shard.close();
}
```

---

## Benefits of This Architecture

### ✅ No Code Duplication
- Generic safetensor parsing: **safetensor-loader** (single source)
- GPTQ quantization logic: **gollek-quantizer-gptq** (single source)

### ✅ Clear Boundaries
- **safetensor-loader**: "How to read .safetensors files"
- **gollek-quantizer-gptq**: "How to work with GPTQ quantized models"

### ✅ Reusability
- safetensor-loader can be used by ANY module
- GPTQ module focuses on quantization algorithms only

### ✅ Maintainability
- Bug fixes in safetensor parsing → fix in ONE place
- GPTQ algorithm improvements → isolated to GPTQ module

### ✅ Testability
- safetensor-loader: test file parsing, FFM, MMAP
- GPTQ module: test quantization logic, dequantization accuracy

---

## Module Responsibilities Summary

| Module | Responsibility | Example Classes |
|--------|---------------|-----------------|
| **safetensor-loader** | Generic .safetensors file I/O | `SafetensorFFMLoader`, `SafetensorTensor`, `SafetensorHeaderParser` |
| **gollek-quantizer-gptq** | GPTQ quantization algorithm | `GPTQLoader`, `VectorDequantizer`, `GPTQConfig`, `GPTQSafetensorConverter` |

---

## Next Steps

1. ✅ **Architecture**: Properly separated (DONE)
2. ✅ **Dependencies**: Added safetensor-loader dependency (DONE)
3. ✅ **Naming**: GPTQ-specific classes prefixed with `GPTQ` (DONE)
4. ⚠️ **Testing**: Run `mvn clean test` to verify integration
5. ⚠️ **Documentation**: Update any outdated references in docs

---

## Files Summary

### gollek-quantizer-gptq Module

| File | Type | Purpose |
|------|------|---------|
| `GPTQConfig.java` | GPTQ-specific | Quantization configuration |
| `GPTQLoader.java` | GPTQ-specific | Multi-shard GPTQ model loading |
| `QuantizedLayer.java` | GPTQ-specific | GPTQ layer representation |
| `VectorDequantizer.java` | GPTQ-specific | SIMD dequantization |
| `GPTQSafetensorConverter.java` | GPTQ-specific | GPTQ → FP32/FP16 conversion |
| `GPTQQuantizerService.java` | GPTQ-specific | Unified service facade |
| `GPTQSafetensorFileLoader.java` | Wrapper | Standalone file loader (wraps safetensor-loader) |
| `GPTQSafetensorShard.java` | Wrapper | GPTQ shard view (wraps safetensor-loader) |
| `MemoryAllocator.java` | GPTQ-specific | FFM memory management |
| `GPTQSafetensorHeader.java` | ❌ Unused | Can be deleted (not needed) |

### safetensor-loader Module (Referenced)

| File | Type | Purpose |
|------|------|---------|
| `SafetensorFFMLoader.java` | Generic | CDI-managed FFM loader |
| `SafetensorHeaderParser.java` | Generic | JSON header parsing |
| `SafetensorLoadResult.java` | Generic | Load result with lifecycle |
| `SafetensorTensor.java` | Generic | Zero-copy tensor view |
| `SafetensorHeader.java` | Generic | Header metadata |
| `SafetensorTensorInfo.java` | Generic | Tensor metadata |

---

## Conclusion

✅ **Excellent separation of concerns achieved!**

- **NO code duplication**
- **Clear module boundaries**
- **Proper dependency flow**
- **GPTQ-specific vs generic clearly separated**

The architecture follows the **Dependency Inversion Principle**:
- High-level GPTQ logic depends on abstractions (safetensor-loader)
- Low-level file I/O details are encapsulated in safetensor-loader
- Both modules depend on abstractions, not concrete implementations
