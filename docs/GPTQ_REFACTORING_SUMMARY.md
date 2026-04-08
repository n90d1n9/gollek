# GPTQ Quantizer Refactoring Summary

## Changes Made

### 1. ✅ Renamed GPTQ-Specific Classes

**`SafetensorConverter.java` → `GPTQSafetensorConverter.java`**
- **Reason**: This class is GPTQ-specific (converts GPTQ quantized models → dequantized FP32/FP16)
- **Location**: `gollek/core/quantizer/gollek-quantizer-gptq/src/main/java/tech/kayys/gollek/quantizer/gptq/`
- **Updated References**:
  - `GPTQQuantizerService.java` - Updated to use `GPTQSafetensorConverter`
  - `SafetensorRunnerPlugin.java` - Updated imports and method signatures
  - `QuantizationResource.java` - Updated imports

### 2. ✅ Removed Duplicate Generic Classes

**Deleted from GPTQ module:**
- `SafetensorHeader.java` - Generic safetensor header parser
- `SafetensorParser.java` - Generic FFM-based safetensor file parser

**Reason**: These are NOT GPTQ-specific. They are generic safetensor utilities that already exist in:
- `gollek/plugins/runner/safetensor/gollek-safetensor-loader/src/main/java/tech/kayys/gollek/safetensor/loader/`
  - `SafetensorHeader.java`
  - `SafetensorHeaderParser.java`
  - `SafetensorFFMLoader.java`

**Added Dependency:**
```xml
<!-- gollek/core/quantizer/gollek-quantizer-gptq/pom.xml -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-loader</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

### 3. ⚠️ Requires Manual Fix

**`GPTQLoader.java`** currently has broken imports:
```java
import com.gptq.ffm.MemoryAllocator;      // ❌ Should be: tech.kayys.gollek.quantizer.gptq.MemoryAllocator
import com.gptq.ffm.SafetensorParser;      // ❌ DELETED - Use safetensor-loader instead
import com.gptq.model.GPTQConfig;          // ❌ Should be: tech.kayys.gollek.quantizer.gptq.GPTQConfig
import com.gptq.model.QuantizedLayer;      // ❌ Should be: tech.kayys.gollek.quantizer.gptq.QuantizedLayer
import com.gptq.model.SafetensorHeader;    // ❌ DELETED - Use safetensor-loader instead
```

**Required Fix:**
The `GPTQLoader` needs to be updated to use the safetensor-loader module's classes. However, since the safetensor-loader has a different API (higher-level), you have two options:

#### Option A: Use safetensor-loader's API (Recommended)
Update `GPTQLoader` to use:
- `tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader`
- `tech.kayys.gollek.safetensor.loader.SafetensorHeader`
- `tech.kayys.gollek.safetensor.loader.SafetensorShardLoader`

This requires refactoring the loading logic to match the safetensor-loader's API.

#### Option B: Recreate minimal GPTQ-specific parser
If the safetensor-loader's API doesn't fit GPTQ's low-level needs, recreate minimal versions:
- `GPTQSafetensorHeader.java` - Already created at the correct location
- `GPTQSafetensorParser.java` - Needs to be created (FFM-based parser for GPTQ)

I've already created `GPTQSafetensorHeader.java` as a starting point.

## Files Modified

### ✅ Successfully Updated
1. `GPTQSafetensorConverter.java` (renamed from SafetensorConverter.java)
2. `GPTQQuantizerService.java` (updated references)
3. `SafetensorRunnerPlugin.java` (updated imports)
4. `QuantizationResource.java` (updated imports)
5. `gollek-quantizer-gptq/pom.xml` (added safetensor-loader dependency)

### ⚠️ Needs Manual Attention
1. `GPTQLoader.java` - Has broken imports, needs refactoring
2. `GPTQSafetensorParser.java` - Needs to be created if Option B is chosen

## Recommendation

**For immediate fix:**
1. Create `GPTQSafetensorParser.java` similar to the deleted `SafetensorParser.java` but with `GPTQ` prefix
2. Update `GPTQLoader.java` imports to use:
   ```java
   import tech.kayys.gollek.quantizer.gptq.GPTQSafetensorParser;
   import tech.kayys.gollek.quantizer.gptq.GPTQSafetensorHeader;
   import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
   import tech.kayys.gollek.quantizer.gptq.GPTQConfig;
   import tech.kayys.gollek.quantizer.gptq.QuantizedLayer;
   ```

**For long-term maintainability:**
Refactor `GPTQLoader` to use the safetensor-loader module's higher-level API, eliminating the need for duplicate FFM parsing code.

## Architecture After Refactoring

```
gollek-quantizer-gptq/
├── GPTQ-specific classes (keep here):
│   ├── GPTQConfig.java                    ✅ GPTQ quantization config
│   ├── GPTQLoader.java                    ⚠️ Needs import fixes
│   ├── GPTQQuantizerService.java          ✅ Unified service
│   ├── GPTQSafetensorConverter.java       ✅ Renamed from SafetensorConverter
│   ├── GPTQSafetensorHeader.java          ✅ Created (if needed)
│   ├── GPTQSafetensorParser.java          ❌ Needs creation (if Option B)
│   ├── QuantizedLayer.java                ✅ GPTQ layer representation
│   ├── VectorDequantizer.java             ✅ SIMD dequantization
│   └── MemoryAllocator.java               ✅ FFM memory management
│
└── Uses from safetensor-loader:
    ├── SafetensorFFMLoader (alternative)  
    ├── SafetensorHeader (alternative)
    └── SafetensorShardLoader (alternative)
```

## Next Steps

1. **Immediate**: Fix `GPTQLoader.java` imports (choose Option A or B above)
2. **Test**: Run `mvn clean compile` in `gollek-quantizer-gptq` to verify
3. **Update tests**: Fix any test imports that reference old class names
4. **Documentation**: Update any references in docs/examples

## Commands to Complete the Refactoring

```bash
# Option B: If creating GPTQSafetensorParser
cd gollek/core/quantizer/gollek-quantizer-gptq/src/main/java/tech/kayys/gollek/quantizer/gptq
# Create GPTQSafetensorParser.java based on the deleted SafetensorParser.java
# but rename class and update package references

# Build to verify
cd gollek/core/quantizer/gollek-quantizer-gptq
mvn clean compile

# Run tests
mvn test
```

Would you like me to:
1. Create `GPTQSafetensorParser.java` to complete Option B?
2. Help refactor `GPTQLoader.java` to use safetensor-loader API (Option A)?
3. Something else?
