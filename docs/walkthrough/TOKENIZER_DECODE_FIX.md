# Tokenizer decode() Type Fix

## Issue

The `HuggingFaceTokenizer.decode()` method expects `long[]` but several files were passing `int[]`, causing compilation errors.

## Error Message

```
The method decode(long[]) in the type HuggingFaceTokenizer is not applicable for the arguments (int[])
String tech.kayys.gollek.tokenizer.HuggingFaceTokenizer.decode(long[] tokenIds)
```

## Files Fixed

### 1. LogprobsEngine.java
**Location:** `inference-gollek/extension/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/LogprobsEngine.java`

**Before:**
```java
return tokenizer.decode(new int[] { tokenId });
```

**After:**
```java
return tokenizer.decode(new long[] { tokenId });
```

### 2. MultimodalInferenceEngine.java
**Location:** `inference-gollek/extension/runner/safetensor/gollek-safetensor-vision/src/main/java/tech/kayys/gollek/safetensor/vision/MultimodalInferenceEngine.java`

**Before:**
```java
String text = tokenizer.decode(new int[] { next });
```

**After:**
```java
String text = tokenizer.decode(new long[] { next });
```

### 3. StructuredOutputSampler.java
**Location:** `inference-gollek/extension/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/warmup/StructuredOutputSampler.java`

**Before:**
```java
String decoded = tokenizer.decode(new int[] { id });
```

**After:**
```java
String decoded = tokenizer.decode(new long[] { id });
```

### 4. WhisperEngine.java (Already Correct)
**Location:** `inference-gollek/extension/runner/safetensor/gollek-safetensor-audio/src/main/java/tech/kayys/gollek/safetensor/audio/WhisperEngine.java`

This file was already using `long[]` correctly:
```java
text.append(tokenizer.decode(new long[] { tokenId }));
```

## Why long[] instead of int[]?

The tokenizer uses `long[]` for several reasons:

1. **Cross-platform compatibility**: Some platforms/languages use 64-bit integers by default
2. **Large vocabularies**: Future-proof for vocabularies > 2^31 tokens
3. **JNI/FFI alignment**: Better alignment with native code interfaces
4. **Consistency**: Matches the underlying native library expectations

## Verification

After the fix, all tokenizer decode calls now use `long[]`:

```bash
# Search for all decode calls
grep -r "tokenizer.decode\|tok.decode" --include="*.java"

# All should now show: new long[] { ... }
```

## Impact

- ✅ Fixes compilation errors
- ✅ Ensures type safety
- ✅ Prevents potential runtime issues
- ✅ Maintains consistency across codebase

## Related Methods

The `HuggingFaceTokenizer` interface:

```java
public class HuggingFaceTokenizer {
    // Encode text to token IDs
    public long[] encode(String text, boolean addSpecialTokens) { ... }
    
    // Decode token IDs back to text
    public String decode(long[] tokenIds) { ... }
    
    // Get vocabulary size
    public int vocabSize() { ... }
    
    // Get EOS token ID
    public long eosId() { ... }
}
```

Note that `encode()` returns `long[]` and `eosId()` returns `long`, so using `long[]` for `decode()` is consistent.

## Best Practice

Always use `long[]` when working with token IDs:

```java
// ✅ Correct
long[] tokenIds = tokenizer.encode(text, true);
String text = tokenizer.decode(tokenIds);

// ❌ Incorrect
int[] tokenIds = ...;  // Type mismatch
String text = tokenizer.decode(tokenIds);  // Compilation error
```

## Testing

After applying fixes, verify compilation:

```bash
mvn clean compile -pl inference-gollek/extension/runner/safetensor/gollek-safetensor-engine
mvn clean compile -pl inference-gollek/extension/runner/safetensor/gollek-safetensor-vision
mvn clean compile -pl inference-gollek/extension/runner/safetensor/gollek-safetensor-audio
```

All modules should compile without errors.

---

**Fixed:** March 20, 2026
**Files Modified:** 3
**Lines Changed:** 3
