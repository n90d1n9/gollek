# Working Examples - Status & Usage

## ✅ What Works

### 1. CLI (Recommended - Full Featured)

```bash
java -jar gollek/ui/gollek-cli/target/gollek-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat playing ball" \
  --steps 10 \
  --output cat.png
```

**Features**: All parameters, auto-open, progress display, error handling

### 2. Shell Script (Quick Usage)

```bash
cd gollek/examples/jbang/multimodal

# Basic
./sd_generate.sh "a cat" 10 cat.png

# Advanced
./sd_generate.sh "cyberpunk city" 30 city.png 123 8.0
# Usage: ./sd_generate.sh <prompt> <steps> <output> <seed> <cfg>
```

**Features**: Wrapper around CLI, simple interface

### 3. Jupyter Notebook (Interactive - NEEDS FIX)

**Status**: Kernel configured with Maven dependencies but needs Jupyter restart.

**To use**:
1. Stop Jupyter: `Ctrl+C`
2. Restart: `jupyter notebook gollek/examples/jupyter/`
3. Open `07-stable-diffusion.ipynb`
4. Select kernel: `Kernel → Change kernel → Gollek ML (Java 25)`
5. Run cells one by one

**Note**: The kernel now uses Maven coordinates (not file:// paths) which should resolve correctly.

### 4. JBang Scripts (NOT WORKING - Quarkus Extension Issue)

**Problem**: The `gollek-runner-onnx` module is a Quarkus extension, not a standalone JAR. JBang can't run it directly without the Quarkus runtime.

**Status**: 
- ❌ `sd_direct.java` - Can't instantiate Quarkus beans outside CDI
- ❌ `stable_diffusion_generation.java` - SDK needs Quarkus CDI container
- ✅ **Shell wrapper** (`sd_generate.sh`) - Works because it calls the CLI uber-JAR

## 📋 Summary Table

| Method | Status | Ease | Features |
|--------|--------|------|----------|
| **CLI** | ✅ Working | Medium | All features |
| **Shell Script** | ✅ Working | Easy | Simplified CLI |
| **Jupyter** | ⚠️ Needs restart | Easy | Interactive |
| **JBang Direct** | ❌ Quarkus issue | Hard | N/A |

## 🚀 Recommended Workflow

### For Quick Testing
```bash
# Use the shell script
cd gollek/examples/jbang/multimodal
./sd_generate.sh "your prompt" 10 output.png
```

### For Production/Integration
```bash
# Use CLI directly with full control
java -jar gollek/ui/gollek-cli/target/gollek-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "your prompt" \
  --seed 42 \
  --steps 30 \
  --guidance-scale 7.5 \
  --output result.png
```

### For Interactive Learning (After Jupyter Restart)
```bash
# Stop current Jupyter
# Restart with:
jupyter notebook gollek/examples/jupyter/

# Then open 07-stable-diffusion.ipynb
# Select "Gollek ML (Java 25)" kernel
```

## 🔧 Why JBang Doesn't Work Directly

The `StableDiffusionOnnxRunner` is a **Quarkus CDI bean**:
```java
@ApplicationScoped  // Requires CDI container
public class StableDiffusionOnnxRunner extends AbstractGollekRunner {
    @Inject  // CDI injection
    Tokenizer tokenizer;
    // ...
}
```

To run it outside Quarkus, you'd need to:
1. Manually initialize CDI container (complex)
2. Or create a standalone version without CDI (duplicate code)

**Solution**: Use the CLI which already has Quarkus runtime initialized.

## 📝 Jupyter Kernel Fix Applied

**Changed**: From `file:///path/to/jar` to Maven coordinates

**Before**:
```json
"JJAVA_DEPS": "file:///Users/bhangun/.gollek/jupyter/gollek-sdk-core-0.1.0-SNAPSHOT.jar,..."
```

**After**:
```json
"JJAVA_DEPS": "tech.kayys.gollek:gollek-sdk-java-local:0.1.0-SNAPSHOT,tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT,..."
```

**Benefit**: Maven resolves transitive dependencies automatically.

## ✅ Verified Working

```bash
# Test the CLI (proven working)
java -jar gollek/ui/gollek-cli/target/gollek-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat" \
  --steps 3 \
  --output test.png

# Expected output:
# ✓ Image saved to: test.png
# File size: ~600-700KB
# Format: PNG 512x512
```

---

**Last Updated**: April 14, 2026  
**Working Methods**: CLI, Shell Script  
**Needs Restart**: Jupyter Notebook  
**Not Working**: Direct JBang (Quarkus limitation)
