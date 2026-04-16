# Gollek Jupyter - PURE Java Solution

## ✅ TRUE Java Notebook - No Python!

```
┌──────────────────────────────────────────┐
│          Jupyter Notebook                 │
│                                           │
│  IJava Kernel (Pure Java)                │
│    ↓                                     │
│  Java cells with //DEPS                  │
│    ↓                                     │
│  Gollek SDK + ONNX Runtime               │
│    ↓                                     │
│  100% JDK 25 for ALL AI                  │
└──────────────────────────────────────────┘
```

## How It Works

### Each Cell Loads Its Dependencies
```java
//DEPS tech.kayys.gollek:gollek-sdk-java-local:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT

import tech.kayys.gollek.sdk.core.GollekSdk;

GollekSdk sdk = GollekSdk.builder().local().build();
// All AI features available
```

### Benefits
1. ✅ **100% Java** - No Python anywhere
2. ✅ **Fast startup** - No heavy dependency loading
3. ✅ **On-demand deps** - Each cell loads what it needs
4. ✅ **JDK 25 features** - Panama FFM, Vector API, Records
5. ✅ **Game changer** - Pure Java AI platform

## Usage

### Start Notebook
```bash
jupyter notebook gollek/examples/jupyter/09-gollek-pure-java.ipynb
```

### Select Kernel
Choose: **"Gollek Java 25"**

### Run Cells
Execute sequentially - each cell is pure Java!

## Example Cells

### Cell 1: Setup
```java
//DEPS tech.kayys.gollek:gollek-sdk-java-local:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-onnx:0.1.0-SNAPSHOT

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModalityType;

System.out.println("Java " + Runtime.version());
System.out.println("✅ SDK ready");
```

### Cell 2: Generate Image
```java
GollekSdk sdk = GollekSdk.builder().local().build();

InferenceRequest request = InferenceRequest.builder()
    .model("CompVis/stable-diffusion-v1-4")
    .prompt("a cat playing ball")
    .parameter("seed", 42L)
    .parameter("steps", 10)
    .streaming(true)
    .build();

sdk.streamCompletion(request).subscribe()...
```

### Cell 3: Display Image
```java
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

BufferedImage img = ImageIO.read(new java.io.File("output.png"));
System.out.println("Dimensions: " + img.getWidth() + "x" + img.getHeight());
System.out.println("![Image](output.png)");
```

## Architecture

```
Notebook (.ipynb)
  ↓
IJava Kernel
  ↓ parses //DEPS
  ↓ loads Maven dependencies
  ↓ executes Java code
  ↓
Gollek SDK
  ↓
ONNX Runtime (Panama FFM)
  ↓
AI Inference
```

## JDK 25 Features Demonstrated

### Project Panama (FFM)
```java
// Direct native library binding
MemorySegment tensor = arena.allocate(bytes, 4);
ort.run(session, inputs, outputs);
```

### Vector API
```java
// SIMD tensor operations
FloatVector a = FloatVector.fromArray(SPECIES_256, data, 0);
FloatVector result = a.mul(scale);
```

### Records
```java
record SDResult(String outputFile, long durationMs, long fileSize) {}

SDResult r = generateImage(...);
System.out.println(r.fileSize());
```

### Pattern Matching
```java
switch (chunk) {
    case TextChunk(var delta) -> System.out.print(delta);
    case ImageChunk(var base64) -> saveImage(base64);
}
```

## Files

| File | Purpose |
|------|---------|
| `09-gollek-pure-java.ipynb` | Pure Java notebook |
| `kernel.json` | IJava kernel config |

## Requirements

- ✅ JDK 25 with preview features
- ✅ IJava kernel installed
- ✅ Gollek SDK in Maven local repo
- ✅ ONNX Runtime installed

## This Is The Game Changer

**Before**: Python dominated AI because of notebooks
**Now**: Java has notebooks with full AI capabilities

### Advantages Over Python

| Aspect | Python | Gollek Java |
|--------|--------|-------------|
| Type Safety | ❌ Runtime errors | ✅ Compile-time |
| Performance | Good | Excellent (native) |
| Deployment | pip/conda | JAR/Maven |
| IDE Support | Good | Excellent (IntelliJ) |
| Production | Requires work | Enterprise-ready |
| Notebook | ✅ Jupyter | ✅ Jupyter |

---

**Java is now a first-class AI platform - notebooks, type safety, production performance!** 🚀
