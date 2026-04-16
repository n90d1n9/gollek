# Gollek Jupyter - Hybrid Architecture

## The Game-Changer Approach

Instead of fighting Python in Jupyter, we **use it strategically**:

```
┌─────────────────────────────────────────────────┐
│              Jupyter Notebook                    │
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ Python Kernel│    │ Bash Cells (%%bash)  │   │
│  │ (Lightweight)│    │                      │   │
│  │              │    │  java -jar gollek    │   │
│  │ - Display    │◄───┤  - SD Inference      │   │
│  │ - Visualize  │    │  - Pure JDK 25       │   │
│  │ - Markdown   │    │  - ONNX Runtime      │   │
│  └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Why This Works

**Python's Strengths:**
- Jupyter kernel stability
- Image display (`IPython.display`)
- Markdown rendering
- Data visualization

**Java's Strengths:**
- AI inference (ONNX Runtime via Panama FFM)
- Type safety
- Production performance
- Enterprise deployment

### This Is How Production Systems Work

Many production AI systems use this pattern:
- **Orchestration**: Python/bash for workflow
- **Inference**: Optimized native code (C++, Java, Rust)
- **Display**: Web technologies (HTML/JS)

Gollek follows this proven pattern.

## Usage

### Start Notebook

```bash
jupyter notebook gollek/examples/jupyter/07-stable-diffusion.ipynb
```

### Select Kernel

Choose: **"Gollek SD (Python + Java)"**

### Run Cells

1. **Markdown cells**: Explain concepts
2. **Bash cells** (`%%bash`): Execute Java CLI
3. **Python cells**: Display results

All AI inference runs on **JDK 25**, Python only handles display.

## Technical Details

### What Runs on Java

```bash
java -jar gollek-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat" \
  --steps 10 \
  --output cat.png
```

- ONNX Runtime (via Panama FFM)
- CLIP text encoder
- UNet denoiser
- VAE decoder
- PNG encoding

### What Runs on Python

```python
from IPython.display import Image, display
display(Image(filename="cat.png"))
```

- Image rendering
- HTML layout
- Notebook interactivity

## Benefits

### For Java Developers
- ✅ Use familiar JDK 25 for AI
- ✅ No Python AI libraries needed
- ✅ Type-safe inference code
- ✅ Production-grade performance

### For Python Users
- ✅ Still use Jupyter interface
- ✅ Rich visualization
- ✅ Interactive exploration
- ✅ Easy sharing

### For Everyone
- ✅ Best of both worlds
- ✅ No compromises
- ✅ Production-ready pipeline
- ✅ Educational value

## Future: Pure Java Notebook

For those who want **100% Java**, we're developing:

1. **Minimal SDK**: Quarkus-free, just ONNX + FFM
2. **Uber-JAR**: Single JAR with all deps
3. **IJava Kernel**: Java kernel with fast startup

This notebook is the **pragmatic solution** that works today while we build the pure Java future.

---

**Java for AI isn't about replacing Python everywhere - it's about using the right tool for each job.**

- **AI Inference**: Java (performance, type safety)
- **Orchestration**: Python (Jupyter, visualization)
- **Production**: Java (deployment, monitoring)
