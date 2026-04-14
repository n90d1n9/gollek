# Gollek Jupyter Notebooks

Interactive notebooks demonstrating Gollek ML framework capabilities.

## Prerequisites

1. **Install Jupyter with Java kernel**:
   ```bash
   # Install Jupyter
   pip install jupyter
   
   # Install Java kernel (IJava)
   jbang app install --force jupyter-java@quarkusio
   jbang jupyter-java@quarkusio
   
   # Or manually:
   curl -Ls https://sh.jbang.dev | bash -s - app install --force jupyter-java@quarkusio
   ```

2. **Launch Jupyter**:
   ```bash
   cd gollek/examples/jupyter/
   jupyter notebook
   ```

## Available Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| [01-getting-started.ipynb](01-getting-started.ipynb) | SDK basics, tensors, autograd, simple NN | ✅ Complete |
| [06-llm-integration.ipynb](06-llm-integration.ipynb) | Unified Runner, Batching, Fusion, Quantization | ✅ Complete |

## Planned Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| 02-neural-network-training.ipynb | Classifier, optimizers, loss, training loop | 📋 Planned |
| 03-advanced-activations.ipynb | ReLU, ELU, Mish comparisons | 📋 Planned |
| 04-transformers.ipynb | Multi-head attention, masking, positional embeddings | 📋 Planned |
| 05-batch-normalization.ipynb | Training vs eval modes, running statistics | 📋 Planned |

## Usage

### Running in Jupyter

1. Open the notebook in Jupyter
2. Execute cells sequentially (Shift+Enter)
3. Modify parameters and re-run to experiment

### Running as Script

Each notebook can be converted to a JBang script:

```bash
# Extract Java code from notebook
jupyter nbconvert --to script 01-getting-started.ipynb

# Run as JBang script
jbang 01-getting-started.java
```

## Framework Requirements

- **Java 25** with preview features enabled
- **Gollek ML framework** (available via Maven)
- **Optional**: CUDA toolkit for GPU acceleration

## Troubleshooting

### Kernel not found
```bash
# List available kernels
jupyter kernelspec list

# Install Java kernel if missing
jbang jupyter-java@quarkusio
```

### Module not found
Ensure Gollek dependencies are available in your Maven repository or configure the notebook to use local JARs.

### GPU not available
Check CUDA installation:
```bash
nvidia-smi
```
