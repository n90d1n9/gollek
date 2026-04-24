# Gollek ML Framework

> **PyTorch-like API in pure Java 25** — with JDK 25 Vector API (SIMD) and FFM (Foreign Function & Memory) support.

[![Build Status](https://github.com/bhangun/gollek/actions/workflows/ci.yml/badge.svg)](https://github.com/bhangun/gollek/actions)
[![License](https://img.shields.io/github/license/bhangun/gollek)](LICENSE)
[![Java](https://img.shields.io/badge/Java-25-blue)](https://openjdk.org/projects/jdk/25/)

## Overview

Gollek ML is a **pure-Java machine learning framework** designed to provide PyTorch/JAX/TensorFlow-like APIs for the JVM ecosystem. It leverages **JDK 25 preview features** — the Vector API for SIMD-accelerated tensor operations and FFM for native library interop — to deliver high-performance ML workflows without leaving Java.

### Key Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Tensor API** | ✅ | `GradTensor` with autograd, broadcasting, reshape, cat, stack, index_select, einsum |
| **Neural Network Layers** | ✅ | Linear, Conv1d/2d/3d, Pool*, BatchNorm, LayerNorm, Dropout, GELU, ReLU, SiLU, Embedding, MultiHeadAttention, LSTM, GRU, Flatten |
| **Autograd Engine** | ✅ | Tape-based reverse-mode AD with `backward()`, `noGrad()`, gradient accumulation |
| **Optimizers** | ✅ | SGD, Adam, AdamW, RMSprop, Lookahead |
| **Trainer** | ✅ | PyTorch Lightning-style training loop with callbacks, LR scheduling, early stopping, mixed precision, gradient clipping |
| **DataLoader** | ✅ | Batching, shuffling, custom collate, virtual thread parallelism |
| **Loss Functions** | ✅ | MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, FocalLoss, HuberLoss, TripletLoss, and more |
| **Model Hub** | ✅ | HuggingFace integration with SafeTensors + GGUF support |
| **Serialization** | ✅ | Save/load stateDict, SafeTensors, GGUF formats |
| **Vector API (SIMD)** | ✅ | JDK 25 `jdk.incubator.vector` for add, mul, div, relu, matmul, FMA, clamp, abs |
| **Export** | ✅ | ONNX, GGUF, LiteRT export |
| **Multimodal** | ✅ | Vision, Audio, Video, Text pipelines |
| **GPU Backend** | 🚧 | FFM-based CUDA/Metal planned (JDK 25 FFM is stable) |

## Quick Start

### Prerequisites

- **JDK 25** (EA or GA) — Vector API and FFM require `--enable-preview`
- **Maven 3.8+**

### Installation

```bash
# Clone the repository
git clone https://github.com/bhangun/gollek.git
cd gollek

# Build (requires JDK 25)
mvn clean install -pl framework/lib -am -DskipTests
```

### Maven Dependency

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-ml-api</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

### Tensor Basics

```java
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

// Create tensors
var x = Gollek.tensor(new float[]{1, 2, 3, 4}, 2, 2);
var zeros = Gollek.zeros(3, 4);
var ones = Gollek.ones(2, 3);
var randn = Gollek.randn(128, 64);     // Normal distribution
var rand = Gollek.rand(128, 64);        // Uniform [0, 1)
var eye = Gollek.eye(10);

// Operations
var a = Gollek.randn(3, 4);
var b = Gollek.randn(3, 4);
var c = a.add(b).relu();
var d = a.matmul(b.transpose());
var e = c.sum();

// Autograd
var w = Gollek.randn(4, 2).requiresGrad(true);
var x2 = Gollek.randn(3, 4);
var y = x2.matmul(w).relu().sum();
y.backward();
w.grad();  // ∂y/∂w
```

### Building Neural Networks

```java
import tech.kayys.gollek.ml.nn.layer.*;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.optim.*;

// Sequential model
var model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Dropout(0.2),
    new Linear(256, 128),
    new ReLU(),
    new Dropout(0.2),
    new Linear(128, 10)
);

System.out.println("Parameters: " + model.parameterCountFormatted());

// Optimizer
var optimizer = new Adam(model.parameters(), 1e-3);
var lossFn = new CrossEntropyLoss();

// Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    model.train();
    double totalLoss = 0;
    int batches = 0;

    for (var batch : trainLoader) {
        var input = batch.get(0);
        var target = batch.get(1);

        optimizer.zeroGrad();
        var pred = model.forward(input);
        var loss = lossFn.compute(pred, target);
        loss.backward();
        optimizer.step();

        totalLoss += loss.item();
        batches++;
    }

    System.out.printf("Epoch %d — Loss: %.4f%n", epoch, totalLoss / batches);
}
```

### DataLoader

```java
import tech.kayys.gollek.ml.data.DataLoader;

var dataset = new DataLoader.TensorDataset(inputs, targets);
var loader = DataLoader.builder(dataset)
    .batchSize(32)
    .shuffle(true)
    .dropLast(true)
    .build();

for (List<GradTensor> batch : loader) {
    var input = batch.get(0);
    var target = batch.get(1);
    // training step
}
```

### Model Hub

```java
import tech.kayys.gollek.sdk.hub.ModelHub;

// Download and load weights
var model = new Sequential(
    new Linear(768, 768),
    new LayerNorm(768)
);
model.loadSafetensors("path/to/model.safetensors");
```

### Training with Trainer (PyTorch Lightning Style)

```java
import tech.kayys.gollek.sdk.train.*;

Trainer trainer = Trainer.builder()
    .model(input -> model.forward(input))
    .optimizer(new Adam(model.parameters(), 1e-3))
    .loss((preds, targets) -> preds.sub(targets).pow(2).mean())
    .callbacks(List.of(
        EarlyStopping.patience(5),
        ModelCheckpoint.at("best.pt"),
        ConsoleLogger.create()
    ))
    .epochs(100)
    .gradientClip(1.0)
    .mixedPrecision(true)
    .build();

trainer.fit(trainLoader, valLoader);
```

## Architecture

```
gollek-framework/
├── gollek-ml-autograd/     # GradTensor, Functions, VectorOps (SIMD)
├── gollek-ml-tensor/       # Tensor wrapper, Device, MemoryManager
├── gollek-ml-nn/           # NNModule, layers (Linear, Conv*, Norm*, Dropout, etc.)
├── gollek-ml-cnn/          # Convolution operations (im2col + GEMM)
├── gollek-ml-rnn/          # RNN, LSTM, GRU cells and layers
├── gollek-ml-data/         # DataLoader, Dataset, TensorDataset
├── gollek-ml-optimize/     # Optimizers (Adam, SGD, etc.), GradScaler
├── gollek-ml-train/        # Trainer, Callbacks, LRScheduler, EarlyStopping
├── gollek-ml-hub/          # ModelHub, HuggingFace integration, SafeTensors
├── gollek-ml-nlp/          # NLP pipelines (text-gen, embedding, classification)
├── gollek-ml-vision/       # Vision models (ResNet), transforms
├── gollek-ml-multimodal/   # Vision/Audio/Video builders
├── gollek-ml-export/       # ONNX, GGUF, LiteRT exporters
├── gollek-ml-api/          # Gollek.java — torch-like entry point
└── gollek-ml-examples/     # MNIST, Transformer examples
```

## JVM Flags Required

JDK 25 Vector API and FFM require preview flags:

```bash
java --enable-preview --add-modules jdk.incubator.vector -jar your-app.jar
```

Maven compiler plugin is pre-configured in `framework/pom.xml`:

```xml
<compilerArgs>
    <arg>--enable-preview</arg>
    <arg>--add-modules</arg>
    <arg>jdk.incubator.vector</arg>
</compilerArgs>
```

## Comparison with Existing Frameworks

| Feature | Gollek | PyTorch | DJL |
|---------|--------|---------|-----|
| **Language**          | Java 25               | Python               | Java |
| **Autograd**          | ✅ Tape-based         | ✅ Dynamic            | ❌ (Inference-only) |
| **SIMD (Vector API)** | ✅ JDK 25             | ❌ (Native)           | ❌ |
| **GPU (FFM)**         | 🚧 Planned            | ✅ CUDA               | ✅ (via engines) |
| **Model Hub**         | ✅ HuggingFace        | ✅ torchvision        | ✅ Model Zoo |
| **Training API**      | ✅ Trainer            | ✅ Lightning          | ❌ |
| **Serialization**     | ✅ SafeTensors, GGUF  | ✅ .pt, .safetensors   | ✅ |
| **Export**            | ✅ ONNX, GGUF         | ✅ ONNX               | ✅ ONNX |

## Roadmap

| Milestone | Target | Details |
|-----------|--------|---------|
| **0.2.0** | Q2 2026 | FFM-based CUDA backend, full einsum, distributed training |
| **0.3.0** | Q3 2026 | Metal backend (Apple Silicon), Python bindings (GraalPy) |
| **0.4.0** | Q4 2026 | `pip install gollek`, HuggingFace transformers integration |
| **1.0.0** | Q1 2027 | Production-ready, comprehensive benchmarks, enterprise features |

## License

Apache License 2.0 — see [LICENSE.md](../../LICENSE.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **PyTorch** — API design inspiration
- **JDK 25 Vector API** — SIMD acceleration
- **HuggingFace** — Model Hub integration
- **SafeTensors** — Weight format support
