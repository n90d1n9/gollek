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
var loader = DataLoader.tensorBuilder(dataset)
    .batchSize(32)
    .shuffle(true)
    .dropLast(true)
    .build();

for (DataLoader.Batch batch : loader) {
    var input = batch.inputs();
    var target = batch.get(1);
    // training step
}
```

### Model Hub

```java
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.hub.HubConfig;
import tech.kayys.gollek.ml.nn.layer.*;

// Download and load weights
var model = new Sequential(
    new Linear(768, 768),
    new LayerNorm(768)
);

Gollek.Hub.loadInto(
    model,
    "Qwen/Qwen2.5-0.5B",
    HubConfig.builder()
        .revision("main")
        .build()
);
```

### Export

```java
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.export.ModelExporter;

ModelExporter exporter = Gollek.Export.model(model)
    .inputShape(1, 3, 224, 224)
    .build();

exporter.toONNX("model.onnx");
exporter.toGGUF("model.gguf", ModelExporter.Quantization.INT4);
exporter.toLiteRT("model.litert");
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
gollek/
├── ml/gollek-ml-autograd/      # GradTensor, autograd, GGUF compatibility
├── ml/gollek-ml-core/          # BaseEstimator, BaseTransformer foundations
├── ml/gollek-ml-nn/            # NNModule, layers, activations, losses
├── ml/gollek-ml-cnn/           # CNN-focused model/layer extensions
├── ml/gollek-ml-data/          # DataLoader, Dataset, Csv/Text datasets
├── ml/gollek-ml-optimize/      # Optimizers, memory/runtime helpers
├── ml/gollek-ml-selection/     # Metrics, CV, search, evaluation
├── ml/gollek-ml-hub/           # ModelHub, HF/local repositories, SafeTensors
├── ml/gollek-ml-export/        # ONNX, GGUF, LiteRT exporters and benchmark
├── ml/gollek-ml-byte-latent/   # Planned byte-latent language-model family
├── training/gollek-train-recursive-reasoning/ # Recursive reasoning family (GRAM-style), :ml:gollek-ml-recursive-reasoning
├── ml/gollek-ml-api/           # Gollek.java umbrella ML entry point
├── trainer/gollek-trainer-api/ # Canonical trainer session/config contracts
├── trainer/gollek-trainer/     # Trainer bridge/runtime facade
└── examples/gollek-ml-examples/# Samples and migration-era examples
```

## Verification

Run the Gradle-wired training module checks from `gollek/`:

```bash
./gradlew --no-daemon checkTrainingModules
```

Run the full training gate for CI/release validation:

```bash
./gradlew --no-daemon verifyTrainingGates
```

Run the optional advisory metadata gate without executing advisory candidate
tests:

```bash
./gradlew --no-daemon verifyTrainingAdvisoryGates
```

The advisory gate writes coverage first, then reuses that generated coverage
artifact for candidate selection and candidate-lock drift checks.

For a faster metadata-only CI preflight that avoids Gradle project
configuration, run:

```bash
scripts/verify-training-advisory-gates-fast.py
```

The fast path derives the hard-check and advisory reports from
`settings.gradle.kts`, checked-in training build files, and the training gate
registry. It validates both `training-module-checks.lock.json` and
`training-module-candidates.lock.json` drift without starting Gradle. It also
writes `training-module-fast-gate-summary.json` for CI jobs that need one
machine-readable pass/fail artifact.

Validate the fast metadata gate contract itself:

```bash
scripts/test-training-fast-gate.sh
```

Compare the fast metadata reports against the Gradle-generated reports when
the Gradle gate logic changes:

```bash
bash scripts/test-training-fast-gradle-parity.sh
```

Force a specific selected-candidate parity case:

```bash
bash scripts/test-training-fast-gradle-parity.sh --candidate audio,vision
```

Keep a parity artifact bundle with raw reports, normalized JSON, diff files,
logs, `training-gradle-parity-summary.json`, and a human-readable `README.md`:

```bash
bash scripts/test-training-fast-gradle-parity.sh \
  --candidate audio,vision \
  --artifact-dir /tmp/gollek-training-gradle-parity
```

Without `--candidate`, the parity test uses
`GOLLEK_TRAINING_PARITY_CANDIDATES` or picks the first available advisory
candidates from the generated coverage report. Without `--artifact-dir`, it
uses `GOLLEK_TRAINING_PARITY_ARTIFACT_DIR` when set.

CI runs the same fast metadata gate through
`.github/workflows/training-fast-gate.yml` for training registry, lock, and
training project metadata changes.

Gradle parity runs separately through
`.github/workflows/training-gradle-parity.yml`. That workflow sets up JDK 25,
generates the Gradle reports, generates the fast reports, and diffs the shared
JSON contracts. It writes the parity receipt into the GitHub Actions step
summary and uploads the parity artifact bundle even on failure. It is
intentionally separate from the fast gate so the metadata preflight can remain
cheap while Gradle alignment still has a repeatable CI entry point.

Inspect the registered training checks:

```bash
./gradlew --no-daemon printTrainingModuleChecks
```

Validate the registry without running module tests:

```bash
./gradlew --no-daemon validateTrainingModuleChecks
```

Write a machine-readable registry report:

```bash
./gradlew --no-daemon writeTrainingModuleChecksReport
```

The report includes a stable SHA-256 `registryFingerprint` over the registered
training check labels and task paths.

Validate the generated report/schema contract:

```bash
./gradlew --no-daemon validateTrainingModuleChecksReportContract
```

Write advisory coverage for all Gradle-included projects under `gollek/training`:

```bash
./gradlew --no-daemon writeTrainingModuleCoverageReport
```

Inspect that coverage report from the terminal:

```bash
./gradlew --no-daemon printTrainingModuleCoverageJson
```

Print a concise coverage summary with suggested checks to promote:

```bash
./gradlew --no-daemon printTrainingModuleCoverageSummary
```

Run advisory tests for all candidate modules before promoting them into the
hard gate:

```bash
./gradlew --no-daemon checkTrainingModuleCandidateTests
```

Run only selected advisory candidates by label:

```bash
./gradlew --no-daemon validateTrainingModuleCandidateSelection -PgollekTrainingCandidate=audio
./gradlew --no-daemon printTrainingModuleCandidateSelectionJson -PgollekTrainingCandidate=audio
./gradlew --no-daemon checkTrainingModuleCandidateTests -PgollekTrainingCandidate=audio
./gradlew --no-daemon checkTrainingModuleCandidateTests -PgollekTrainingCandidate=audio,vision
```

Print copy-pasteable registry snippets for promotion:

```bash
./gradlew --no-daemon printTrainingModuleCandidatePromotionSnippets -PgollekTrainingCandidate=audio
```

Generate a guarded fast promotion plan without editing the registry:

```bash
scripts/verify-training-advisory-gates-fast.py \
  --candidate audio \
  --promote-selected \
  --dry-run
```

Validate an existing promotion plan artifact without regenerating reports:

```bash
scripts/verify-training-advisory-gates-fast.py \
  --validate-promotion-plan build/reports/gollek/training-module-promotion-plan.json \
  --promotion-plan-validation-report-file build/reports/gollek/training-module-promotion-plan-validation.json
```

Apply an intentional selected-candidate promotion and refresh the checked-in
hard/advisory locks:

```bash
scripts/verify-training-advisory-gates-fast.py \
  --candidate audio \
  --promote-selected
```

The fast promotion path requires concrete `--candidate` labels; empty,
`all`, and `*` selections are rejected for promotion so broad advisory coverage
cannot accidentally become a hard gate change. Dry-run promotion writes a
`training-module-promotion-plan.json` artifact with the snippets, selected
tasks, follow-up commands, and a stable SHA-256 `promotionFingerprint` over the
selected candidates plus the logical registry/lock mutation targets. Apply mode
snapshots the training registry and both metadata locks first, then rolls those
files back if promotion validation or report writing fails.

The coverage report shows which training projects have tests and which of those
projects are currently registered in the hard training gate. It is advisory by
default so experimental modules can remain visible without immediately breaking
CI. The `candidateChecks` section lists ready-to-consider `TrainingModuleCheck`
entries for tested projects that are not yet part of the hard gate. The
candidate selection report also includes `nextCommands` with the selected
`-PgollekTrainingCandidate=...` value preserved and a stable SHA-256
`selectionFingerprint` over the selected candidate rows.

Write or refresh the checked-in registry lock after an intentional gate change:

```bash
./gradlew --no-daemon writeTrainingModuleChecksLock
```

Validate the current registry against the checked-in lock:

```bash
./gradlew --no-daemon validateTrainingModuleChecksLock
```

Validate the generated lock drift report contract:

```bash
./gradlew --no-daemon validateTrainingModuleChecksLockDriftReportContract
```

Write or validate the optional advisory candidate lock:

```bash
./gradlew --no-daemon writeTrainingModuleCandidateLock
./gradlew --no-daemon validateTrainingModuleCandidateLock
./gradlew --no-daemon printTrainingModuleCandidateLockDriftJson
```

The report and schema are written to:

- `gollek/build/reports/gollek/training-module-checks.json`
- `gollek/build/reports/gollek/training-module-checks.v2.schema.json`
- `gollek/build/reports/gollek/training-module-checks-lock-drift.json`
- `gollek/build/reports/gollek/training-module-coverage.json`
- `gollek/build/reports/gollek/training-module-candidate-selection.json`
- `gollek/build/reports/gollek/training-module-candidates-lock-drift.json`
- `gollek/build/reports/gollek/training-module-fast-gate-summary.json`

The checked-in lock lives at
`gollek/gradle/training-module-checks.lock.json`.
The optional advisory candidate lock lives at
`gollek/gradle/training-module-candidates.lock.json`.

The training gate build logic lives in
`gollek/buildSrc/src/main/kotlin/gollek.training-gates.gradle.kts`.
The compatibility entry point remains
`gollek/gradle/training-module-checks.gradle.kts`.

For the recursive reasoning family only:

```bash
./gradlew --no-daemon checkRecursiveReasoningTraining
```

## Adding a Gradle-Wired Training Module

1. Include the module in `gollek/settings.gradle.kts` with the intended
   project path.
2. Run `./gradlew --no-daemon printTrainingModuleCoverageSummary` to confirm
   the module appears in training coverage and review suggested gate entries.
3. Generate a dry-run promotion plan:
   `scripts/verify-training-advisory-gates-fast.py --candidate <label> --promote-selected --dry-run`.
4. Validate the focused advisory selection:
   `./gradlew --no-daemon validateTrainingModuleCandidateSelection -PgollekTrainingCandidate=<label>`.
5. Write or inspect the candidate selection report:
   `./gradlew --no-daemon printTrainingModuleCandidateSelectionJson -PgollekTrainingCandidate=<label>`.
6. Run a focused advisory readiness pass before promotion:
   `./gradlew --no-daemon checkTrainingModuleCandidateTests -PgollekTrainingCandidate=<label>`.
7. Apply the selected promotion and refresh both training metadata locks:
   `scripts/verify-training-advisory-gates-fast.py --candidate <label> --promote-selected`.
8. Run `./gradlew --no-daemon verifyTrainingGates` before handing the change
   to CI.

Fast Byte Latent Transformer style work belongs in a new byte-latent language
modeling family, not in the diffusion OPD stack. See
`gollek/docs/FAST_BYTE_LATENT_TRANSFORMER_INTEGRATION_PLAN.md` for the paper
mapping and the code paths that should be reused.

Generative Recursive Reasoning / GRAM style work belongs in a new recursive
reasoning family, not in the diffusion OPD stack or the byte-latent language
modeling family. See
`gollek/docs/GENERATIVE_RECURSIVE_REASONING_INTEGRATION_PLAN.md` for the paper
mapping and the code paths that should be reused.

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
