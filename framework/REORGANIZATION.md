# Gollek ML Framework — Module Reorganization

## ✅ COMPLETED

### Build Structure
- **All artifactId collisions fixed** — every module has a unique ID
- **All dependency declarations added** — all 24 modules have proper `<dependency>` entries
- **Framework in root pom** — `framework` module added to `/gollek/pom.xml`
- **Parent chain** — `gollek-framework-root` → `gollek-framework-ml-parent`
- **Preview features** — `--enable-preview` and `jdk.incubator.vector` configured

### Package Namespace
- **All 202 source files** use `tech.kayys.gollek.ml.*` — zero `lib.*` or `sdk.*` remaining
- **Every package declaration matches its directory path**
- **41 canonical packages** across 24 modules

### Deleted Junk
- `vision/` subdirectory in `gollek-ml-cnn` — 5 deprecated stubs that returned zero-output tensors
- `other/` subdirectory in `gollek-ml-cnn` — 1 deprecated stub
- `gollek-ml- copy/`, `gollek-ml-/`, `gollek-ml-core/` — garbage directories

### Fixed Import Paths
- `GGUFExporter.java` — `ml.nn.util.gguf.*` → `ml.gguf.*`
- `LiteRTSdk.java` — `lib.litert.*` → `ml.litert.*`
- `LiteRTInferenceEngine.java` — `lib.litert.*` → `ml.litert.*`, commented out non-existent provider imports
- `vision/ResNet.java` — `lib.vision.ops.*` → `ml.vision.ops.*`

---

## Module Structure (24 modules, 202 source files, 41 packages)

### Layer 0: Core
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-tensor` | `gollek-ml-tensor` | 8 | `ml.tensor` |
| `gollek-ml-autograd` | `gollek-ml-autograd` | 7 | `ml.autograd` |

### Layer 1: NN Building Blocks
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-nn` | `gollek-ml-nn` | ~30 | `ml.nn`, `ml.nn.layer`, `ml.nn.loss`, `ml.nn.backend` |
| `gollek-ml-optimize` | `gollek-ml-optimize` | 22 | `ml.optim`, `ml.optimize` |
| `gollek-ml-data` | `gollek-ml-data` | 14 | `ml.data`, `ml.data.multimodal` |

### Layer 2: Architectures
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-cnn` | `gollek-ml-cnn` | 10 | `ml.cnn` |
| `gollek-ml-rnn` | `gollek-ml-rnn` | 10 | `ml.rnn`, `ml.rnn.cells`, `ml.rnn.layers` |
| `gollek-ml-transformer` | `gollek-transformer` | 7 | `ml.transformer` |
| `gollek-ml-models` | `gollek-models` | 16 | `ml.models`, `ml.registry`, `ml.serving`, `ml.profiler` |

### Layer 3: Applications
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-train` | `gollek-ml-train` | 11 | `ml.train`, `ml.train.other` |
| `gollek-ml-nlp` | `gollek-ml-nlp` | 13 | `ml.nlp` |
| `gollek-ml-vision` | `gollek-ml-vision` | 5 | `ml.vision.transforms`, `ml.vision.models`, `ml.vision.ops` |
| `gollek-ml-audio` | `gollek-audio` | 1 | `ml.audio` |
| `gollek-ml-multimodal` | `gollek-ml-multimodal` | 8 | `ml.multimodal` |

### Layer 4: Infrastructure
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-hub` | `gollek-ml-hub` | 6 | `ml.hub` |
| `gollek-ml-safetensor` | `gollek-safetensor` | 2 | `ml.safetensors` |
| `gollek-ml-gguf` | `gollek-gguf` | 5 | `ml.gguf` |
| `gollek-ml-export` | `gollek-ml-export` | 6 | `ml.export` |
| `gollek-ml-metrics` | `gollek-metrics` | 9 | `ml.metrics` |
| `gollek-ml-augment` | `gollek-ml-augment` | 2 | `ml.augment` |
| `gollek-ml-litert` | `gollek-ml-litert` | 6 | `ml.litert`, `ml.litert.config`, `ml.litert.inference`, `ml.litert.model` |
| `gollek-ml-api` | `gollek-ml-ml` | 2 | `ml` |

### Support
| Module | artifactId | Files | Packages |
|--------|-----------|-------|----------|
| `gollek-ml-examples` | `gollek-examples` | 3 | `ml.examples` |
| `gollek-ml-onnx` | `gollek-onnx` | 0 | (stub) |

---

## Remaining Known Issues

### Duplicate Class Names (Different Packages — Works, but Confusing)
| Class | Locations |
|-------|-----------|
| `Trainer` | `ml.train.Trainer`, `ml.train.other.Trainer` |
| `EarlyStopping` | `ml.train.EarlyStopping`, `ml.metrics.EarlyStopping` |
| `Embedding` | `ml.nn.Embedding`, `ml.nlp.Embedding` |
| `StepLR` | `ml.optim.StepLR`, `ml.train.StepLR` |
| `CosineAnnealingLR` | `ml.optim.CosineAnnealingLR`, `ml.train.CosineAnnealingLR` |
| `LRScheduler` | `ml.optim.LRScheduler`, `ml.train.LRScheduler` |
| `Bidirectional` | `ml.rnn.Bidirectional`, `ml.rnn.layers.Bidirectional` |
| `NoGrad` | `ml.autograd.NoGrad`, `ml.tensor.NoGrad` |
| `VectorOps` | `ml.autograd.VectorOps`, `ml.tensor.VectorOps` |
| `Sequential` | `ml.nn.layer.Sequential`, `ml.tensor.Sequential` |
| `ResNet` | `ml.models.ResNet`, `ml.vision.models.ResNet` |
| `StateDict` | `ml.registry.StateDict` (hub imports `ml.nn.StateDict` — needs fixing) |

### Broken External Imports (Need External Modules to Compile)
- `ModelHub.java` — imports `tech.kayys.gollek.model.core.*`, `tech.kayys.gollek.spi.model.*`
- `SafeTensorBridge.java` — imports `tech.kayys.gollek.safetensor.loader.*`
- `MultimodalBuilder.java` — imports `tech.kayys.gollek.lib.api.GollekSdk`, `tech.kayys.gollek.spi.model.*`
- `examples/*.java` — imports `tech.kayys.gollek.ml.nn.optim.*` (doesn't exist)

### Test Package Mismatches
- 4 test files still use `lib.*` packages instead of `ml.*`

### Empty Module
- `gollek-ml-onnx` — 0 source files, intentional stub
