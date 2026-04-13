---

    GOLLEK ML FRAMEWORK — Full Assessment & Reorganization Plan

    Executive Summary

    The framework has 24 modules, ~240 Java source files, and meaningful autograd/training infrastructure. But it suffers from:
     - 7 modules sharing the same `artifactId` (gollek-ml-models) — Maven build collision
     - Two competing package namespaces (tech.kayys.gollek.lib.* vs tech.kayys.gollek.ml.*)
     - Files physically in wrong modules (audio in nn, gguf in nn, transformer in nn)
     - 8 modules with placeholder pom.xml (no dependencies declared)
     - 1 empty module (onnx = 0 files)

    ---

    Current Module State (Problems Identified)


    ┌───┬──────────┬──────────────────┬──────────────────┬─────┬────────────────────────────────────────────────────────────────┐
    │ # │ Module   │ Dir Name         │ ArtifactId       │ Fil │ Problems                                                       │
    ├───┼──────────┼──────────────────┼──────────────────┼─────┼────────────────────────────────────────────────────────────────┤
    │ 1 │ API F... │ gollek-ml-api    │ gollek-ml-ml    │ 1   │ ✅ OK                                                          │
    │ 2 │ Tenso... │ gollek-ml-tensor │ `gollek-ml-t... │ 8   │ Mixed packages (lib.core + ml.tensor)                          │
    │ 3 │ Autograd │ `gollek-ml-au... │ `gollek-ml-a... │ 6   │ ✅ OK                                                          │
    │ 4 │ Neura... │ gollek-ml-nn     │ gollek-ml-nn    │ 41  │ Bloated — contains audio, metrics, gguf, safetensor, transf... │
    │ 5 │ Conv     │ gollek-ml-cnn    │ gollek-ml-cnn   │ 15  │ Conv classes in wrong package (ml.nn not ml.cnn)               │
    │ 6 │ RNN      │ gollek-ml-rnn    │ gollek-ml-rnn   │ 6   │ Package lib.rnn should be ml.rnn                               │
    │ 7 │ Trans... │ `gollek-ml-tr... │ `gollek-ml-m... │ 4   │ Wrong artifactId, no deps, files in nn module                  │
    │ 8 │ Data     │ gollek-ml-data   │ gollek-ml-data  │ 12  │ ✅ OK                                                          │
    │ 9 │ Optimize │ `gollek-ml-op... │ `gollek-ml-o... │ 22  │ Mixed packages (ml.nn.optim + ml.optimize)                     │
    │ 1 │ Train    │ gollek-ml-train  │ gollek-ml-train │ 10  │ Package lib.train should be ml.train                           │
    │ 1 │ Hub      │ gollek-ml-hub    │ gollek-ml-hub   │ 5   │ Package lib.hub should be ml.hub                               │
    │ 1 │ NLP      │ gollek-ml-nlp    │ gollek-ml-nlp   │ 12  │ ✅ OK                                                          │
    │ 1 │ Vision   │ gollek-ml-vision │ `gollek-ml-v... │ 4   │ Package lib.vision should be ml.vision                         │
    │ 1 │ Audio    │ gollek-ml-audio  │ `gollek-ml-m... │ 1   │ Wrong artifactId, no deps, file is in nn module                │
    │ 1 │ Multi... │ `gollek-ml-mu... │ `gollek-ml-m... │ 8   │ Package lib.multimodal should be ml.multimodal                 │
    │ 1 │ Augment  │ `gollek-ml-au... │ `gollek-ml-a... │ 2   │ Package lib.augment should be ml.augment                       │
    │ 1 │ Export   │ gollek-ml-export │ `gollek-ml-e... │ 6   │ Mixed packages                                                 │
    │ 1 │ SafeT... │ `gollek-ml-sa... │ `gollek-ml-m... │ 2   │ Wrong artifactId, no deps, files in nn module                  │
    │ 1 │ GGUF     │ gollek-ml-gguf   │ `gollek-ml-m... │ 5   │ Wrong artifactId, no deps, files in nn module                  │
    │ 2 │ LiteRT   │ gollek-ml-litert │ `gollek-ml-l... │ 6   │ ✅ OK                                                          │
    │ 2 │ ONNX     │ gollek-ml-onnx   │ gollek-ml-onnx  │ 0   │ EMPTY — no source code                                         │
    │ 2 │ Models   │ gollek-ml-models │ `gollek-ml-m... │ 15  │ No deps declared in pom                                        │
    │ 2 │ Metrics  │ `gollek-ml-me... │ `gollek-ml-m... │ 9   │ Wrong artifactId, no deps, file in nn module                   │
    │ 2 │ Examples │ `gollek-ml-ex... │ `gollek-ml-m... │ 3   │ Wrong artifactId, no deps                                      │
    └───┴──────────┴──────────────────┴──────────────────┴─────┴────────────────────────────────────────────────────────────────┘

    ---

    Proposed Module Structure (Separation of Concerns)

    The framework should be organized into 5 layers following clean architecture:

      1 gollek-framework/
      2 │
      3 ├── layer-0/  CORE         — No ML-specific deps, foundational primitives
      4 │   ├── gollek-tensor/         # Tensor storage, Device, DType, MemoryManager
      5 │   └── gollek-autograd/       # GradTensor, Functions, VectorOps, Function tape
      6 │
      7 ├── layer-1/  NN PRIMITIVES  — Building blocks, no high-level logic
      8 │   ├── gollek-nn/             # NNModule, Parameter, all layer types, loss functions
      9 │   ├── gollek-optim/          # Optimizers, LR schedulers, GradScaler
     10 │   └── gollek-data/           # Dataset, DataLoader, CollateFn
     11 │
     12 ├── layer-2/  MODEL ARCH     — Composable architectures
     13 │   ├── gollek-cnn/            # Conv1d/2d/3d, Pool*, BatchNorm
     14 │   ├── gollek-rnn/            # RNN, LSTM, GRU cells & layers
     15 │   ├── gollek-transformer/    # TransformerBlock, Encoder, Decoder, MHA, Attention
     16 │   └── gollek-models/         # Pre-built: BERT, GPT2, LLaMA, ResNet, ViT, VAE
     17 │
     18 ├── layer-3/  APPLICATION    — High-level pipelines & tooling
     19 │   ├── gollek-train/          # Trainer, Callbacks, EarlyStopping, Checkpoint
     20 │   ├── gollek-nlp/            # Text pipelines, Tokenizer, Classification
     21 │   ├── gollek-vision/         # Vision models, transforms, ImageDataset
     22 │   ├── gollek-audio/          # MelSpectrogram, audio datasets
     23 │   └── gollek-multimodal/     # Vision+Audio+Video builders, cross-attention
     24 │
     25 ├── layer-4/  INFRASTRUCTURE — I/O, export, registry, API
     26 │   ├── gollek-hub/            # ModelHub, HuggingFace, SafeTensors, GGUF
     27 │   ├── gollek-export/         # ONNX/GGUF/LiteRT exporters
     28 │   ├── gollek-metrics/        # Accuracy, ClassificationMetrics, TensorBoard
     29 │   ├── gollek-augment/        # Data augmentation transforms
     30 │   └── gollek-api/            # Gollek.java facade (torch-like entry point)
     31 │
     32 └── support/
     33     ├── gollek-examples/       # MNIST, Transformer, FFN examples
     34     └── gollek-onnx/           # ONNX runtime integration (currently empty)

    ---

    Dependency Graph (Proposed)

      1 Layer 0:  tensor ← (no internal deps)
      2           autograd ← tensor
      3 
      4 Layer 1:  nn ← autograd
      5           optim ← autograd, nn
      6           data ← autograd
      7 
      8 Layer 2:  cnn ← nn, autograd
      9           rnn ← nn, autograd
     10           transformer ← nn, autograd
     11           models ← cnn, rnn, transformer, nn, hub
     12 
     13 Layer 3:  train ← optim, data, nn, metrics
     14           nlp ← nn, transformer, data
     15           vision ← nn, cnn, data
     16           audio ← nn, data
     17           multimodal ← vision, audio, nlp
     18 
     19 Layer 4:  hub ← nn, safetensors, gguf (merged)
     20           export ← nn
     21           metrics ← nn, data
     22           augment ← vision
     23           api ← autograd, nn, hub, nlp, vision, multimodal
     24 
     25 Support:  examples ← api, train, data, models
     26           onnx ← export (when populated)

    ---

    Concrete Actions Required

    Phase 1: Fix Maven Build Collisions (Critical)

    ┌───────────────────────────────────────┬───────────────────────────────────────────────────┬──────────────────────────────┐
    │ Action                                │ Modules                                           │ Detail                       │
    ├───────────────────────────────────────┼───────────────────────────────────────────────────┼──────────────────────────────┤
    │ Fix artifactId gollek-ml-models c... │ transformer, audio, safetensor, gguf, metrics,... │ Give each a unique artifa... │
    │ Add missing <dependency> declarations │ All 8 stub poms                                   │ Declare actual inter-modu... │
    │ Delete empty module                   │ gollek-ml-onnx                                    │ 0 files, remove from pare... │
    └───────────────────────────────────────┴───────────────────────────────────────────────────┴──────────────────────────────┘


    Phase 2: Unify Package Namespace

    ┌──────────────────────────────────┬─────────────────────────────────┬──────────────────┐
    │ Old Package                      │ New Package                     │ Modules Affected │
    ├──────────────────────────────────┼─────────────────────────────────┼──────────────────┤
    │ tech.kayys.gollek.lib.core       │ tech.kayys.gollek.ml.tensor     │ tensor           │
    │ tech.kayys.gollek.lib.train      │ tech.kayys.gollek.ml.train      │ train            │
    │ tech.kayys.gollek.lib.hub        │ tech.kayys.gollek.ml.hub        │ hub              │
    │ tech.kayys.gollek.lib.rnn.*      │ tech.kayys.gollek.ml.rnn.*      │ rnn              │
    │ tech.kayys.gollek.lib.vision.*   │ tech.kayys.gollek.ml.vision.*   │ vision           │
    │ tech.kayys.gollek.lib.multimodal │ tech.kayys.gollek.ml.multimodal │ multimodal       │
    │ tech.kayys.gollek.lib.augment    │ tech.kayys.gollek.ml.augment    │ augment          │
    │ tech.kayys.gollek.lib.export     │ tech.kayys.gollek.ml.export     │ export           │
    │ tech.kayys.gollek.lib.litert     │ tech.kayys.gollek.ml.litert     │ litert           │
    │ tech.kayys.gollek.lib.augment    │ tech.kayys.gollek.ml.augment    │ augment          │
    │ tech.kayys.gollek.ml.nn.optim    │ tech.kayys.gollek.ml.optim      │ optimize         │
    │ tech.kayys.gollek.ml.nn.metrics  │ tech.kayys.gollek.ml.metrics    │ nn, metrics      │
    └──────────────────────────────────┴─────────────────────────────────┴──────────────────┘


    Phase 3: Move Misplaced Files

    ┌────────────────────────────────────┬───────────────────────────────┬─────────────────────────────────────────────────────┐
    │ File                               │ Currently In                  │ Should Be In                                        │
    ├────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────────┤
    │ MelSpectrogram.java                │ gollek-ml-nn/.../ml/nn/audio/ │ gollek-ml-audio/.../ml/audio/                       │
    │ SafetensorReader/Writer.java       │ gollek-ml-nn/.../ml/nn/util/  │ gollek-ml-hub/.../ml/hub/safetensors/               │
    │ GgufReader/Writer.java + meta c... │ `gollek-ml-nn/.../ml/nn/ut... │ gollek-ml-hub/.../ml/hub/gguf/                      │
    │ TransformerBlock.java              │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ TransformerEncoder.java            │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ TransformerEncoderLayer.java       │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ TransformerDecoderLayer.java       │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ MultiHeadAttention.java            │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ FlashAttention.java                │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ PositionalEncoding.java            │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-transformer/.../ml/transformer/           │
    │ LSTM.java, GRU.java, `Bidirecti... │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-rnn/.../ml/rnn/                           │
    │ Conv2d.java, Conv1d.java (cnn m... │ tech.kayys.gollek.ml.nn pkg   │ tech.kayys.gollek.ml.cnn pkg                        │
    │ Accuracy.java (metrics)            │ `gollek-ml-nn/.../ml/nn/me... │ gollek-ml-metrics/.../ml/metrics/                   │
    │ LoRALinear.java                    │ gollek-ml-nn/.../ml/nn/       │ gollek-ml-models/.../ml/models/ (or keep in nn a... │
    └────────────────────────────────────┴───────────────────────────────┴─────────────────────────────────────────────────────┘


    Phase 4: Consolidate gollek-ml-nn (Currently 41 files → should be ~20)

    Split gollek-ml-nn into:
     - Keep in nn: NNModule, Parameter, basic layers (Linear, ReLU, GELU, ELU, LeakyReLU, SiLU, Dropout, LayerNorm, GroupNorm,
       BatchNorm*, Sequential, Flatten, ResidualBlock, Embedding)
     - Move to transformer: MultiHeadAttention, FlashAttention, PositionalEncoding, TransformerBlock, TransformerEncoder,
       TransformerEncoderLayer, TransformerDecoderLayer
     - Move to rnn: LSTM, GRU, Bidirectional
     - Move to hub: SafetensorReader/Writer, GgufReader/Writer + meta classes
     - Move to metrics: Accuracy.java
     - Move to audio: MelSpectrogram.java

    Phase 5: Merge Redundant Serialization

    gollek-ml-safetensor (2 files) + gollek-ml-gguf (5 files) → merge into gollek-ml-hub as sub-packages ml.hub.safetensors and
    ml.hub.gguf. Eliminates 2 modules with broken poms.

    ---

    Final Module Count & Structure


    ┌─────────┬────────────────────┬────────────────────┬────────────┬──────────────────────────────────┐
    │ Layer   │ Module             │ artifactId         │ Est. Files │ Package Root                     │
    ├─────────┼────────────────────┼────────────────────┼────────────┼──────────────────────────────────┤
    │ core    │ gollek-tensor      │ gollek-tensor      │ 6          │ tech.kayys.gollek.ml.tensor      │
    │ core    │ gollek-autograd    │ gollek-autograd    │ 6          │ tech.kayys.gollek.ml.autograd    │
    │ nn      │ gollek-nn          │ gollek-nn          │ 20         │ tech.kayys.gollek.ml.nn          │
    │ nn      │ gollek-optim       │ gollek-optim       │ 18         │ tech.kayys.gollek.ml.optim       │
    │ nn      │ gollek-data        │ gollek-data        │ 12         │ tech.kayys.gollek.ml.data        │
    │ arch    │ gollek-cnn         │ gollek-cnn         │ 10         │ tech.kayys.gollek.ml.cnn         │
    │ arch    │ gollek-rnn         │ gollek-rnn         │ 8          │ tech.kayys.gollek.ml.rnn         │
    │ arch    │ gollek-transformer │ gollek-transformer │ 8          │ tech.kayys.gollek.ml.transformer │
    │ arch    │ gollek-models      │ gollek-models      │ 15         │ tech.kayys.gollek.ml.models      │
    │ app     │ gollek-train       │ gollek-train       │ 10         │ tech.kayys.gollek.ml.train       │
    │ app     │ gollek-nlp         │ gollek-nlp         │ 12         │ tech.kayys.gollek.ml.nlp         │
    │ app     │ gollek-vision      │ gollek-vision      │ 5          │ tech.kayys.gollek.ml.vision      │
    │ app     │ gollek-audio       │ gollek-audio       │ 3          │ tech.kayys.gollek.ml.audio       │
    │ app     │ gollek-multimodal  │ gollek-multimodal  │ 8          │ tech.kayys.gollek.ml.multimodal  │
    │ infra   │ gollek-hub         │ gollek-hub         │ 12         │ tech.kayys.gollek.ml.hub         │
    │ infra   │ gollek-export      │ gollek-export      │ 6          │ tech.kayys.gollek.ml.export      │
    │ infra   │ gollek-metrics     │ gollek-metrics     │ 9          │ tech.kayys.gollek.ml.metrics     │
    │ infra   │ gollek-augment     │ gollek-augment     │ 2          │ tech.kayys.gollek.ml.augment     │
    │ infra   │ gollek-api         │ gollek-api         │ 2          │ tech.kayys.gollek.ml             │
    │ support │ gollek-examples    │ gollek-examples    │ 3          │ tech.kayys.gollek.ml.examples    │
    │ support │ gollek-litert      │ gollek-litert      │ 6          │ tech.kayys.gollek.ml.litert      │
    └─────────┴────────────────────┴────────────────────┴────────────┴──────────────────────────────────┘


    Total: 21 modules (down from 24, eliminated 3 broken ones via merge)

    ---