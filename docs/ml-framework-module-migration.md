# Gollek ML Framework Module Migration

This document defines the recommended naming and packaging direction for the
Java training / ML framework side of Gollek.

It is intentionally aligned with the Gollek vision:

1. Gollek's core identity is an inference / serving engine.
2. Gollek also provides a Java-first ML / DL / NN framework as an alternative
   for Java developers and type-safe ML workflows.
3. Integration surfaces such as JBang, Jupyter, and LangChain4j remain
   adapters around those cores, not replacements for them.

## Problem Summary

The current training/framework area has three naming problems:

1. Gradle module paths use mixed vocabulary:
   - `gollek-nn`
   - `gollek-train-*`
   - `gollek-nlp`
   - `gollek-serializer`
2. Maven artifactIds already point toward a cleaner `gollek-ml-*` scheme, but
   they no longer match the Gradle module paths consistently.
3. Some modules are misnamed by concern:
   - `gollek-train-transformer` is mostly preprocessing / feature engineering
   - `gollek-train-estimator` is mostly classical ML estimators
   - `gollek-train-api` mixes framework facade and trainer-facing concerns

## Naming Policy

Use these top-level groups:

- `ml/*`
  - Java ML framework primitives, model building blocks, preprocessing,
    estimators, serialization, domain packs
- `trainer/*`
  - Training orchestration, trainer runtime, experiment workflows
- `integration/*`
  - External adapters such as JBang, Jupyter, LangChain4j

Use these artifact naming rules:

- `gollek-ml-*` for framework libraries
- `gollek-trainer-*` for training workflow/runtime libraries
- `gollek-integration-*` for adapter libraries

Use these package naming rules:

- `tech.kayys.gollek.ml.*` for framework code
- `tech.kayys.gollek.trainer.*` for trainer/runtime code
- `tech.kayys.gollek.integration.*` for integrations

## Important Observation

The Maven-side training parent already hints at the target vocabulary:

- `gollek-ml-nn`
- `gollek-ml-cnn`
- `gollek-ml-rnn`
- `gollek-ml-data`
- `gollek-ml-optimize`
- `gollek-ml-api`
- `gollek-ml-nlp`

But the current Gradle module paths still use legacy names such as
`training:gollek-nn` and `training:gollek-train-api`.

So the cleanest migration path is not to invent brand new names. It is to make
Gradle module paths, artifactIds, and package roots converge on the existing
`gollek-ml-*` direction.

## Canonical Rename Matrix

### Framework Core

| Current Gradle module | Current artifact direction | Recommended Gradle module | Recommended artifactId | Recommended package root | Notes |
|---|---|---|---|---|---|
| `training:gollek-nn` | `gollek-ml-nn` | `ml:gollek-ml-nn-core` | `gollek-ml-nn-core` | `tech.kayys.gollek.ml.nn` | Core NN primitives only |
| `training:gollek-nn-optimize` | `gollek-ml-optimize` | `ml:gollek-ml-optim` | `gollek-ml-optim` | `tech.kayys.gollek.ml.optim` | Optimizers / schedulers |
| `training:gollek-train-data` | `gollek-ml-data` | `ml:gollek-ml-data` | `gollek-ml-data` | `tech.kayys.gollek.ml.data` | Data loading, datasets, transforms |
| `training:gollek-serializer` | none consistent | `ml:gollek-ml-serialization` | `gollek-ml-serialization` | `tech.kayys.gollek.ml.serialization` | NumPy / pickle / bridge layer |

### Architecture Families

| Current Gradle module | Current artifact direction | Recommended Gradle module | Recommended artifactId | Recommended package root | Notes |
|---|---|---|---|---|---|
| `training:gollek-nn-rnn` | `gollek-ml-rnn` | `ml:gollek-ml-rnn` | `gollek-ml-rnn` | `tech.kayys.gollek.ml.rnn` | Recurrent models |
| `training:gollek-nn-cnn` | `gollek-ml-cnn` | `ml:gollek-ml-cnn` | `gollek-ml-cnn` | `tech.kayys.gollek.ml.cnn` | CNN family |
| `training:gollek-train-transformer` | none consistent | `ml:gollek-ml-preprocessing` | `gollek-ml-preprocessing` | `tech.kayys.gollek.ml.preprocessing` | Current code is preprocessing / feature engineering, not transformer runtime |
| split from `training:gollek-nn` | mixed in `gollek-ml-nn` | `ml:gollek-ml-transformer` | `gollek-ml-transformer` | `tech.kayys.gollek.ml.transformer` | Move transformer classes out of `gollek-nn` |

### Domain Packs

| Current Gradle module | Current artifact direction | Recommended Gradle module | Recommended artifactId | Recommended package root | Notes |
|---|---|---|---|---|---|
| `training:gollek-nlp` | `gollek-ml-nlp` | `ml:gollek-ml-text` | `gollek-ml-text` | `tech.kayys.gollek.ml.text` | Better matches the domain role; keep `gollek-ml-nlp` only if you want classic NLP branding |
| `training:gollek-nn-vision` | `gollek-ml-vision` | `ml:gollek-ml-vision` | `gollek-ml-vision` | `tech.kayys.gollek.ml.vision` | Vision task layer |
| `training:gollek-nn-audio` | `gollek-audio` | `ml:gollek-ml-audio` | `gollek-ml-audio` | `tech.kayys.gollek.ml.audio` | Align audio naming with the rest |
| `training:gollek-nn-multimodal` | `gollek-ml-multimodal` | `ml:gollek-ml-multimodal` | `gollek-ml-multimodal` | `tech.kayys.gollek.ml.multimodal` | Multimodal task layer |

### Classical ML and Pipeline Layer

| Current Gradle module | Current artifact direction | Recommended Gradle module | Recommended artifactId | Recommended package root | Notes |
|---|---|---|---|---|---|
| `training:gollek-train-estimator` | none consistent | `ml:gollek-ml-estimator` | `gollek-ml-estimator` | `tech.kayys.gollek.ml.estimator` | Classical ML estimators |
| split from `training:gollek-train-transformer` | none consistent | `ml:gollek-ml-pipeline` | `gollek-ml-pipeline` | `tech.kayys.gollek.ml.pipeline` | ColumnTransformer, StandardScaler, PCA, etc. |
| split from `training:gollek-train-transformer` | none consistent | `ml:gollek-ml-feature` | `gollek-ml-feature` | `tech.kayys.gollek.ml.feature` | Feature engineering and selection |

### Training Runtime / User Entry Points

| Current Gradle module | Current artifact direction | Recommended Gradle module | Recommended artifactId | Recommended package root | Notes |
|---|---|---|---|---|---|
| `training:gollek-train-api` | `gollek-ml-api` | `ml:gollek-ml-api` and `trainer:gollek-trainer-api` | `gollek-ml-api`, `gollek-trainer-api` | `tech.kayys.gollek.ml`, `tech.kayys.gollek.trainer.api` | Split framework facade from trainer runtime contracts |
| `training:gollek-train` | mixed | `trainer:gollek-trainer` | `gollek-trainer` | `tech.kayys.gollek.trainer` | Training orchestration runtime |
| `training:gollek-train-examples` | `gollek-examples` | `examples:gollek-ml-examples` | `gollek-ml-examples` | `tech.kayys.gollek.examples.ml` | Examples should not be a core framework module |

## Package Cleanup Recommendations

The current code mixes several roots:

- `tech.kayys.gollek.ml.nn.*`
- `tech.kayys.gollek.ml.optim.*`
- `tech.kayys.gollek.ml.transformer.*`
- `tech.kayys.gollek.nlp.*`
- `tech.kayys.gollek.train.*`
- `tech.kayys.gollek.ml.serialize.*`

Target package cleanup:

- `tech.kayys.gollek.nlp.*` -> `tech.kayys.gollek.ml.text.*`
- `tech.kayys.gollek.train.runner.*` -> `tech.kayys.gollek.ml.runner.*`
- `tech.kayys.gollek.train.estimator.*` -> `tech.kayys.gollek.ml.estimator.*`
- `tech.kayys.gollek.train.transformer.*` -> `tech.kayys.gollek.ml.preprocessing.*` or `tech.kayys.gollek.ml.pipeline.*`
- `tech.kayys.gollek.ml.serialize.*` -> `tech.kayys.gollek.ml.serialization.*`

## Recommended Directory Target

Long term:

```text
gollek/
  ml/
    gollek-ml-nn-core
    gollek-ml-optim
    gollek-ml-transformer
    gollek-ml-rnn
    gollek-ml-cnn
    gollek-ml-data
    gollek-ml-pipeline
    gollek-ml-feature
    gollek-ml-estimator
    gollek-ml-serialization
    gollek-ml-text
    gollek-ml-vision
    gollek-ml-audio
    gollek-ml-multimodal
    gollek-ml-api
  trainer/
    gollek-trainer-api
    gollek-trainer
  examples/
    gollek-ml-examples
  integration/
    gollek-integration-jbang
    gollek-integration-jupyter
    gollek-integration-langchain4j
```

## Migration Order

1. Align Gradle module paths with existing `gollek-ml-*` artifact vocabulary.
2. Introduce a dedicated `trainer:gollek-trainer-api` module so trainer
   lifecycle contracts no longer have to live inside the broad ML convenience
   API.
3. Split `training:gollek-nn` into:
   - `gollek-ml-nn-core`
   - `gollek-ml-optim`
   - `gollek-ml-transformer`
4. Rename `training:gollek-train-transformer` to preprocessing / pipeline names.
5. Rename `training:gollek-train-estimator` to `gollek-ml-estimator`.
6. Split `training:gollek-train-api` into:
   - `gollek-ml-api`
   - `gollek-trainer-api`
7. Move package roots to `tech.kayys.gollek.ml.*` and
   `tech.kayys.gollek.trainer.*`.
8. After the split is stable, move the top-level folder group from
   `training/` to `ml/` plus `trainer/`.

## Progress Note

The Gradle migration now includes a dedicated
`trainer:gollek-trainer-api` module for canonical trainer lifecycle contracts:

- `TrainerSession`
- `TrainerConfig`
- `TrainingListener`
- `TrainingSummary`

That gives the trainer/runtime side a stable home even before the larger
package move away from `tech.kayys.gollek.train.*` is complete.

The unified runner surface has also been realigned to `tech.kayys.gollek.ml.runner.*`
so the public ML-framework API matches the examples and the ONNX runner
integration that already expected that namespace.

The `trainer:gollek-trainer` module is now rooted under the canonical
`trainer/gollek-trainer` path and is currently treated as a namespace bridge
rather than a full runtime port. Its compiled surface is limited to
`tech.kayys.gollek.trainer.*`, with a reflective entrypoint back to the legacy
`tech.kayys.gollek.ml.train.Trainer` implementation while the autograd/NN
foundation is still being migrated into Gradle-managed modules.

To keep that runner surface maintainable, it is now split behind a dedicated
Gradle module path:

- `:ml:gollek-ml-runner-api`

That lets backend adapters such as ONNX compile against a focused contract
module without dragging the entire `gollek-ml-api` umbrella and its older NN
dependency graph into every adapter build.

## Compatibility Strategy

During migration:

- keep old Maven artifactIds as compatibility aliases only when needed
- use Gradle project path aliases temporarily if build logic is still being
  migrated
- move imports package-by-package instead of all at once
- do not migrate inference / serving modules into the `ml/*` tree

## Public Docs and Website Sync

The public naming must stay aligned across:

- repository module paths
- Maven / Gradle artifact names
- package roots
- documentation
- website / GitHub Pages content

If the website is maintained in a separate repo or sibling path such as
`website/gollek-ai.github.io`, treat it as part of the same migration wave.

Recommended public wording:

- "Gollek Inference Engine" for serving/runtime materials
- "Gollek ML Framework" for the Java training / ML / DL / NN side
- "Gollek Integrations" for JBang, Jupyter, LangChain4j, and related adapters

That keeps the public story consistent with the internal module split:

- runtime / runner / sdk / spi
- ml
- trainer
- integration

## Recommendation Summary

The clean target is:

- inference / serving: keep under runtime, runner, sdk, spi
- Java ML framework: move to `ml/*` with `gollek-ml-*`
- training orchestration: move to `trainer/*` with `gollek-trainer-*`
- integrations: keep under `integration/*`

This naming keeps Gollek's primary identity clear while making the Java ML
framework side coherent and maintainable.
