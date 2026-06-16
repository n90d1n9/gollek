# Gollek ML Framework Phase 1 Plan

This document turns the naming strategy into a concrete first migration wave.

Phase 1 is intentionally conservative:

- no package-root rewrite yet for the bulk of code
- no directory move on disk yet
- no attempt to fix every stale training dependency in one sweep
- focus on converging Gradle project paths, artifactIds, and public naming

The goal is to make later Gradle-only migration easier without causing a giant
one-shot refactor.

## Status Update (May 13, 2026)

- Local installer default path is now Gradle-only via
  `./scripts/install-local-macos.sh`.
- Maven command fallback has been removed from
  `scripts/install-local-runtime.sh`.
- Optional compatibility publishing still exists behind
  `GOLLEK_PUBLISH_RUNTIME_LOCAL=true`, but it is no longer part of the default
  install flow.

## Phase 1 Objectives

1. Make Gradle project paths reflect the `gollek-ml-*` direction already used
   by the Maven-side artifacts.
2. Remove the confusing split between:
   - `training:gollek-nn`
   - `training:gollek-train-*`
   - `gollek-ml-*` artifactIds
3. Keep directory moves and package moves for later phases.
4. Keep inference/serving identity separate from the ML framework identity.

## Phase 1 Non-Goals

Do not do these in Phase 1:

- move `training/` sources into new folders on disk
- fully split `gollek-nn` internals into new modules yet
- rewrite all package names yet
- force immediate package-root rewrites across all training code yet

## Source of Truth

For framework naming, prefer the existing Maven-side artifact direction as the
canonical target where it already exists:

- `gollek-ml-nn`
- `gollek-ml-optimize`
- `gollek-ml-rnn`
- `gollek-ml-cnn`
- `gollek-ml-vision`
- `gollek-ml-nlp`
- `gollek-ml-data`
- `gollek-ml-api`

Those names are already closer to the intended framework identity than the
current Gradle project paths.

## Recommended Phase 1 Gradle Project Renames

Keep the physical directories under `training/` for now, but rename the Gradle
project paths in `settings.gradle.kts`.

### Core and Domain Modules

| Current project path | Phase 1 project path | Keep directory for now |
|---|---|---|
| `training:gollek-nn` | `ml:gollek-ml-nn` | `training/gollek-nn` |
| `training:gollek-nn-optimize` | `ml:gollek-ml-optimize` | `training/gollek-nn-optimize` |
| `training:gollek-nn-rnn` | `ml:gollek-ml-rnn` | `training/gollek-nn-rnn` |
| `training:gollek-nn-cnn` | `ml:gollek-ml-cnn` | `training/gollek-nn-cnn` |
| `training:gollek-nn-vision` | `ml:gollek-ml-vision` | `training/gollek-nn-vision` |
| `training:gollek-nn-audio` | `ml:gollek-ml-audio` | `training/gollek-nn-audio` |
| `training:gollek-nn-multimodal` | `ml:gollek-ml-multimodal` | `training/gollek-nn-multimodal` |
| `training:gollek-nlp` | `ml:gollek-ml-nlp` | `training/gollek-nlp` |
| `training:gollek-train-data` | `ml:gollek-ml-data` | `training/gollek-train-data` |
| `training:gollek-serializer` | `ml:gollek-ml-serialization` | `training/gollek-serializer` |

### Higher-Level Workflow Modules

| Current project path | Phase 1 project path | Keep directory for now |
|---|---|---|
| `training:gollek-train-api` | `ml:gollek-ml-api` | `training/gollek-train-api` |
| `training:gollek-train-estimator` | `ml:gollek-ml-estimator` | `training/gollek-train-estimator` |
| `training:gollek-train-transformer` | `ml:gollek-ml-preprocessing` | `training/gollek-train-transformer` |
| `training:gollek-train` | `trainer:gollek-trainer` | `trainer/gollek-trainer` |
| `training:gollek-train-examples` | `examples:gollek-ml-examples` | `training/gollek-train-examples` |

## Why These Are the Right First Renames

### `gollek-train-transformer` should not keep that name

Its code is mostly:

- `tech.kayys.gollek.ml.pipeline`
- `tech.kayys.gollek.ml.feature`
- `tech.kayys.gollek.ml.feature_selection`
- `tech.kayys.gollek.ml.base`

So the module behaves more like preprocessing / feature engineering than a
transformer-runtime module.

### `gollek-train-estimator` is really classical ML

Its code contains:

- tree-based models
- ensemble models
- naive Bayes
- SVM
- clustering
- base estimators

So `gollek-ml-estimator` is a better first-class name.

### `gollek-train-api` mixes two meanings

It currently acts like:

- framework umbrella API
- training/runtime-facing API

Phase 1 keeps it as one module but renames it to `ml:gollek-ml-api`.
Later, it should split into:

- `ml:gollek-ml-api`
- `trainer:gollek-trainer-api`

## Phase 1 ArtifactId Alignment

Where artifactIds are already good, keep them.

Where they are inconsistent, align them:

| Module directory | Current artifactId | Recommended Phase 1 artifactId |
|---|---|---|
| `training/gollek-nn-audio` | `gollek-audio` | `gollek-ml-audio` |
| `training/gollek-train-examples` | `gollek-examples` | `gollek-ml-examples` |

The rest should stay aligned with their existing `gollek-ml-*` form for now.

## Phase 1 Package Policy

Do not mass-rename all packages yet.

Phase 1 package policy:

- keep `tech.kayys.gollek.ml.nn.*`
- keep `tech.kayys.gollek.ml.optim.*`
- keep `tech.kayys.gollek.ml.transformer.*`
- keep `tech.kayys.gollek.ml.serialize.*` temporarily
- keep `tech.kayys.gollek.nlp.*` temporarily
- keep `tech.kayys.gollek.train.*` temporarily

Only add TODO markers in the plan:

- `tech.kayys.gollek.nlp.*` -> future `tech.kayys.gollek.ml.text.*`
- `tech.kayys.gollek.train.*` -> future `tech.kayys.gollek.trainer.*`
- `tech.kayys.gollek.ml.serialize.*` -> future `tech.kayys.gollek.ml.serialization.*`

## Files That Must Change in Phase 1

### 1. Gradle settings

- `settings.gradle.kts`

Replace the `include("training:...")` paths with the new `ml:` / `trainer:` /
`examples:` paths and explicitly map them to the existing directories using
`project(...).projectDir = file(...)`.

### 2. Gradle project references

The following current references are known and need rewriting to the new
project paths:

- `examples/jupyter-deps/build.gradle.kts`
- `models/gollek-model-common/build.gradle.kts`
- `runner/pickle/gollek-ml-pickle/build.gradle.kts`
- `runner/onnx/gollek-ml-export-onnx/build.gradle.kts`
- `runner/onnx/gollek-ml-onnx/build.gradle.kts`
- `integration/gollek-jupyter-kernel/build.gradle.kts`
- `integration/langchain4j/gollek-langchain4j/build.gradle.kts`
- all affected `training/*/build.gradle.kts`

### 3. Artifact alignment

Update inconsistent `artifactId` values in:

- `training/gollek-nn-audio/pom.xml`
- `training/gollek-train-examples/pom.xml`

and their Gradle publications when we make Gradle the canonical publisher.

## Example `settings.gradle.kts` Shape

Illustrative example only:

```kotlin
include("ml:gollek-ml-nn")
project(":ml:gollek-ml-nn").projectDir = file("training/gollek-nn")

include("ml:gollek-ml-optimize")
project(":ml:gollek-ml-optimize").projectDir = file("training/gollek-nn-optimize")

include("ml:gollek-ml-data")
project(":ml:gollek-ml-data").projectDir = file("training/gollek-train-data")

include("trainer:gollek-trainer")
project(":trainer:gollek-trainer").projectDir = file("trainer/gollek-trainer")
```

## Phase 1 Dependency Rewrite Examples

Before:

```kotlin
implementation(project(":training:gollek-nn"))
implementation(project(":training:gollek-train-data"))
implementation(project(":training:gollek-train-api"))
```

After:

```kotlin
implementation(project(":ml:gollek-ml-nn"))
implementation(project(":ml:gollek-ml-data"))
implementation(project(":ml:gollek-ml-api"))
```

## Why Phase 1 Helps Gradle Migration

Right now the build graph is confusing because:

- project paths are legacy
- artifactIds are partly future-facing
- package roots are mixed

Phase 1 does not solve all code issues, but it removes one whole layer of
confusion:

- Gradle project path
- artifact name
- public docs label

will start converging.

That makes Phase 2 much safer.

## Phase 2 Preview

After Phase 1 is stable:

1. split `gollek-nn` into:
   - `gollek-ml-nn-core`
   - `gollek-ml-optim`
   - `gollek-ml-transformer`
2. split `gollek-train-api` into:
   - `gollek-ml-api`
   - `gollek-trainer-api`
3. move package roots:
   - `gollek.nlp` -> `gollek.ml.text`
   - `gollek.train` -> `gollek.trainer`
4. move directories from `training/` into:
   - `ml/`
   - `trainer/`

## Recommended Immediate Action

The first safe code migration wave should be:

1. rename Gradle project paths in `settings.gradle.kts`
2. rewrite Gradle `project(...)` dependencies to those new names
3. align the two inconsistent artifactIds
4. leave packages and directories unchanged for now

That gives Gollek a cleaner identity for the Java ML framework side without
creating a giant all-at-once refactor.
