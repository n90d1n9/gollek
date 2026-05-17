This directory is the Gradle namespace root for the `:trainer:*` projects.

Current trainer modules:

- `:trainer:gollek-trainer-api`
  - canonical trainer lifecycle contracts
  - intentionally small and dependency-light
- `:trainer:gollek-trainer`
  - canonical trainer runtime facade module located at
    `trainer/gollek-trainer`
  - exposes the forward-looking package entrypoint
    `tech.kayys.gollek.trainer.Trainers`
  - uses a reflective bridge to the legacy
    `tech.kayys.gollek.ml.train.Trainer` runtime when available
  - falls back to an in-module canonical runtime
    `tech.kayys.gollek.trainer.CanonicalTrainerRuntime` when legacy runtime
    classes are not on the classpath
  - exposes typed canonical helpers:
    `Trainers.canonicalBuilder()`, `Trainers.newSession(...)`,
    `Trainers.runtimeModeType()`, and `Trainers.runtimeMode()`
  - exposes a typed compatibility builder:
    `Trainers.sessionBuilder()`
    that attempts legacy runtime only when model/optimizer/loss are provided,
    and otherwise falls back to canonical runtime
  - canonical runtime now triggers `TrainingListener.onTrainingError(...)`
    before rethrowing runtime failures from the fit loop
  - listener callback failures are treated as non-fatal and reported via
    `onTrainingError(...)`, with `listenerErrors` tracked in summary metadata
  - canonical runtime supports pluggable real loss evaluators:
    `trainBatchLoss(...)`, `validationBatchLoss(...)`, `trainEpochLoss(...)`,
    and `validationEpochLoss(...)`
  - canonical runtime supports early stopping with
    `earlyStopping(patience[, minDelta])`
  - canonical runtime supports checkpoint persistence and resume using
    `checkpointDir(...)` + `resumeFromCheckpoint(...)`
  - checkpoint resume is schema-version guarded and fails fast by default on
    incompatible checkpoint formats, with an opt-out
    `failOnCheckpointLoadError(false)` when best-effort fallback is preferred
  - typed `CanonicalTrainer` in `training/gollek-train-api` layers model-weight
    snapshots (`canonical-model.safetensors`) on top of runtime checkpoints
    so resume can restore both trainer state and model parameters
  - when no custom loss evaluator is wired, canonical runtime keeps synthetic
    fallback behavior for compatibility

Phase 2 separates trainer-facing orchestration contracts from the broader ML
convenience API. The runtime and API now both live under the canonical
`trainer/` namespace in Gradle.
