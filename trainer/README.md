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
    missing runtime checkpoints or incompatible checkpoint formats, with an opt-out
    `failOnCheckpointLoadError(false)` when best-effort fallback is preferred
  - typed `CanonicalTrainer` in `training/gollek-train-api` layers model-weight
    snapshots (`canonical-model.safetensors`) on top of runtime checkpoints
    so resume can restore both trainer state and model parameters
  - typed model snapshots now include `canonical-model.metadata`; strict resume
    validates model class, parameter signatures, byte size, and SHA-256 before
    loading weights, while lenient mode reports
    `checkpointResumeCompatibilityMismatches`
  - typed checkpoints also write `canonical-checkpoints.metadata`, a manifest
    with byte-size/SHA-256 integrity entries for runtime, optimizer, scheduler,
    GradScaler, history, report, and model artifacts before resume trusts those
    files; best-model restore-at-end is guarded by the same manifest checks
  - typed batches validate non-null, non-empty, sample-aligned, finite inputs
    and labels before forward/loss execution, so malformed or NaN/Infinity
    dataset values fail before optimizer mutation and surface through
    structured summary metadata
  - typed custom losses must return a single-value tensor; vector/matrix losses
    fail before backward so trainer summaries and gradients cannot disagree
  - typed trainer paths validate prediction tensors before loss/metric
    evaluation, surfacing exploding activations as `nonFiniteKind=prediction`
  - typed trainer metric snapshots validate finite metric values after real
    train/validation batches have run, surfacing duplicate names, broken or
    throwing custom metrics, and invalid `DetailedMetric` payloads through
    `invalidMetric*` metadata while preserving empty-phase compatibility
  - typed history CSV now keeps metric maps and `DetailedMetric` diagnostics as
    deterministic JSON cells and restores them during resume, making the
    checkpoint history usable by dashboards and spreadsheet workflows; malformed
    structured cells are reported as history load errors and rejected by strict
    resume, and ambiguous CSV shapes such as duplicate headers or extra row
    cells are rejected before values can be overwritten or ignored; epoch is a
    required unique non-negative integer key
  - when no custom loss evaluator is wired, canonical runtime keeps synthetic
    fallback behavior for compatibility

Phase 2 separates trainer-facing orchestration contracts from the broader ML
convenience API. The runtime and API now both live under the canonical
`trainer/` namespace in Gradle.
