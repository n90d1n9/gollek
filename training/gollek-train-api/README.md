# Gollek SDK :: ML Framework Façade

The `gollek-ml-ml` module is the top-level aggregator for the Gollek ML framework. It provides the central `GollekSdk` façade, exposing user-friendly entry points for common ML tasks without requiring the developer to instantiate complex internal pipelines manually.

## Features

- **Model Builder API**: Simplifies the loading of models, tokenizers, and weights via `GollekSdk.builder()`.
- **Pre-configured Pipelines**: Provides native access to `TextGenerationPipeline`, `EmbeddingsPipeline`, etc.
- **Unified Interface**: Masks the complexity of `gollek-ml-autograd`, `gollek-ml-nn`, and the execution kernels.
- **Canonical Trainer Bridge**: `Gollek.DL.trainer()` now provides a typed path into
  `:trainer:gollek-trainer` with real forward/loss/backward execution.
- **One-call Training Presets**: `Gollek.DL.fit(...)` can run preset training
  modes (MSE/Huber/CrossEntropy + AdamW/SGD) without manual runtime wiring.
- **Public Training Metrics**: `Gollek.DL.trainingOptions()` can attach
  canonical metrics such as MAE, MSE, RMSE, R2, accuracy, macro precision,
  macro recall, macro F1, and top-k accuracy to the one-call `fit(...)` path, with
  train/validation values returned in `TrainingSummary.metadata()`.
- **Trainer Metric Contracts**: trainer metrics now have standalone
  `TrainingMetric` and `DetailedTrainingMetric` contracts plus the
  `TrainingMetrics` factory catalog. Built-in metric implementations live
  outside `CanonicalTrainer` and are split by regression, binary,
  multiclass, and multilabel families, so custom metrics and built-ins can be
  used without coupling code to the trainer monolith. The legacy
  `CanonicalTrainer.Metric`, `.DetailedMetric`, and `.Metrics` names remain as
  source-compatible aliases.
- **Robust Regression Preset**:
  `TrainingPreset.REGRESSION_HUBER_*` uses `Gollek.DL.huberLoss(...)` for
  outlier-tolerant regression with a real autograd backward path.
- **Regression Metric Preset**:
  `Gollek.DL.trainingOptions().regressionMetrics()` records MAE, MSE, RMSE,
  and R2 for regression outputs shaped like their targets.
- **Public LR Scheduling**: `Gollek.DL.trainingOptions().stepLrBatches(...)`,
  `.stepLrEpochs(...)`, `.cosineAnnealingLrBatches(...)`, and
  `.cosineAnnealingLrEpochs(...)` attach scheduler policies to the one-call
  `fit(...)` path while reusing canonical scheduler metadata and checkpoint
  state.
- **Warmup Cosine Scheduling**:
  `Gollek.DL.trainingOptions().warmupCosineLrBatches(...)` and
  `.warmupCosineLrEpochs(...)` provide Transformer-style linear warmup followed
  by cosine decay. The scheduler starts new runs at LR `0.0`, supports
  checkpoint resume, and reports `learningRateSchedulerState.*` metadata.
- **Reduce LR On Plateau Scheduling**:
  `Gollek.DL.trainingOptions().reduceLrOnPlateauValidationLoss(...)` reduces
  LR after validation loss stalls, while `.reduceLrOnPlateauMetric(...)` can
  watch a named validation metric such as F1 or AUROC. Plateau scheduler state
  is checkpointed and flattened into summary/report metadata.
- **Built-in Early Stopping**: Canonical trainer and `Gollek.DL.fit(...)` now
  support patience/min-delta controls for validation-driven stopping. The
  default monitor is validation loss, and
  `.earlyStoppingMonitorMetric("f1", BestModelMonitorMode.MAX)` can stop by a
  named validation metric without running an extra validation pass.
- **Per-Epoch History**: `TrainingSummary.metadata().get("epochHistory")`
  exposes a row per completed epoch with train/validation losses, flattened
  metric values, nested metric maps, learning rate, optimizer steps, and
  scheduler steps for post-run debugging and dashboard rendering.
- **Gradient Diagnostics**: Each summary and history row includes real gradient
  and parameter norms (`latestGradientL2NormBeforeClip`,
  `latestGradientL2Norm`, `latestGradientClipped`,
  `latestParameterL2Norm`) so exploding, clipped, or missing gradients are
  visible without custom callbacks.
- **Non-Finite Training Guard**: The canonical trainer rejects NaN/Inf losses
  before backward and NaN/Inf gradients before `optimizer.step()`, clears the
  pending gradients, skips failed model/optimizer checkpoints, and records
  `nonFinite*` metadata plus a `stopReason` such as
  `non-finite-train-gradient`.
- **Throughput Diagnostics**: Summary and history metadata include batch,
  sample, tensor-element, compute-millisecond, and samples/second counters for
  train and validation phases, making performance regressions visible in normal
  SDK runs.
- **Durable History CSV**: When `checkpointDir(...)` is configured, the trainer
  writes `canonical-history.csv` beside the model/optimizer/scheduler
  checkpoints so learning curves survive the process and can be opened in
  spreadsheets or dashboard tooling. Resumed runs load existing rows first and
  append later epochs instead of replacing earlier history. Nested metric maps
  and detailed diagnostics are stored as deterministic JSON cells, not Java
  `Map.toString()` output, so tooling can parse confusion matrices and similar
  payloads reliably across resume. Malformed structured history cells are
  reported as `trainingHistoryLoadError`; strict resume fails closed instead of
  trusting ambiguous history. Duplicate CSV headers, blank headers, and rows
  with extra cells are also rejected so resume cannot overwrite or ignore
  history values silently. The `epoch` column is required and must contain
  unique non-negative integers, making it the stable key for resumed history
  rows.
- **Structured Training Report**: The same checkpoint directory now receives
  `canonical-report.json`, a dependency-free machine-readable report containing
  the final summary, epoch history, metrics, checkpoints, accelerator metadata,
  gradient diagnostics, and throughput counters for CLIs, dashboards, and JBang
  examples that should not parse console output.
- **Checkpoint Resume**: `Gollek.DL.trainer()` / `CanonicalTrainer.Builder`
  can resume from checkpoint state via `checkpointDir(...).resumeFromCheckpoint()`.
- **Optimizer State Resume**: Checkpoints preserve optimizer-internal state for
  SGD, Adam, AdamW, and RMSprop, including momentum/velocity buffers and Adam
  moments, so resumed training follows the same numerical path as an
  uninterrupted run.
- **Partial Resume Diagnostics**: trainer summaries now distinguish checkpoint
  files present after training from artifacts that were missing at resume time,
  covering model, optimizer, scheduler, GradScaler, history, and report outputs.
  Strict resume fails closed when runtime state or required
  model/optimizer/scheduler/scaler artifacts are missing;
  `failOnCheckpointLoadError(false)` keeps best-effort fallback and records
  `checkpointResumePartial` plus missing artifact names.
- **Checkpoint Compatibility Metadata**: model checkpoints now write
  `canonical-model.metadata` beside `canonical-model.safetensors`. Resume
  validates the model class, parameter count, and named-parameter signature when
  metadata is present. The same metadata records checkpoint byte size and
  SHA-256, so corrupted or truncated model files fail fast by default and report
  `checkpointResumeCompatibilityMismatches` in lenient mode.
- **Checkpoint Manifest Integrity**: checkpoint directories now include
  `canonical-checkpoints.metadata` with byte-size and SHA-256 entries for
  runtime, optimizer, scheduler, GradScaler, history, report, and model
  artifacts. Resume validates runtime-state files before deserializing them, so
  tampered runtime, optimizer, or scheduler checkpoints fail closed by default
  and stay visible in lenient `checkpointResumeCompatibilityMismatches`
  diagnostics. Best-model restore-at-end uses the same manifest guard,
  preventing corrupted `canonical-best-model.safetensors` from silently
  replacing current weights.
- **Batch Data Guard**: train and validation batches now validate non-null,
  non-empty, sample-aligned input/label tensors before forward/loss execution.
  NaN or Infinity in dataset values fails fast, records
  `nonFinitePhase`/`nonFiniteKind`, skips optimizer mutation for train batches,
  and still writes the structured failure report; malformed batches surface
  `invalidBatchReason` before model code runs.
- **Loss Shape Guard**: custom trainer losses must reduce to exactly one tensor
  value before backward. Vector or matrix losses now fail before gradients are
  propagated, report `invalidLossShape`, and avoid writing model/optimizer
  checkpoints for failed train steps.
- **Prediction Finite Guard**: model outputs are validated before loss and
  metric evaluation. NaN or Infinity predictions now surface as
  `nonFiniteKind=prediction`, preventing exploding activations from reaching
  custom losses or validation metrics.
- **Metric Finite Guard**: metrics are checked at train/validation reporting
  boundaries after real batches have been observed. Duplicate metric names,
  NaN or Infinity custom metrics, metric-value exceptions, and invalid
  `DetailedMetric` payloads stop the run, mark the summary with
  `invalidMetric*` metadata, and skip final model/optimizer checkpoint writes;
  intentionally empty phases may still report undefined metrics without failing
  compatibility flows.
- **Optimizer Safety Guards**: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta,
  Lion, LAMB, and `GradientClipper` now validate finite hyperparameters,
  parameters, and gradients before mutating state; Adam L2 decay no longer
  mutates gradient buffers, and Adagrad weight decay is applied to every tensor
  element.
- **Scheduler Resume Guards**: StepLR, CosineAnnealingLR,
  WarmupCosineScheduler, and ReduceLROnPlateau validate finite configuration
  and checkpoint state before restoring resumed learning rates, rejecting NaN
  LRs, negative counters, and infinite plateau metrics.
- **Real Post-Training Quantization Inputs**: `QuantizationEngine` now consumes
  explicit in-memory weight tensors or a simple text weight file, rejects
  missing/unsupported sources instead of inventing placeholder weights, and
  writes quantized code/dequantized artifacts with accuracy/compression metrics.
- **Real Mixed-Precision Loss Scaling**: `GradScaler.scale(loss)` now returns a
  differentiable scaled loss, unscales `Optimizer`/`Parameter` gradients,
  skips optimizer steps on NaN/Inf gradients, and grows/backs off the scale
  according to observed overflow. Scaler state is checkpointable, validates
  incompatible restore payloads, and avoids partially unscaling gradients when
  overflow is detected.
- **Scaler-Backed Trainer Mixed Precision**:
  `CanonicalTrainer.Builder.mixedPrecision(true)`, `.gradScaler(...)`, and
  `Gollek.DL.trainingOptions().mixedPrecision()` now execute the real
  GradScaler path during training. Public training options also accept
  `trainingOptions().gradScaler(customScaler)` /
  `.mixedPrecision(customScaler)` for tuned initial scale and growth policy.
  Overflowed scaled gradients skip optimizer and scheduler steps, back off the
  loss scale, and save/resume `canonical-grad-scaler.state` beside
  model/optimizer/scheduler checkpoints. Resume rejects incompatible scaler
  policies by default; set lenient checkpoint loading only when you explicitly
  want to continue with a freshly configured scaler and inspect the metadata
  load error plus `gradScalerCheckpointFallbackUsed`.
- **Composed Autograd Losses**: `GradTensor` now backpropagates through common
  Java-side training expressions such as view ops, `sum`/`mean`, `pow`,
  unary math/activations, and last-dimension `softmax`, so custom losses no
  longer have to install manual backward hooks for basic compositions.
- **Autograd Graph Builders**: `Gollek.cat(...)`, `Gollek.stack(...)`, and
  `Gollek.where(...)` now preserve gradients through concatenation, stacking,
  and broadcasted selection paths instead of returning detached tensors.
- **Transformer Attention Autograd**: Attention compatibility einsums
  (`bhid,bhjd->bhij` and `bhij,bhjd->bhid`) now backpropagate to query/key
  and attention/value tensors, so Java transformer blocks can train through
  score and context projections instead of detaching at attention.
- **Public Multi-Head Attention Gradients**: `MultiHeadAttention` now keeps
  gradients connected through head split/merge reshapes and causal mask
  application; `TransformerBlock` dropout also uses differentiable masking
  instead of recreating detached tensors.
- **FlashAttention Training Correctness**: `FlashAttention` now switches to a
  differentiable exact attention path whenever gradients are tracked, while its
  no-grad tiled inference path keeps the memory-efficient loop and correctly
  merges `[B, H, T, D]` back to `[B, T, dModel]`.
- **Attention Dropout Wiring**: `MultiHeadAttention(..., dropoutP)` now applies
  dropout to attention probabilities after softmax, and encoder/decoder layers
  pass their configured dropout probability into self- and cross-attention.
- **TransformerBlock GELU FFN**: `TransformerBlock` now uses a registered
  `GELU` activation in its feed-forward network, matching the documented
  transformer block behavior and preserving smooth negative-side gradients.
- **Rotary Embedding Autograd**: `RotaryEmbedding.apply(...)` now preserves
  gradients through RoPE by backpropagating the inverse rotation and rejects
  invalid ranks, head dimensions, or sequence lengths before cache indexing.
- **Root GroupNorm Autograd**: `tech.kayys.gollek.ml.nn.GroupNorm` now uses
  differentiable reshape/mean/variance tensor ops instead of returning a
  detached tensor, while validating group count, epsilon, rank, and channels.
- **Layer Configuration Guards**: `Dropout` now rejects non-finite probabilities
  and uses deterministic zero-output/zero-gradient behavior at `p=1`, while
  `tech.kayys.gollek.ml.nn.layer.GroupNorm` validates group counts, epsilon,
  rank, and channel count before training.
- **Weighted CrossEntropy Mean**: `CrossEntropyLoss(classWeights)` now divides
  weighted loss and gradients by the sum of selected sample weights rather than
  raw batch size, matching standard weighted-mean training semantics.
- **Class Target Validation**: CrossEntropy, Focal, LabelSmoothing, ArcFace,
  and CTC losses reject wrongly-shaped, non-finite, fractional, or out-of-range
  class labels before indexing logits or sequence probabilities.
- **Focal Loss Guards**: `FocalLoss` and `BinaryFocalWithLogitsLoss` now
  reject empty/non-finite logits before focal weighting, and binary focal
  gradients are covered for positive-weighted binary and multi-label training.
- **Regression Loss Guards**: `MSELoss`, `L1Loss`, `SmoothL1Loss`, and
  `HuberLoss` now reject empty tensors and NaN/Inf predictions or targets
  before computing losses; `SmoothL1Loss` uses standard beta-scaled loss and
  gradient semantics.
- **Label Smoothing Autograd**: `LabelSmoothingLoss` now preserves gradients
  with the standard `softmax - smoothedTarget` backward path, making it usable
  in real trainer loops instead of returning a detached scalar.
- **Stable BCE With Logits**: `BCEWithLogitsLoss` now uses a stable softplus
  formulation, so confident wrong binary predictions retain their true large
  loss instead of being capped by `log(sigmoid + epsilon)`, and non-finite
  logits or invalid targets are rejected before training.
- **Dice Loss Autograd**: `DiceLoss` now backpropagates the overlap objective
  for segmentation training and validates probability predictions, binary
  masks, matching shapes, and smoothing values before computing the ratio.
- **IoU Loss Autograd**: `IoULoss` now backpropagates through predicted
  bounding boxes and validates `[batch, 4]` finite, ordered coordinates before
  computing object-detection overlap loss.
- **Triplet Loss Autograd**: `TripletLoss` now backpropagates active margin
  violations to anchor, positive, and negative embeddings and validates
  matching `[batch, dim]` embedding tensors.
- **Contrastive Loss Autograd**: `ContrastiveLoss` now backpropagates positive
  pairs and active negative margin violations while validating binary labels
  and matching `[batch, dim]` embedding tensors.
- **Cosine Embedding Loss Autograd**: `CosineEmbeddingLoss` now
  backpropagates positive and active negative cosine pairs while validating
  strict `1.0`/`-1.0` labels, margin bounds, and matching `[batch, dim]`
  embedding tensors.
- **CTC Loss Autograd**: `CTCLoss` now backpropagates log-probability inputs
  using log-space forward-backward posteriors and rejects blank/impossible
  target alignments before training.
- **ArcFace Loss Autograd**: `ArcFaceLoss` now backpropagates through feature
  embeddings and learned class centers, including normalization and angular
  margin derivatives for metric-learning training.
- **Knowledge Distillation Gradients**: The distillation trainer now uses
  teacher-to-student KL with a stable closed-form backward path and standard
  CrossEntropy for hard labels, so student parameters receive gradients from
  both branches while the teacher remains detached.
- **CPU Backend Fallbacks**: The default NN backend now implements depthwise
  convolution, image resize/crop/normalize, scaled dot-product attention, and
  multi-head attention instead of returning null for these runtime paths.
- **Version Guard**: Checkpoint resume validates schema version and fails fast
  by default on incompatible state (optional lenient mode via
  `failOnCheckpointLoadError(false)`).
- **Auto Model Snapshots**: When `checkpointDir(...)` is set, canonical trainer
  saves model weights to `canonical-model.safetensors`, writes
  `canonical-model.metadata`, and reloads only compatible, integrity-checked
  snapshots on resume.
- **Best Model Snapshots**: Validation runs also write
  `canonical-best-model.safetensors` when the monitored validation signal
  improves. The default monitor is validation loss (`MIN`), while
  `.bestModelMonitorMetric("f1", BestModelMonitorMode.MAX)` can choose the best
  epoch by a named validation metric. `restoreBestModelAtEnd()` reloads that best
  epoch before returning, making the trained model immediately deployment-ready.
- **Deterministic Tensor Splits**: `Gollek.DL.trainValidationSplit(...)`
  creates train/validation tensor datasets and ready-to-use seeded loaders for
  reproducible trainer runs. `Gollek.DL.fit(model, split, ...)` can train from
  the split directly.
- **Classification Loader Helpers**: `Gollek.DL.classificationDataLoader(...)`
  and `.classificationTrainValidationSplit(...)` accept Java `int[]` class
  labels and produce CrossEntropy-compatible class-index tensors.
- **Stratified Classification Splits**:
  `Gollek.DL.classificationStratifiedTrainValidationSplit(...)` and
  `.binaryStratifiedTrainValidationSplit(...)` preserve class/binary label
  coverage across train and validation sets for small classification datasets.
- **Multi-Label Stratified Splits**:
  `Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(...)` balances
  per-label positive counts while preserving exact train/validation sizes for
  multi-label BCE datasets.
- **Classification Metric Preset**:
  `Gollek.DL.trainingOptions().classificationMetrics()` enables accuracy,
  macro precision, macro recall, and macro F1 for logits shaped
  `[batch, classes]` with either class-index targets or one-hot targets.
  Use `.topKAccuracyMetric(k)` alongside it when top-2/top-5 style reporting
  is needed.
- **Structured Confusion Matrix Metric**:
  `Gollek.DL.confusionMatrixMetric()` records scalar
  `confusion_matrix_accuracy` plus structured matrix details under
  `latestTrainMetricDetails`, `latestValidationMetricDetails`, epoch history,
  `canonical-history.csv`, and `canonical-report.json`. Rows are actual
  classes and columns are predicted classes.
- **Classification Ranking Metrics**:
  `Gollek.DL.classificationMacroRocAucMetric()`,
  `.classificationMacroAveragePrecisionMetric()`, and
  `.trainingOptions().classificationRankingMetrics()` compute one-vs-rest
  macro ranking quality from multi-class logits.
- **Imbalanced CrossEntropy Weighting**:
  `Gollek.DL.classWeights(...)` derives balanced class weights from Java
  class-index labels, `Gollek.DL.classWeightsFor(numClasses, ...)` handles
  explicit class counts, and
  `Gollek.DL.trainingOptions().crossEntropyClassWeights(...)` applies those
  weights to the preset CrossEntropy trainer path.
- **Focal Classification Preset**:
  `TrainingPreset.CLASSIFICATION_FOCAL_*` uses `Gollek.DL.focalLoss(...)`
  to down-weight easy examples and focus training on hard examples.
  `Gollek.DL.trainingOptions().focalGamma(...)`, `.focalAlpha(...)`, and
  `.focalClassWeights(...)` configure the preset trainer path, while
  non-finite logits fail before softmax/focal weighting.
- **Binary Focal With Logits Preset**:
  `TrainingPreset.BINARY_FOCAL_WITH_LOGITS_*` uses
  `Gollek.DL.binaryFocalWithLogitsLoss(...)` for imbalanced binary and
  multi-label tasks. It accepts `.focalGamma(...)`, `.focalAlpha(...)`,
  and the existing `.bcePositiveWeight(...)` / `.bcePositiveWeights(...)`
  imbalance controls, with finite-logit validation and tested gradients for
  positive-weighted multi-label batches.
- **Binary Classification Preset**:
  `Gollek.DL.binaryDataLoader(...)`, `.binaryTrainValidationSplit(...)`,
  `.binaryStratifiedTrainValidationSplit(...)`, and
  `TrainingPreset.BINARY_BCE_WITH_LOGITS_*` support one-output binary
  classifiers shaped `[batch, 1]` with BCE-with-logits loss.
- **Imbalanced BCE Weighting**:
  `Gollek.DL.binaryPositiveWeight(...)` derives `negatives / positives` from
  Java labels and `Gollek.DL.trainingOptions().bcePositiveWeight(...)` applies
  it to the preset BCE trainer path. For multi-label tasks,
  `Gollek.DL.multiLabelPositiveWeights(...)` and `.bcePositiveWeights(...)`
  apply per-label positive weighting.
- **Binary Metrics**:
  `Gollek.DL.trainingOptions().binaryClassificationMetrics()` records
  thresholded binary accuracy, precision, recall, and F1 using logit `0.0`
  as the probability `0.5` decision boundary.
- **Configurable Binary Threshold Metrics**:
  `Gollek.DL.binaryAccuracyMetric(logitThreshold)`,
  `.binaryPrecisionMetric(logitThreshold)`, `.binaryRecallMetric(logitThreshold)`,
  `.binaryF1Metric(logitThreshold)`, and
  `.trainingOptions().binaryClassificationMetrics(logitThreshold)` evaluate
  BCE/focal logits at deployment-specific operating points.
- **Binary Confusion Matrix Metric**:
  `Gollek.DL.binaryConfusionMatrixMetric()` and
  `.trainingOptions().binaryConfusionMatrixMetric(logitThreshold)` record
  scalar `binary_confusion_matrix_accuracy` plus TN/FP/FN/TP, specificity,
  balanced accuracy, and a structured `[[TN, FP], [FN, TP]]` matrix in
  summary metadata, epoch history, history CSV, evaluation summaries, and
  `canonical-report.json`.
- **Binary Ranking Metrics**:
  `Gollek.DL.binaryRocAucMetric()`, `.binaryAveragePrecisionMetric()`, and
  `.trainingOptions().binaryRankingMetrics()` report ranking quality directly
  from logits, which is especially useful for imbalanced datasets where a
  fixed threshold can hide model quality.
- **Multi-Label BCE Helpers**:
  `Gollek.DL.multiLabelBinaryDataLoader(...)` and
  `.multiLabelBinaryTrainValidationSplit(...)` /
  `.multiLabelBinaryStratifiedTrainValidationSplit(...)` accept Java
  `int[][]`, `boolean[][]`, or `float[][]` targets and preserve
  `[batch, labels]` shape for multi-label BCE training.
- **Multi-Label Metrics**:
  `Gollek.DL.trainingOptions().multiLabelBinaryMetrics()` records exact-match
  accuracy, Hamming loss, macro precision, macro recall, and macro F1 for
  multi-label BCE outputs.
- **Configurable Multi-Label Threshold Metrics**:
  The multi-label metric factories and
  `.trainingOptions().multiLabelBinaryMetrics(logitThreshold)` accept custom
  logit thresholds while preserving the existing logit `0.0` default.
- **Multi-Label Ranking Metrics**:
  `Gollek.DL.multiLabelMacroRocAucMetric()`,
  `.multiLabelMacroAveragePrecisionMetric()`, and
  `.trainingOptions().multiLabelRankingMetrics()` compute per-label macro
  ranking quality from logits, complementing thresholded multi-label metrics.
- **No-Grad Evaluation**: `Gollek.DL.evaluate(...)` measures test/validation
  loss and metrics under `model.eval()` while preserving the caller's previous
  training mode and accelerator preference. Preset losses are supported for the
  common regression/classification paths.

## Example Usage

```java
import tech.kayys.gollek.ml.Gollek;

// Create an instance tied to the local execution backend
Gollek gollek = Gollek.builder()
    .model("Qwen/Qwen2.5-0.5B")
    .device("METAL")
    .build();

// Simple text completion hiding tokenizer complexity
String answer = gollek.createCompletion("What is the capital of France?");
```
