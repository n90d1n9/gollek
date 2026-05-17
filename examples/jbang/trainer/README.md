# Trainer JBang Examples

This folder is the canonical JBang lane for Gollek trainer runtime examples.

## Run

From `gollek/examples/jbang`:

```bash
jbang trainer/trainer_runtime_bootstrap.java
jbang trainer/trainer_diffusion_opd_ddim.java
jbang trainer/trainer_diffusion_opd_ddim.java 4
jbang trainer/trainer_accelerated_autograd.java auto
jbang trainer/trainer_accelerated_autograd.java metal
jbang trainer/trainer_warmup_cosine_scheduler.java auto
jbang trainer/trainer_warmup_cosine_scheduler.java metal
jbang trainer/trainer_reduce_lr_on_plateau.java auto
jbang trainer/trainer_reduce_lr_on_plateau.java metal
jbang trainer/trainer_resume_history.java auto
jbang trainer/trainer_resume_history.java metal
jbang trainer/trainer_robust_regression_huber.java auto
jbang trainer/trainer_robust_regression_huber.java metal
jbang trainer/trainer_classification_metrics.java auto
jbang trainer/trainer_classification_metrics.java metal
jbang trainer/trainer_imbalanced_crossentropy_weighted.java auto
jbang trainer/trainer_imbalanced_crossentropy_weighted.java metal
jbang trainer/trainer_focal_classification.java auto
jbang trainer/trainer_focal_classification.java metal
jbang trainer/trainer_binary_focal_weighted.java auto
jbang trainer/trainer_binary_focal_weighted.java metal
jbang trainer/trainer_binary_bce_metrics.java auto
jbang trainer/trainer_binary_bce_metrics.java metal
jbang trainer/trainer_imbalanced_bce_weighted.java auto
jbang trainer/trainer_imbalanced_bce_weighted.java metal
jbang trainer/trainer_multilabel_bce_metrics.java auto
jbang trainer/trainer_multilabel_bce_metrics.java metal
```

## What This Example Covers

- Trainer runtime mode detection (`legacy-bridge` vs `canonical-fallback`)
- Java-first DiffusionOPD demo with `Gollek.DL.diffusionOpdTrainer()`
- Existing diffusion runner adapter usage via `RunnerDiffusionAdapters`
- DDIM-style ODE mean-matching rollout scaffold for diffusion distillation
- Canonical trainer builder usage (`Trainers.canonicalBuilder()`)
- Training listener lifecycle callbacks
- Training summary reporting
- Public `Gollek.DL.fit(...)` training options
- Public `Gollek.DL.trainValidationSplit(...)` plus deterministic seeded
  train loaders
- Stratified `Gollek.DL.classificationStratifiedTrainValidationSplit(...)`
  `.binaryStratifiedTrainValidationSplit(...)`, and
  `.multiLabelBinaryStratifiedTrainValidationSplit(...)` for label-safe
  classification validation sets
- Direct `Gollek.DL.fit(model, split, ...)` split training
- `Gollek.DL.trainingOptions().meanAbsoluteErrorMetric().mseMetric()` for
  train/validation metrics in the returned training summary
- `Gollek.DL.trainingOptions().regressionMetrics()` for MAE, MSE, RMSE, and
  R2 reporting on regression loaders
- `Gollek.DL.trainingOptions().checkpointDir(...).restoreBestModelAtEnd()` for
  validation-driven best-model snapshots in `canonical-best-model.safetensors`;
  add `.bestModelMonitorMetric("f1", CanonicalTrainer.BestModelMonitorMode.MAX)`
  when a task metric should choose the best epoch instead of validation loss
- `Gollek.DL.trainingOptions().earlyStopping(...).earlyStoppingMonitorMetric(...)`
  when early stopping should follow a validation metric such as F1, accuracy,
  AUROC, or average precision instead of validation loss
- `TrainingSummary.metadata().get("epochHistory")` for per-epoch train loss,
  validation loss, metric values, learning rate, optimizer steps, and scheduler
  steps so trainer runs can be debugged after completion
- Gradient and parameter diagnostics in summary/history metadata, including
  `latestGradientL2NormBeforeClip`, `latestGradientL2Norm`,
  `latestGradientClipped`, and `latestParameterL2Norm`
- Non-finite guard metadata (`nonFiniteDetected`, `nonFinitePhase`,
  `nonFiniteKind`, `nonFiniteOptimizerStepSkipped`) when NaN/Inf loss or
  gradients stop training before corrupted checkpoints are written
- Train/validation throughput diagnostics in summary/history metadata,
  including `trainBatchCount`, `trainSampleCount`, `trainSamplesPerSecond`,
  `validationBatchCount`, and `validationSamplesPerSecond`
- `canonical-history.csv` next to `checkpointDir(...)` checkpoints for a durable
  learning-curve artifact that can be opened in spreadsheets or dashboards;
  `resumeFromCheckpoint()` reloads existing rows before appending new epochs
- `canonical-report.json` next to `checkpointDir(...)` checkpoints for a
  structured run report containing summary metadata, epoch history, accelerator
  selection, checkpoint state, gradient diagnostics, and throughput counters
- Optimizer checkpoint state for SGD, Adam, AdamW, and RMSprop so interrupted
  runs resume with momentum/velocity buffers and Adam moments intact
- `Gollek.DL.TrainingPreset.REGRESSION_HUBER_*` and
  `Gollek.DL.huberLoss(...)` for robust regression with outliers
- `Gollek.DL.trainingOptions().classificationMetrics()` for accuracy, macro
  precision, macro recall, and macro F1 on CrossEntropy-style classification
  loaders
- `Gollek.DL.confusionMatrixMetric()` and
  `.trainingOptions().confusionMatrixMetric()` for structured multi-class
  confusion matrix details in summary metadata, epoch history, and
  `canonical-report.json`
- `Gollek.DL.trainingOptions().topKAccuracyMetric(k)` for top-k
  classification reporting
- `Gollek.DL.classificationMacroRocAucMetric()`,
  `.classificationMacroAveragePrecisionMetric()`, and
  `.trainingOptions().classificationRankingMetrics()` for one-vs-rest
  multi-class ranking quality from CrossEntropy-style logits
- `Gollek.DL.classWeights(...)` / `.classWeightsFor(numClasses, ...)` and
  `.trainingOptions().crossEntropyClassWeights(...)` for imbalanced
  multi-class CrossEntropy training
- `Gollek.DL.TrainingPreset.CLASSIFICATION_FOCAL_*`,
  `Gollek.DL.focalLoss(...)`, and `.trainingOptions().focalGamma(...)` /
  `.focalClassWeights(...)` for hard-example-focused classification training
- `Gollek.DL.TrainingPreset.BINARY_FOCAL_WITH_LOGITS_*`,
  `Gollek.DL.binaryFocalWithLogitsLoss(...)`, and
  `.trainingOptions().focalGamma(...)` / `.focalAlpha(...)` with
  `.bcePositiveWeight(...)` for imbalanced binary or multi-label focal
  training
- `Gollek.DL.binaryStratifiedTrainValidationSplit(...)` and
  `Gollek.DL.trainingOptions().binaryClassificationMetrics()` for
  BCE-with-logits binary classification
- `Gollek.DL.binaryAccuracyMetric(logitThreshold)` and
  `.trainingOptions().binaryClassificationMetrics(logitThreshold)` for
  thresholded binary metrics at deployment-specific logit operating points
- `Gollek.DL.binaryConfusionMatrixMetric(logitThreshold)` and
  `.trainingOptions().binaryConfusionMatrixMetric(logitThreshold)` for
  structured TN/FP/FN/TP diagnostics, specificity, balanced accuracy, and a
  `[[TN, FP], [FN, TP]]` matrix in trainer summaries and reports
- `Gollek.DL.binaryRocAucMetric()`, `.binaryAveragePrecisionMetric()`, and
  `.trainingOptions().binaryRankingMetrics()` for imbalanced binary ranking
  quality beyond a fixed threshold
- `Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW` for one-output
  binary classifiers shaped `[batch, 1]`
- `Gollek.DL.binaryPositiveWeight(...)` and
  `.trainingOptions().bcePositiveWeight(...)` for imbalanced binary BCE
  training without hand-wiring a custom loss
- `Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(...)` for
  multi-label BCE targets shaped `[batch, labels]`
- `Gollek.DL.multiLabelPositiveWeights(...)` plus
  `.trainingOptions().bcePositiveWeights(...)` for per-label imbalanced
  multi-label BCE training
- `Gollek.DL.trainingOptions().multiLabelBinaryMetrics()` for exact-match,
  Hamming loss, macro precision, macro recall, and macro F1 reporting on
  multi-label BCE outputs
- `Gollek.DL.multiLabelMacroF1Metric(logitThreshold)` and
  `.trainingOptions().multiLabelBinaryMetrics(logitThreshold)` for
  thresholded multi-label metrics at a chosen logit operating point
- `Gollek.DL.multiLabelMacroRocAucMetric()`,
  `.multiLabelMacroAveragePrecisionMetric()`, and
  `.trainingOptions().multiLabelRankingMetrics()` for per-label macro
  ranking quality on multi-label BCE/focal outputs
- `Gollek.DL.trainingOptions().stepLrBatches(...)` and
  `.cosineAnnealingLrBatches(...)` for public LR scheduling
- `Gollek.DL.trainingOptions().warmupCosineLrBatches(...)` and
  `.warmupCosineLrEpochs(...)` for Transformer-style linear warmup followed
  by cosine decay, with scheduler state included in summary metadata and
  `canonical-report.json`
- `Gollek.DL.trainingOptions().reduceLrOnPlateauValidationLoss(...)` and
  `.reduceLrOnPlateauMetric(...)` for validation-driven learning-rate
  reductions with checkpointable scheduler state and report metadata
- Public `Gollek.DL.evaluate(...)` for no-grad validation/test loss and
  metrics
- End-to-end toy classification with `trainer_classification_metrics.java`
- End-to-end interrupted/resumed regression with durable history in
  `trainer_resume_history.java`
- End-to-end robust regression with `trainer_robust_regression_huber.java`
- End-to-end warmup/cosine scheduling with
  `trainer_warmup_cosine_scheduler.java`
- End-to-end validation plateau scheduling with
  `trainer_reduce_lr_on_plateau.java`
- End-to-end imbalanced multi-class classification with
  `trainer_imbalanced_crossentropy_weighted.java`
- End-to-end focal-loss classification with `trainer_focal_classification.java`
- End-to-end imbalanced binary focal-loss classification with
  `trainer_binary_focal_weighted.java`
- End-to-end binary classification with `trainer_binary_bce_metrics.java`
- End-to-end imbalanced binary classification with
  `trainer_imbalanced_bce_weighted.java`
- End-to-end multi-label classification with
  `trainer_multilabel_bce_metrics.java`
- Honest accelerator metadata (`executionBackend`, `executionAccelerated`,
  `acceleratedMatmulCalls`)
- End-to-end Java/JBang diffusion distillation walkthrough with
  `trainer_diffusion_opd_ddim.java`

## Prerequisites

From `gollek/` project root:

```bash
./gradlew publishJbangTrainerExamplesToMavenLocal
```

The task is Gradle-only and publishes the local snapshot runtime graph required
by the trainer JBang examples.
