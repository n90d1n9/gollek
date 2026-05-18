This directory is a Gradle namespace root for the `:ml:*` projects.

Phase 1 of the ML framework migration keeps the implementation modules in
their existing `training/` directories and remaps them through
`settings.gradle.kts` using `projectDir`. The physical `ml/` root exists so
Gradle 9 can resolve the intermediate `:ml` project path cleanly.

Current ML API surface also includes:

- `:ml:gollek-ml-byte-latent`
  - initial Java package root for byte-latent language modeling
  - carries byte-sequence batch contracts, latent-state transport types, and
    architecture metadata for Fast Byte Latent Transformer style work
  - keeps byte-native language-modeling code separate from
    `training/gollek-train-diffusion-opd`
- `:ml:gollek-ml-runner-api`
  - unified `tech.kayys.gollek.ml.runner.*` contracts
  - intentionally small, so backend adapters like ONNX can depend on runner
    contracts without pulling the full ML umbrella
- `:ml:gollek-ml-autograd`
  - in-repo compatibility tensor/autograd surface
  - first Gradle-owned foundation for trainer/data modules that previously
    depended on stale Maven-local ML artifacts
  - differentiates composed loss graphs through view ops, reductions, unary
    math/activation ops, and last-dimension softmax so Java-side training loops
    no longer silently drop gradients in common expressions
  - hardens optimizers and gradient clipping with finite hyperparameter,
    parameter, and gradient checks before state mutation; Adam weight decay no
    longer mutates gradient buffers and Adagrad applies weight decay across the
    whole tensor
  - hardens learning-rate scheduler configuration and checkpoint restore for
    StepLR, CosineAnnealingLR, WarmupCosineScheduler, and ReduceLROnPlateau so
    corrupted scheduler state cannot silently poison resumed optimizer rates
  - exposes partial-resume diagnostics for checkpoint artifacts, including
    whether model, optimizer, scheduler, GradScaler, and history files were
    missing when resume began even if they are recreated later in the run;
    strict resume fails closed on missing runtime state or required artifacts
    unless lenient checkpoint loading is explicitly enabled
  - writes `canonical-model.metadata` beside model checkpoints and validates
    model class/parameter signatures plus byte-size/SHA-256 integrity on
    resume, surfacing `checkpointResumeCompatibilityMismatches` when lenient
    fallback is enabled
  - writes `canonical-checkpoints.metadata` for runtime-state artifacts and
    validates runtime, optimizer, scheduler, GradScaler, history, report, and
    model checkpoint integrity before resume deserializes or trusts those files;
    best-model restore-at-end now uses the same guard before replacing weights
  - validates non-null, non-empty, sample-aligned, finite train/validation input
    and label tensors before forward or loss execution, rejecting malformed or
    NaN/Infinity dataset values before they can poison gradients, optimizer
    state, or validation metrics
  - requires custom trainer loss functions to return a single-value tensor,
    rejecting vector/matrix losses before `backward()` can silently sum them
    while logging only the first value
  - validates model predictions before trainer loss/metric evaluation, so
    exploding activations fail as `nonFiniteKind=prediction` instead of
    leaking into custom losses or validation metrics
  - validates train/validation metric snapshots after observed batches, so
    duplicate names and broken or throwing custom metrics fail as
    `invalidMetric*` metadata and stop final checkpoint writes;
    `DetailedMetric` payloads are guarded the same way without rejecting
    intentionally empty evaluation phases
  - exposes trainer metric contracts as standalone `TrainingMetric` and
    `DetailedTrainingMetric` types, with `TrainingMetrics` as the future-facing
    built-in metric catalog backed by domain-split implementations outside
    `CanonicalTrainer`; the nested `CanonicalTrainer.*` metric names remain
    available as compatibility aliases
  - writes nested metric maps and `DetailedMetric` payloads into
    `canonical-history.csv` as deterministic JSON cells and parses them back
    during resume, so dashboard tools do not need to scrape Java map strings;
    malformed structured history cells surface as resume load errors and fail
    closed when strict checkpoint loading is enabled; duplicate headers, blank
    headers, extra row cells, missing epoch keys, non-integer epochs, and
    duplicate epochs are treated as ambiguous history shape errors
  - persists and restores `GradScaler` mixed-precision state while validating
    checkpoint compatibility and avoiding partial gradient unscale on overflow
  - wires mixed precision into `CanonicalTrainer` and
    `Gollek.DL.trainingOptions().mixedPrecision()` so scaled-loss backward,
    overflow skip/backoff, and `canonical-grad-scaler.state` resume happen in
    the public trainer path; custom `GradScaler` instances can be supplied via
    `trainingOptions().gradScaler(...)` for tuned scale growth/backoff policy,
    with strict resume rejection when a saved scaler policy no longer matches
    and explicit fallback metadata for opt-in lenient loading
  - routes gradients through public graph-construction helpers such as
    `cat`, `stack`, and `where`, including broadcasted `where` selections
  - differentiates transformer attention compatibility einsums for
    `QK^T` score tensors and attention-weight/value application
  - preserves gradients through public multi-head attention head split/merge,
    causal mask application, and Transformer block dropout paths
  - uses a differentiable FlashAttention training path while keeping the tiled
    no-grad inference path with correct multi-head output merge
  - applies the public `MultiHeadAttention(..., dropoutP)` attention dropout
    option and propagates encoder/decoder layer dropout into self/cross attention
  - uses a registered GELU activation in `TransformerBlock` feed-forward paths
    instead of falling back to ReLU while documenting transformer-style FFNs
  - backpropagates through rotary position embeddings using the inverse
    rotation, with explicit rank, head-dimension, and max-sequence validation
  - keeps the root `tech.kayys.gollek.ml.nn.GroupNorm` path differentiable by
    sharing the autograd-native reshape/mean/variance formulation used by
    `tech.kayys.gollek.ml.nn.layer.GroupNorm`
  - fails fast on invalid dropout and GroupNorm configuration, including
    non-finite dropout probabilities, invalid epsilons, and channel mismatches
  - normalizes weighted `CrossEntropyLoss` by the sum of selected class/sample
    weights, matching standard weighted-mean semantics for loss and gradients
  - validates class-index targets in CrossEntropy, Focal, LabelSmoothing,
    ArcFace, and CTC losses so wrongly-shaped, non-finite, fractional, or
    out-of-range labels fail before logits are indexed
  - validates classification logits for finite values in `CrossEntropyLoss`,
    `LabelSmoothingLoss`, `FocalLoss`, `BCEWithLogitsLoss`, and
    `BinaryFocalWithLogitsLoss`, with finite-difference coverage for weighted
    BCE/focal gradients
  - validates regression losses (`MSELoss`, `L1Loss`, `SmoothL1Loss`,
    `HuberLoss`) against empty tensors and NaN/Inf predictions or targets,
    with standard beta-scaled `SmoothL1Loss` gradients
  - backpropagates through `LabelSmoothingLoss` with the standard
    `softmax - smoothedTarget` gradient for calibrated classification training
  - computes `BCEWithLogitsLoss` with a stable softplus formulation, preserving
    large losses for confident wrong binary predictions while rejecting
    non-finite logits and invalid labels before training
  - backpropagates through `DiceLoss` for segmentation masks and validates
    probability predictions, binary targets, shapes, and smoothing values
  - backpropagates through `IoULoss` for predicted bounding boxes and validates
    `[batch, 4]` finite, ordered box coordinates before computing overlap
  - backpropagates through active `TripletLoss` margin violations for anchor,
    positive, and negative embeddings with strict `[batch, dim]` validation
  - backpropagates through `ContrastiveLoss` for positive and active negative
    embedding pairs, with strict binary labels and `[batch, dim]` validation
  - backpropagates through `CosineEmbeddingLoss` for positive and active
    negative embedding pairs, with strict `1.0`/`-1.0` labels, margin bounds,
    and `[batch, dim]` validation
  - backpropagates through `CTCLoss` log-probability inputs using log-space
    forward-backward posteriors for speech/OCR sequence training
  - backpropagates through `ArcFaceLoss` embeddings and learned class centers,
    including row-normalization and angular-margin derivatives
- **Knowledge distillation** now trains students from both soft teacher
  distributions and hard labels: the teacher-to-student KL branch has a stable
  closed-form backward path, the hard branch uses real CrossEntropy autograd,
  and teacher logits stay detached.
- **CPU NN backend fallback** now returns real tensors for depthwise convolution,
  image resize/crop/normalize, scaled dot-product attention, and multi-head
  attention instead of null placeholders.
