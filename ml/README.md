This directory is a Gradle namespace root for the `:ml:*` projects.

Phase 1 of the ML framework migration keeps the implementation modules in
their existing `training/` directories and remaps them through
`settings.gradle.kts` using `projectDir`. The physical `ml/` root exists so
Gradle 9 can resolve the intermediate `:ml` project path cleanly.

Current ML API surface also includes:

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
    ArcFace, and CTC losses so non-finite, fractional, or out-of-range labels
    fail before logits are indexed
  - backpropagates through `LabelSmoothingLoss` with the standard
    `softmax - smoothedTarget` gradient for calibrated classification training
  - computes `BCEWithLogitsLoss` with a stable softplus formulation, preserving
    large losses for confident wrong binary predictions and rejecting NaN labels
  - backpropagates through `DiceLoss` for segmentation masks and validates
    probability predictions, binary targets, shapes, and smoothing values
  - backpropagates through `IoULoss` for predicted bounding boxes and validates
    `[batch, 4]` finite, ordered box coordinates before computing overlap
