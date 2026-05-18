# Gollek SDK :: Neural Network Modules

The `gollek-ml-nn` component offers an object-oriented Module API (akin to PyTorch's `nn.Module`) built on top of the underlying Autograd tape.

## Modules provided:
- `Linear`: Computes affine transformations $y = xW^T + b$ using gradients.
- `Module`: The core abstract class that organizes `Parameter` containers sequentially. 

## Utilities
- Provides state dictionary parsing for module parameter freezing and gradient initialization.
- `GradScaler` now performs real mixed-precision loss scaling: call
  `GradTensor scaledLoss = scaler.scale(loss)`, backpropagate the returned
  tensor, then unscale via `scaler.unscaleAndCheck(optimizer)` before stepping.
  Its runtime state can be saved/restored beside optimizer checkpoints, and
  overflow checks validate gradients before writing unscaled values back.
  `CanonicalTrainer` uses this same scaler when mixed precision is enabled,
  including custom scalers passed through `trainingOptions().gradScaler(...)`.
- Optimizers and `GradientClipper` reject invalid hyperparameters, non-finite
  gradients, and non-finite parameter values before mutating training state;
  Adam weight decay keeps gradients intact and Adagrad applies decay to every
  tensor element.
- Learning-rate schedulers reject invalid configuration and corrupted
  checkpoint state before restoring resumed optimizer learning rates.
- Focal losses (`FocalLoss`, `BinaryFocalWithLogitsLoss`) reject empty or
  non-finite logits before focal weighting; binary focal gradients are covered
  for positive-weighted multi-label batches.
- Classification losses now require class-index targets shaped exactly
  `[batch]`, and CrossEntropy, LabelSmoothing, BCE-with-logits, and focal
  losses reject non-finite logits before softmax/sigmoid loss computation.
- Regression losses (`MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`) now
  reject empty tensors and NaN/Inf values before loss/gradient computation;
  `SmoothL1Loss(beta)` uses standard beta-scaled SmoothL1 semantics.

```java
import tech.kayys.gollek.ml.nn.*;

class MLP extends Module {
    Linear fc1 = new Linear(128, 64);
    Linear fc2 = new Linear(64, 10);
}
```

```java
var scaler = GradScaler.builder().initScale(65536.0).build();
GradTensor scaledLoss = scaler.scale(loss);
scaledLoss.backward();
if (!scaler.unscaleAndCheck(optimizer)) {
    scaler.step(optimizer);
}
scaler.update();
optimizer.zeroGrad();
```
