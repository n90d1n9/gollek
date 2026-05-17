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
