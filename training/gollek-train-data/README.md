# Gollek SDK :: Data Loaders

The `gollek-ml-data` module is responsible for bridging incoming data types to memory-efficient `MultiTensor` constructs compatible with the inference engines.

## Key Features
- **Dataset Abstraction**: Implements `Dataset<T>` maps and utilities to lazy-load training/inference data.
- **Batched Iteration**: `DataLoader` clusters heterogeneous input records into memory-contiguous `DType`/`Device` tensors.
- **Data Transforms**: Functional map pipelines that pre-process data before passing it into `gollek-ml-autograd`.
- **Tensor Training Loader**: `DataLoader.tensorDataset(...)`,
  `DataLoader.split(...)`, and seeded `TensorDataLoader` builders provide
  reproducible mini-batches for the canonical trainer.
- **Classification Labels**: `DataLoader.classification(...)` and
  `DataLoader.classificationSplit(...)` turn Java `int[]` class labels into
  CrossEntropy-compatible class-index tensors.

## Typical Workflow
In inference setups, input strings or vectors bypass the `DataLoader` directly to the `Pipeline`. But for training regimens, `gollek-ml-data` allows:
```java
var split = DataLoader.split(inputs, labels, 0.8, 42L);
var train = split.trainLoader(64, true, 42L);
var validation = split.validationLoader(64);

for (var batch : train) {
    // Feed batch.inputs() and batch.labels() into the trainer
}
```
