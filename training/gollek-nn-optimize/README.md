# Gollek SDK - Optimizer Suite

Complete optimizer suite for neural network training with Adam, AdamW, SGD, and RMSprop.

## Features

- ✅ **Adam** - Adaptive Moment Estimation with AMSGrad variant
- ✅ **AdamW** - Adam with decoupled weight decay
- ✅ **SGD** - Stochastic Gradient Descent with momentum and Nesterov
- ✅ **RMSprop** - Root Mean Square Propagation
- ✅ **Gradient Clipping** - By norm and by value
- ✅ **Weight Decay** - L2 regularization (Adam/AdamW style)
- ✅ **Optimizer Safety** - Finite hyperparameter, parameter, and gradient
  checks before mutating optimizer state
- ✅ **Knowledge Distillation** - Teacher-to-student KL plus hard-label cross
  entropy with real gradients into student parameters
- ✅ **Builder Pattern** - Fluent API for configuration

## Quick Start

### Adam Optimizer

```java
import tech.kayys.gollek.lib.optimize.Adam;
import tech.kayys.gollek.ml.autograd.GradTensor;

List<GradTensor> parameters = model.parameters();

Optimizer optimizer = Adam.builder(parameters, 0.001)
    .betas(0.9, 0.999)
    .eps(1e-8)
    .weightDecay(0.01)
    .amsgrad(false)
    .build();

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    for (Batch batch : trainLoader) {
        GradTensor loss = model.forward(batch.inputs)
            .mseLoss(batch.targets);
        
        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
    }
}
```

### AdamW (Recommended for Transformers)

```java
import tech.kayys.gollek.lib.optimize.AdamW;

Optimizer optimizer = AdamW.builder(parameters, 0.001)
    .betas(0.9, 0.999)
    .weightDecay(0.01)  // Decoupled weight decay
    .build();
```

### SGD with Momentum

```java
import tech.kayys.gollek.lib.optimize.SGD;

Optimizer optimizer = SGD.builder(parameters, 0.01)
    .momentum(0.9)
    .nesterov(true)       // Nesterov accelerated gradient
    .weightDecay(0.0001)
    .build();
```

### RMSprop

```java
import tech.kayys.gollek.lib.optimize.RMSprop;

Optimizer optimizer = RMSprop.builder(parameters, 0.01)
    .alpha(0.99)
    .eps(1e-8)
    .momentum(0.0)
    .build();
```

### Knowledge Distillation

```java
KnowledgeDistillation distiller = KnowledgeDistillation.builder()
    .teacher(teacherModel)
    .student(studentModel)
    .optimizer(SGD.builder(studentModel.parameters(), 0.01f).build())
    .temperature(4.0f)
    .alpha(0.7f)  // soft KL weight; hard CE weight is 0.3
    .epochs(5)
    .build();

distiller.fit(trainLoader);
```

The distillation loss uses `KL(softmax(teacher/T) || softmax(student/T)) * T^2`
for the soft branch and standard CrossEntropy for hard labels. The teacher is
detached, while the student receives gradients from both branches.

## Gradient Clipping

Prevent exploding gradients:

```java
// Clip by global norm
optimizer.clipGradNorm(1.0);

// Clip by value
optimizer.clipGradValue(-5.0, 5.0);
```

Optimizers and clippers fail fast on NaN/Inf gradients, invalid learning
rates, invalid beta/momentum/epsilon values, and unordered clip ranges before
updating parameters or optimizer moments. Adam's L2 weight decay leaves the
original gradient buffer untouched, and Adagrad applies weight decay uniformly
across the whole tensor.

Learning-rate schedulers also validate finite configuration and checkpoint
state before restoring. `StepLR`, `CosineAnnealingLR`,
`WarmupCosineScheduler`, and `ReduceLROnPlateau` reject NaN learning rates,
negative step counters, infinite plateau metrics, and mismatched scheduler
payloads during resume.

## Optimizer Comparison

| Optimizer | Best For | Speed | Memory | Generalization |
|-----------|----------|-------|--------|----------------|
| **Adam** | Quick prototyping | Fast | Medium | Good |
| **AdamW** | Transformers, LLMs | Fast | Medium | **Best** |
| **SGD** | CNNs, final tuning | Medium | Low | **Best** |
| **RMSprop** | RNNs, RL | Fast | Medium | Good |

## API Reference

### Adam

| Method | Default | Description |
|--------|---------|-------------|
| `betas(beta1, beta2)` | 0.9, 0.999 | Exponential decay rates |
| `eps(eps)` | 1e-8 | Numerical stability |
| `weightDecay(wd)` | 0.0 | L2 regularization |
| `amsgrad(enabled)` | false | Use AMSGrad variant |

### AdamW

| Method | Default | Description |
|--------|---------|-------------|
| `betas(beta1, beta2)` | 0.9, 0.999 | Exponential decay rates |
| `eps(eps)` | 1e-8 | Numerical stability |
| `weightDecay(wd)` | 0.01 | Decoupled weight decay |

### SGD

| Method | Default | Description |
|--------|---------|-------------|
| `momentum(m)` | 0.0 | Momentum factor |
| `nesterov(enabled)` | false | Nesterov accelerated gradient |
| `weightDecay(wd)` | 0.0 | L2 regularization |

### RMSprop

| Method | Default | Description |
|--------|---------|-------------|
| `alpha(alpha)` | 0.99 | Smoothing constant |
| `eps(eps)` | 1e-8 | Numerical stability |
| `momentum(m)` | 0.0 | Momentum factor |
| `weightDecay(wd)` | 0.0 | L2 regularization |

## Maven Dependency

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-ml-optimize</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## Integration with Trainer

```java
import tech.kayys.gollek.lib.train.Trainer;
import tech.kayys.gollek.lib.optimize.AdamW;

Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(AdamW.builder(model.parameters(), 0.001)
        .weightDecay(0.01)
        .build())
    .loss((preds, targets) -> preds.mse(targets))
    .epochs(100)
    .build();

trainer.fit(trainLoader, valLoader);
```

## License

MIT License - Copyright (c) 2026 Kayys.tech
