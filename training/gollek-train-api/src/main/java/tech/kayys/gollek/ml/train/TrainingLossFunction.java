package tech.kayys.gollek.ml.train;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Loss contract used by trainer implementations.
 */
@FunctionalInterface
public interface TrainingLossFunction {
    GradTensor compute(GradTensor predictions, GradTensor targets);
}
