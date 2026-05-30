package tech.kayys.gollek.ml.train;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Updates a custom metric state with one prediction/target batch.
 *
 * @param <S> mutable state owned by one metric instance
 */
@FunctionalInterface
public interface TrainingMetricUpdater<S> {
    void update(S state, GradTensor predictions, GradTensor targets);
}
