package tech.kayys.gollek.ml.train;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Streaming training/evaluation metric contract used by the canonical trainer.
 */
public interface TrainingMetric {
    String name();

    void reset();

    void update(GradTensor predictions, GradTensor targets);

    double value();
}
