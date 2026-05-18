package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Metric contract for metrics that expose structured details such as
 * confusion matrices in addition to a scalar value.
 */
public interface DetailedTrainingMetric extends TrainingMetric {
    Map<String, Object> details();
}
