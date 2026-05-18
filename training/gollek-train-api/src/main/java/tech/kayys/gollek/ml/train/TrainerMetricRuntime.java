package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.trainer.api.TrainerSession;

/**
 * Coordinates trainer metric reset, update, and snapshot rules.
 */
final class TrainerMetricRuntime {
    private final List<TrainingMetric> trainMetrics;
    private final List<TrainingMetric> validationMetrics;
    private final TrainerMetricSnapshots.FailureRecorder failures;

    TrainerMetricRuntime(
            List<TrainingMetric> trainMetrics,
            List<TrainingMetric> validationMetrics,
            TrainerMetricSnapshots.FailureRecorder failures) {
        this.trainMetrics = List.copyOf(Objects.requireNonNull(trainMetrics, "trainMetrics must not be null"));
        this.validationMetrics = List.copyOf(
                Objects.requireNonNull(validationMetrics, "validationMetrics must not be null"));
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
    }

    void reset() {
        TrainerMetricSnapshots.reset(trainMetrics);
        TrainerMetricSnapshots.reset(validationMetrics);
    }

    void updateTrain(GradTensor predictions, GradTensor targets) {
        update(trainMetrics, predictions, targets);
    }

    void updateValidation(GradTensor predictions, GradTensor targets) {
        update(validationMetrics, predictions, targets);
    }

    MetricSnapshot snapshotTrain(TrainerSession session, ThroughputSnapshot throughput) {
        return snapshot(session, trainMetrics, "train", finiteRequired(throughput));
    }

    MetricSnapshot snapshotValidation(TrainerSession session, ThroughputSnapshot throughput) {
        return snapshot(session, validationMetrics, "validation", finiteRequired(throughput));
    }

    private MetricSnapshot snapshot(
            TrainerSession session,
            List<TrainingMetric> metrics,
            String phase,
            boolean finiteRequired) {
        return new MetricSnapshot(
                TrainerMetricSnapshots.snapshotValues(session, metrics, phase, finiteRequired, failures),
                TrainerMetricSnapshots.snapshotDetails(session, metrics, phase, finiteRequired, failures));
    }

    private static void update(
            List<TrainingMetric> metrics,
            GradTensor predictions,
            GradTensor targets) {
        for (TrainingMetric metric : metrics) {
            metric.update(predictions, targets);
        }
    }

    private static boolean finiteRequired(ThroughputSnapshot throughput) {
        return throughput != null && throughput.hasBatchesAndSamples();
    }

    record MetricSnapshot(Map<String, Double> values, Map<String, Object> details) {
    }
}
