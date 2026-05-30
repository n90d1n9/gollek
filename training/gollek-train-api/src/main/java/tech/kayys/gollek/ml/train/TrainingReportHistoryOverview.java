package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.mapValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalInt;

import java.util.Map;
import java.util.OptionalInt;

/**
 * Type-safe aggregate view over the canonical report history summary.
 */
public record TrainingReportHistoryOverview(
        boolean available,
        int size,
        OptionalInt firstEpoch,
        OptionalInt lastEpoch,
        TrainingReportLossSummary trainLoss,
        Map<String, TrainingReportMetricSummary> trainMetrics,
        TrainingReportLossSummary validationLoss,
        Map<String, TrainingReportMetricSummary> validationMetrics,
        TrainingReportGeneralizationSummary generalization,
        TrainingReportLearningRateSummary learningRate,
        TrainingReportOptimizationSummary optimization) {
    public TrainingReportHistoryOverview {
        size = Math.max(0, size);
        firstEpoch = firstEpoch == null ? OptionalInt.empty() : firstEpoch;
        lastEpoch = lastEpoch == null ? OptionalInt.empty() : lastEpoch;
        trainLoss = trainLoss == null ? TrainingReportLossSummary.empty() : trainLoss;
        trainMetrics = trainMetrics == null ? Map.of() : Map.copyOf(trainMetrics);
        validationLoss = validationLoss == null ? TrainingReportLossSummary.empty() : validationLoss;
        validationMetrics = validationMetrics == null ? Map.of() : Map.copyOf(validationMetrics);
        generalization = generalization == null ? TrainingReportGeneralizationSummary.empty() : generalization;
        learningRate = learningRate == null ? TrainingReportLearningRateSummary.empty() : learningRate;
        optimization = optimization == null ? TrainingReportOptimizationSummary.empty() : optimization;
    }

    public static TrainingReportHistoryOverview empty() {
        return new TrainingReportHistoryOverview(
                false,
                0,
                OptionalInt.empty(),
                OptionalInt.empty(),
                TrainingReportLossSummary.empty(),
                Map.of(),
                TrainingReportLossSummary.empty(),
                Map.of(),
                TrainingReportGeneralizationSummary.empty(),
                TrainingReportLearningRateSummary.empty(),
                TrainingReportOptimizationSummary.empty());
    }

    public static TrainingReportHistoryOverview fromMap(Map<String, ?> summary) {
        if (summary == null || summary.isEmpty()) {
            return empty();
        }
        return new TrainingReportHistoryOverview(
                booleanValue(summary.get("available")),
                intValue(summary.get("size"), 0),
                optionalInt(summary.get("firstEpoch")),
                optionalInt(summary.get("lastEpoch")),
                TrainingReportLossSummary.fromMap(mapValue(summary, "trainLoss")),
                TrainingReportMetricSummary.fromMaps(mapValue(summary, "trainMetrics")),
                TrainingReportLossSummary.fromMap(mapValue(summary, "validationLoss")),
                TrainingReportMetricSummary.fromMaps(mapValue(summary, "validationMetrics")),
                TrainingReportGeneralizationSummary.fromMap(mapValue(summary, "generalization")),
                TrainingReportLearningRateSummary.fromMap(mapValue(summary, "learningRate")),
                TrainingReportOptimizationSummary.fromMap(mapValue(summary, "optimization")));
    }

    public boolean hasTrainMetrics() {
        return !trainMetrics.isEmpty();
    }

    public boolean hasValidationMetrics() {
        return !validationMetrics.isEmpty();
    }
}
