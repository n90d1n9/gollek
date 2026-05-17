package tech.kayys.gollek.trainer.api;

import java.util.Map;

/**
 * Immutable end-of-run summary for trainer integrations and reporting.
 */
public record TrainingSummary(
        int epochCount,
        double bestValidationLoss,
        int bestValidationEpoch,
        Double latestTrainLoss,
        Double latestValidationLoss,
        long durationMs,
        Map<String, Object> metadata) {
}
