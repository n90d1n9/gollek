package tech.kayys.gollek.ml.train;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Builds the persisted trainer report payload.
 */
final class TrainerTrainingReport {
    static final String SCHEMA = "gollek.canonical-trainer.report.v1";

    private TrainerTrainingReport() {
    }

    static Map<String, Object> payload(TrainingSummary summary, Instant generatedAt) {
        Objects.requireNonNull(summary, "summary must not be null");
        Objects.requireNonNull(generatedAt, "generatedAt must not be null");

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema", SCHEMA);
        payload.put("generatedAt", generatedAt.toString());
        payload.put("epochCount", summary.epochCount());
        payload.put("bestValidationLoss", summary.bestValidationLoss());
        payload.put("bestValidationEpoch", summary.bestValidationEpoch());
        payload.put("latestTrainLoss", summary.latestTrainLoss());
        payload.put("latestValidationLoss", summary.latestValidationLoss());
        payload.put("durationMs", summary.durationMs());
        payload.put("metadata", summary.metadata());
        return payload;
    }
}
