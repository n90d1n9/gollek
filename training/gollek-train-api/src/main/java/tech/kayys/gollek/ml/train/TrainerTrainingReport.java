package tech.kayys.gollek.ml.train;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
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
        List<Map<String, Object>> history = TrainerReportHistory.rows(summary.metadata());

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema", SCHEMA);
        payload.put("generatedAt", generatedAt.toString());
        payload.put("epochCount", summary.epochCount());
        payload.put("bestValidationLoss", summary.bestValidationLoss());
        payload.put("bestValidationEpoch", summary.bestValidationEpoch());
        payload.put("latestTrainLoss", summary.latestTrainLoss());
        payload.put("latestValidationLoss", summary.latestValidationLoss());
        payload.put("durationMs", summary.durationMs());
        payload.put("history", history);
        payload.put("historySummary", TrainerReportHistory.summary(history));
        payload.put("runHealth", TrainingReportRunHealth.fromMetadata(summary.metadata()).toMap());
        payload.put("dataHealth", TrainingReportDataHealth.fromMetadata(summary.metadata()).toMap());
        List<TrainingReportDiagnostics.Finding> diagnostics = TrainingReportDiagnostics.analyze(payload);
        payload.put("diagnostics", TrainingReportDiagnostics.toMaps(diagnostics));
        payload.put("diagnosticsSummary", TrainingReportDiagnostics.summary(diagnostics));
        payload.put("metadata", summary.metadata());
        return payload;
    }
}
