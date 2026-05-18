package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Persists runtime-facing trainer artifacts without owning trainer state policy.
 */
final class TrainerRuntimeArtifactWriter {
    private TrainerRuntimeArtifactWriter() {
    }

    static WriteResult writeHistory(Path historyFile, List<Map<String, Object>> rows) {
        if (historyFile == null) {
            return WriteResult.skipped();
        }
        try {
            TrainerCheckpointIO.writeStringAtomically(historyFile, TrainingHistoryCsv.write(rows));
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    static WriteResult writeReport(Path reportFile, TrainingSummary summary, Instant generatedAt) {
        if (reportFile == null || summary == null) {
            return WriteResult.skipped();
        }
        try {
            Map<String, Object> report = TrainerTrainingReport.payload(summary, generatedAt);
            TrainerCheckpointIO.writeStringAtomically(reportFile, TrainerJson.toJson(report) + "\n");
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    static WriteResult writeManifest(
            Path manifestFile,
            Map<String, Path> artifacts,
            int formatVersion,
            Instant generatedAt) {
        if (manifestFile == null) {
            return WriteResult.skipped();
        }
        try {
            TrainerCheckpointManifest.write(manifestFile, artifacts, formatVersion, generatedAt);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    record WriteResult(boolean written, String error) {
        static WriteResult skipped() {
            return new WriteResult(false, null);
        }

        static WriteResult success() {
            return new WriteResult(true, null);
        }

        static WriteResult failed(Exception error) {
            return new WriteResult(false, error.getMessage());
        }
    }
}
