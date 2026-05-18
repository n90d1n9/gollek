package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Applies runtime artifact save policy, including manifest refresh after visible artifacts change.
 */
final class TrainerRuntimeArtifactPersistence {
    private TrainerRuntimeArtifactPersistence() {
    }

    static ArtifactResult persistHistory(
            Path historyFile,
            List<Map<String, Object>> rows,
            ManifestRequest manifestRequest) {
        SaveResult history = fromWriteResult(TrainerRuntimeArtifactWriter.writeHistory(historyFile, rows));
        return new ArtifactResult(history, refreshManifestIfSaved(history, manifestRequest));
    }

    static ArtifactResult persistReport(
            Path reportFile,
            TrainingSummary summary,
            ManifestRequest manifestRequest) {
        SaveResult report = fromWriteResult(TrainerRuntimeArtifactWriter.writeReport(
                reportFile,
                summary,
                Instant.now()));
        return new ArtifactResult(report, refreshManifestIfSaved(report, manifestRequest));
    }

    static SaveResult persistManifest(ManifestRequest manifestRequest) {
        if (manifestRequest == null) {
            return SaveResult.skipped();
        }
        return fromWriteResult(TrainerRuntimeArtifactWriter.writeManifest(
                manifestRequest.manifestFile(),
                manifestRequest.artifacts(),
                manifestRequest.formatVersion(),
                Instant.now()));
    }

    private static SaveResult refreshManifestIfSaved(SaveResult artifact, ManifestRequest manifestRequest) {
        return artifact.saved() ? persistManifest(manifestRequest) : SaveResult.skipped();
    }

    private static SaveResult fromWriteResult(TrainerRuntimeArtifactWriter.WriteResult writeResult) {
        if (writeResult.written()) {
            return SaveResult.savedResult();
        }
        if (writeResult.error() != null) {
            return SaveResult.failed(writeResult.error());
        }
        return SaveResult.skipped();
    }

    record ManifestRequest(Path manifestFile, Map<String, Path> artifacts, int formatVersion) {
    }

    record ArtifactResult(SaveResult artifact, SaveResult manifest) {
    }

    record SaveResult(boolean stateChanged, boolean saved, String error) {
        static SaveResult skipped() {
            return new SaveResult(false, false, null);
        }

        static SaveResult savedResult() {
            return new SaveResult(true, true, null);
        }

        static SaveResult failed(String error) {
            return new SaveResult(true, false, error);
        }
    }
}
