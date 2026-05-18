package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Tracks resume-time checkpoint diagnostics exposed in trainer summaries.
 */
final class TrainerCheckpointResumeDiagnostics {
    private final List<String> compatibilityMismatches = new ArrayList<>();

    void recordCompatibilityMismatch(String artifactName, String reason) {
        String value = artifactName + ": " + reason;
        if (!compatibilityMismatches.contains(value)) {
            compatibilityMismatches.add(value);
        }
    }

    List<String> compatibilityMismatches() {
        return List.copyOf(compatibilityMismatches);
    }

    static IllegalStateException missingArtifactException(String artifactName, Path checkpointFile) {
        return new IllegalStateException(
                "Missing " + artifactName + " checkpoint artifact for resume: " + checkpointFile);
    }

    static List<String> missingArtifacts(
            boolean modelMissing,
            boolean optimizerMissing,
            boolean schedulerMissing,
            boolean gradScalerMissing,
            boolean historyMissing) {
        List<String> missing = new ArrayList<>();
        addIfMissing(missing, modelMissing, "model");
        addIfMissing(missing, optimizerMissing, "optimizer");
        addIfMissing(missing, schedulerMissing, "scheduler");
        addIfMissing(missing, gradScalerMissing, "gradScaler");
        addIfMissing(missing, historyMissing, "history");
        return List.copyOf(missing);
    }

    private static void addIfMissing(List<String> missing, boolean isMissing, String artifactName) {
        if (isMissing) {
            missing.add(artifactName);
        }
    }
}
