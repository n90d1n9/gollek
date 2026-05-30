package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Tracks resume-time checkpoint diagnostics exposed in trainer summaries.
 */
final class TrainerCheckpointResumeDiagnostics {
    private static final String UNKNOWN_ARTIFACT = "unknown";
    private static final String UNKNOWN_COMPATIBILITY_REASON = "checkpoint compatibility mismatch";

    private final List<String> compatibilityMismatches = new ArrayList<>();
    private final List<String> manifestEntryMissingArtifacts = new ArrayList<>();

    void recordCompatibilityMismatch(String artifactName, String reason) {
        String artifact = normalize(artifactName, UNKNOWN_ARTIFACT);
        String mismatchReason = normalize(reason, UNKNOWN_COMPATIBILITY_REASON);
        addUnique(compatibilityMismatches, artifact + ": " + mismatchReason);
    }

    void recordManifestEntryMissing(String artifactName) {
        addUnique(manifestEntryMissingArtifacts, normalize(artifactName, UNKNOWN_ARTIFACT));
    }

    List<String> compatibilityMismatches() {
        return List.copyOf(compatibilityMismatches);
    }

    List<String> manifestEntryMissingArtifacts() {
        return List.copyOf(manifestEntryMissingArtifacts);
    }

    static IllegalStateException missingArtifactException(String artifactName, Path checkpointFile) {
        return new IllegalStateException(
                "Missing " + normalize(artifactName, UNKNOWN_ARTIFACT)
                        + " checkpoint artifact for resume: " + checkpointFile);
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

    private static void addUnique(List<String> values, String value) {
        if (!values.contains(value)) {
            values.add(value);
        }
    }

    private static String normalize(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value.trim();
    }
}
