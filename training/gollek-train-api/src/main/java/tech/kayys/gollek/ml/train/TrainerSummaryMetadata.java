package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Shared helpers for stable trainer summary metadata keys.
 */
final class TrainerSummaryMetadata {
    private TrainerSummaryMetadata() {
    }

    static void putCheckpointResumeOverview(
            Map<String, Object> metadata,
            boolean resumeRequested,
            List<String> missingArtifacts,
            List<String> compatibilityMismatches) {
        List<String> missing = immutableList(missingArtifacts);
        List<String> mismatches = immutableList(compatibilityMismatches);
        metadata.put("checkpointResumeMissingArtifacts", missing);
        metadata.put("checkpointResumeCompatibilityMismatches", mismatches);
        metadata.put("checkpointCompatibilityMismatches", mismatches);
        metadata.put("checkpointCompatibilityMismatch", !mismatches.isEmpty());
        metadata.put(
                "checkpointResumePartial",
                resumeRequested && (!missing.isEmpty() || !mismatches.isEmpty()));
    }

    static void putCheckpointStatus(
            Map<String, Object> metadata,
            String prefix,
            boolean enabled,
            boolean resumeRequested,
            Path path,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
        metadata.put(prefix + "Enabled", enabled);
        metadata.put(prefix + "ResumeRequested", resumeRequested);
        putArtifactState(metadata, prefix, path, missingOnResume, loaded, saved, loadError, saveError);
    }

    static void putSupportedCheckpointStatus(
            Map<String, Object> metadata,
            String prefix,
            boolean enabled,
            boolean supported,
            boolean resumeRequested,
            Path path,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
        metadata.put(prefix + "Enabled", enabled);
        metadata.put(prefix + "Supported", supported);
        metadata.put(prefix + "ResumeRequested", resumeRequested);
        putArtifactState(metadata, prefix, path, missingOnResume, loaded, saved, loadError, saveError);
    }

    static void putArtifactStatus(
            Map<String, Object> metadata,
            String prefix,
            boolean enabled,
            Path path,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
        metadata.put(prefix + "Enabled", enabled);
        putArtifactState(metadata, prefix, path, missingOnResume, loaded, saved, loadError, saveError);
    }

    static void putSaveOnlyArtifactStatus(
            Map<String, Object> metadata,
            String prefix,
            boolean enabled,
            Path path,
            boolean saved,
            String saveError) {
        metadata.put(prefix + "Enabled", enabled);
        metadata.put(prefix + "Present", TrainerMetadataSupport.filePresent(path));
        metadata.put(prefix + "Saved", saved);
        metadata.put(prefix + "SaveFailed", saveError != null);
    }

    static void putOptionalPath(Map<String, Object> metadata, String key, Path path) {
        putOptionalPath(metadata, key, path, true);
    }

    static void putOptionalPath(Map<String, Object> metadata, String key, Path path, boolean include) {
        if (include && path != null) {
            metadata.put(key, path.toString());
        }
    }

    static void putOptionalError(Map<String, Object> metadata, String key, String error) {
        if (error != null) {
            metadata.put(key, error);
        }
    }

    private static List<String> immutableList(List<String> values) {
        return values == null || values.isEmpty() ? List.of() : List.copyOf(values);
    }

    private static void putArtifactState(
            Map<String, Object> metadata,
            String prefix,
            Path path,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
        metadata.put(prefix + "Present", TrainerMetadataSupport.filePresent(path));
        metadata.put(prefix + "MissingOnResume", missingOnResume);
        metadata.put(prefix + "Loaded", loaded);
        metadata.put(prefix + "Saved", saved);
        metadata.put(prefix + "LoadFailed", loadError != null);
        metadata.put(prefix + "SaveFailed", saveError != null);
    }
}
