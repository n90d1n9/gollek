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
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues =
                TrainerCheckpointResumeIssueMetadata.snapshot(
                        missingArtifacts,
                        compatibilityMismatches,
                        manifestEntryMissingArtifacts);
        metadata.put("checkpointResumeMissingArtifacts", resumeIssues.missingArtifacts());
        metadata.put("checkpointResumeCompatibilityMismatches", resumeIssues.compatibilityMismatches());
        metadata.put("checkpointCompatibilityMismatches", resumeIssues.compatibilityMismatches());
        metadata.put("checkpointCompatibilityMismatch", resumeIssues.hasCompatibilityMismatches());
        metadata.put("checkpointResumeManifestEntryMissingArtifacts", resumeIssues.manifestEntryMissingArtifacts());
        metadata.put("checkpointResumeManifestEntryMissing", resumeIssues.hasManifestEntryMissingArtifacts());
        metadata.put("checkpointResumeIssueDetected", resumeIssues.issueDetected());
        metadata.put("checkpointResumeIssueCount", resumeIssues.issueCount());
        metadata.put("checkpointResumeIssueKinds", resumeIssues.issueKinds());
        metadata.put("checkpointResumeAffectedArtifacts", resumeIssues.affectedArtifacts());
        metadata.put("checkpointResumeIssues", resumeIssues.issues());
        metadata.put("checkpointResumeBlockingIssueDetected", resumeIssues.blockingIssueDetected());
        metadata.put("checkpointResumeBlockingIssueCount", resumeIssues.blockingIssueCount());
        metadata.put("checkpointResumeBlockingIssueKinds", resumeIssues.blockingIssueKinds());
        metadata.put("checkpointResumeBlockingIssues", resumeIssues.blockingIssues());
        metadata.put("checkpointResumeIssueCodes", resumeIssues.issueCodes());
        metadata.put("checkpointResumeIssueSeverities", resumeIssues.issueSeverities());
        metadata.put("checkpointResumeRecommendedActions", resumeIssues.recommendedActions());
        putCheckpointResumePrimaryIssue(metadata, resumeIssues);
        metadata.put("checkpointResumeIssueCountsByKind", resumeIssues.issueCountsByKind());
        metadata.put("checkpointResumeIssueSeveritiesByKind", resumeIssues.issueSeveritiesByKind());
        metadata.put("checkpointResumeIssueBlockingByKind", resumeIssues.issueBlockingByKind());
        metadata.put("checkpointResumeIssueArtifactsByKind", resumeIssues.issueArtifactsByKind());
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(resumeRequested, resumeIssues);
        metadata.put("checkpointResumeComplete", plan.complete());
        metadata.put("checkpointResumeStatus", plan.status());
        metadata.put("checkpointResumeStateUsable", plan.stateUsable());
        metadata.put("checkpointResumeRecommendedMode", plan.recommendedMode());
        metadata.put("checkpointResumeNextAction", plan.nextAction());
        metadata.put("checkpointResumePlan", plan.metadata());
        metadata.put("checkpointResumePartial", resumeRequested && resumeIssues.issueDetected());
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

    private static void putCheckpointResumePrimaryIssue(
            Map<String, Object> metadata,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        metadata.put("checkpointResumePrimaryIssueAvailable", resumeIssues.primaryIssueAvailable());
        putCheckpointResumePrimaryBlockingIssue(metadata, resumeIssues);
        if (!resumeIssues.primaryIssueAvailable()) {
            return;
        }
        metadata.put("checkpointResumePrimaryIssueKind", resumeIssues.primaryIssueKind());
        metadata.put("checkpointResumePrimaryIssueCode", resumeIssues.primaryIssueCode());
        metadata.put("checkpointResumePrimaryIssueSeverity", resumeIssues.primaryIssueSeverity());
        metadata.put("checkpointResumePrimaryIssueBlocking", resumeIssues.primaryIssueBlocking());
        metadata.put("checkpointResumePrimaryAffectedArtifact", resumeIssues.primaryAffectedArtifact());
        metadata.put("checkpointResumePrimaryIssueMessage", resumeIssues.primaryIssueMessage());
        metadata.put("checkpointResumePrimaryRecommendedAction", resumeIssues.primaryRecommendedAction());
    }

    private static void putCheckpointResumePrimaryBlockingIssue(
            Map<String, Object> metadata,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        metadata.put("checkpointResumePrimaryBlockingIssueAvailable", resumeIssues.primaryBlockingIssueAvailable());
        if (!resumeIssues.primaryBlockingIssueAvailable()) {
            return;
        }
        metadata.put("checkpointResumePrimaryBlockingIssueKind", resumeIssues.primaryBlockingIssueKind());
        metadata.put("checkpointResumePrimaryBlockingIssueCode", resumeIssues.primaryBlockingIssueCode());
        metadata.put("checkpointResumePrimaryBlockingIssueSeverity", resumeIssues.primaryBlockingIssueSeverity());
        metadata.put("checkpointResumePrimaryBlockingAffectedArtifact", resumeIssues.primaryBlockingAffectedArtifact());
        metadata.put("checkpointResumePrimaryBlockingIssueMessage", resumeIssues.primaryBlockingIssueMessage());
        metadata.put("checkpointResumePrimaryBlockingRecommendedAction", resumeIssues.primaryBlockingRecommendedAction());
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
