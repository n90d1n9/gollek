package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Publishes trainer runtime artifact state for history, reports, manifests, and runtime checkpoints.
 */
final class TrainerRuntimeArtifactMetadata {
    private TrainerRuntimeArtifactMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            Artifact history,
            SaveOnlyArtifact report,
            Artifact manifest,
            boolean manifestIntegrityMismatch,
            RuntimeCheckpoint runtimeCheckpoint) {
        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "trainingHistory",
                history.enabled(),
                history.file(),
                history.missingOnResume(),
                history.loaded(),
                history.saved(),
                history.loadError(),
                history.saveError());
        TrainerSummaryMetadata.putSaveOnlyArtifactStatus(
                metadata,
                "trainingReport",
                report.enabled(),
                report.file(),
                report.saved(),
                report.saveError());
        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "checkpointManifest",
                manifest.enabled(),
                manifest.file(),
                manifest.missingOnResume(),
                manifest.loaded(),
                manifest.saved(),
                manifest.loadError(),
                manifest.saveError());
        metadata.put("checkpointManifestIntegrityMismatch", manifestIntegrityMismatch);
        metadata.put("runtimeCheckpointPresent", TrainerMetadataSupport.filePresent(runtimeCheckpoint.file()));
        metadata.put("runtimeCheckpointResumeAllowed", runtimeCheckpoint.resumeAllowed());
        metadata.put("runtimeCheckpointIntegrityMismatch", runtimeCheckpoint.integrityMismatch());
        metadata.put("runtimeCheckpointResumeSkipped", runtimeCheckpoint.resumeSkipped());
        metadata.put("runtimeCheckpointLoadFailed", runtimeCheckpoint.loadError() != null);
        metadata.put("runtimeCheckpointResumeDecision", runtimeCheckpoint.decisionCode());
        metadata.put("runtimeCheckpointRecommendedAction", runtimeCheckpoint.recommendedAction());
        metadata.put("runtimeCheckpointResumePlan", runtimeCheckpoint.resumePlan());
    }

    record Artifact(
            boolean enabled,
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
    }

    record SaveOnlyArtifact(
            boolean enabled,
            Path file,
            boolean saved,
            String saveError) {
    }

    record RuntimeCheckpoint(
            Path file,
            boolean resumeAllowed,
            boolean integrityMismatch,
            boolean resumeSkipped,
            String loadError,
            String decisionCode,
            String recommendedAction) {
        RuntimeCheckpoint(
                Path file,
                boolean integrityMismatch,
                boolean resumeSkipped,
                String loadError) {
            this(
                    file,
                    defaultResumeAllowed(file, integrityMismatch, resumeSkipped),
                    integrityMismatch,
                    resumeSkipped,
                    loadError,
                    defaultDecisionCode(file, integrityMismatch, resumeSkipped),
                    defaultRecommendedAction(file, integrityMismatch, resumeSkipped));
        }

        RuntimeCheckpoint(
                Path file,
                boolean integrityMismatch,
                boolean resumeSkipped,
                String loadError,
                String decisionCode,
                String recommendedAction) {
            this(
                    file,
                    defaultResumeAllowed(file, integrityMismatch, resumeSkipped),
                    integrityMismatch,
                    resumeSkipped,
                    loadError,
                    decisionCode,
                    recommendedAction);
        }

        RuntimeCheckpoint {
            decisionCode = decisionCode == null || decisionCode.isBlank()
                    ? defaultDecisionCode(file, integrityMismatch, resumeSkipped)
                    : decisionCode;
            recommendedAction = recommendedAction == null || recommendedAction.isBlank()
                    ? defaultRecommendedAction(file, integrityMismatch, resumeSkipped)
                    : recommendedAction;
        }

        Map<String, Object> resumePlan() {
            Map<String, Object> plan = new LinkedHashMap<>();
            plan.put("version", 1);
            plan.put("present", TrainerMetadataSupport.filePresent(file));
            plan.put("resumeAllowed", resumeAllowed);
            plan.put("integrityMismatch", integrityMismatch);
            plan.put("resumeSkipped", resumeSkipped);
            plan.put("loadFailed", loadError != null);
            plan.put("decision", decisionCode);
            plan.put("recommendedAction", recommendedAction);
            if (loadError != null) {
                plan.put("loadError", loadError);
            }
            return Collections.unmodifiableMap(plan);
        }

        private static boolean defaultResumeAllowed(
                Path file,
                boolean integrityMismatch,
                boolean resumeSkipped) {
            return file != null && !integrityMismatch && !resumeSkipped;
        }

        private static String defaultDecisionCode(
                Path file,
                boolean integrityMismatch,
                boolean resumeSkipped) {
            if (integrityMismatch) {
                return "runtime-checkpoint-integrity-mismatch";
            }
            if (resumeSkipped) {
                return "runtime-checkpoint-resume-skipped";
            }
            return file == null ? "runtime-checkpoint-not-configured" : "runtime-checkpoint-ready";
        }

        private static String defaultRecommendedAction(
                Path file,
                boolean integrityMismatch,
                boolean resumeSkipped) {
            if (integrityMismatch) {
                return "skip runtime checkpoint and rebuild runtime state from trainer artifacts";
            }
            if (resumeSkipped) {
                return "continue without runtime checkpoint resume";
            }
            return file == null
                    ? "continue without runtime checkpoint integrity check"
                    : "continue with runtime checkpoint resume when requested";
        }
    }
}
