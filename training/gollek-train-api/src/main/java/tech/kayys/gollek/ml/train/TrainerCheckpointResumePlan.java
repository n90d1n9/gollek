package tech.kayys.gollek.ml.train;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Owns checkpoint resume policy fields before they are published in trainer summaries.
 */
final class TrainerCheckpointResumePlan {
    private static final int VERSION = 1;

    private TrainerCheckpointResumePlan() {
    }

    static Snapshot snapshot(
            boolean resumeRequested,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        boolean complete = resumeRequested && !resumeIssues.issueDetected();
        String status = status(resumeRequested, resumeIssues.issueDetected());
        boolean stateUsable = !resumeIssues.blockingIssueDetected();
        String recommendedMode = recommendedMode(resumeRequested, resumeIssues);
        String nextAction = nextAction(resumeRequested, resumeIssues);
        return new Snapshot(
                complete,
                status,
                stateUsable,
                recommendedMode,
                nextAction,
                metadata(
                        resumeRequested,
                        complete,
                        status,
                        stateUsable,
                        recommendedMode,
                        nextAction,
                        resumeIssues));
    }

    private static String status(boolean resumeRequested, boolean issueDetected) {
        if (!resumeRequested) {
            return "not-requested";
        }
        return issueDetected ? "partial" : "complete";
    }

    private static String recommendedMode(
            boolean resumeRequested,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        if (!resumeRequested) {
            return "not-requested";
        }
        if (resumeIssues.blockingIssueDetected()) {
            return "fresh-start";
        }
        return resumeIssues.issueDetected() ? "partial-resume" : "complete-resume";
    }

    private static String nextAction(
            boolean resumeRequested,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        if (!resumeRequested) {
            return "run without checkpoint resume";
        }
        if (resumeIssues.blockingIssueDetected()) {
            return "start a fresh run or choose a compatible checkpoint before resuming";
        }
        if (resumeIssues.issueDetected()) {
            return "continue with partial resume or restore missing checkpoint artifacts";
        }
        return "continue with restored checkpoint state";
    }

    private static Map<String, Object> metadata(
            boolean resumeRequested,
            boolean complete,
            String status,
            boolean stateUsable,
            String recommendedMode,
            String nextAction,
            TrainerCheckpointResumeIssueMetadata.Snapshot resumeIssues) {
        Map<String, Object> plan = new LinkedHashMap<>();
        plan.put("version", VERSION);
        plan.put("requested", resumeRequested);
        plan.put("status", status);
        plan.put("complete", complete);
        plan.put("stateUsable", stateUsable);
        plan.put("recommendedMode", recommendedMode);
        plan.put("nextAction", nextAction);
        plan.put("issueDetected", resumeIssues.issueDetected());
        plan.put("issueCount", resumeIssues.issueCount());
        plan.put("issueKinds", resumeIssues.issueKinds());
        plan.put("affectedArtifacts", resumeIssues.affectedArtifacts());
        plan.put("blockingIssueDetected", resumeIssues.blockingIssueDetected());
        plan.put("blockingIssueCount", resumeIssues.blockingIssueCount());
        plan.put("blockingIssueKinds", resumeIssues.blockingIssueKinds());
        plan.put("primaryIssueAvailable", resumeIssues.primaryIssueAvailable());
        if (resumeIssues.primaryIssueAvailable()) {
            plan.put("primaryIssueKind", resumeIssues.primaryIssueKind());
            plan.put("primaryIssueCode", resumeIssues.primaryIssueCode());
            plan.put("primaryIssueSeverity", resumeIssues.primaryIssueSeverity());
            plan.put("primaryIssueBlocking", resumeIssues.primaryIssueBlocking());
            plan.put("primaryAffectedArtifact", resumeIssues.primaryAffectedArtifact());
            plan.put("primaryIssueMessage", resumeIssues.primaryIssueMessage());
            plan.put("primaryRecommendedAction", resumeIssues.primaryRecommendedAction());
        }
        plan.put("primaryBlockingIssueAvailable", resumeIssues.primaryBlockingIssueAvailable());
        if (resumeIssues.primaryBlockingIssueAvailable()) {
            plan.put("primaryBlockingIssueKind", resumeIssues.primaryBlockingIssueKind());
            plan.put("primaryBlockingIssueCode", resumeIssues.primaryBlockingIssueCode());
            plan.put("primaryBlockingIssueSeverity", resumeIssues.primaryBlockingIssueSeverity());
            plan.put("primaryBlockingAffectedArtifact", resumeIssues.primaryBlockingAffectedArtifact());
            plan.put("primaryBlockingIssueMessage", resumeIssues.primaryBlockingIssueMessage());
            plan.put("primaryBlockingRecommendedAction", resumeIssues.primaryBlockingRecommendedAction());
        }
        return Collections.unmodifiableMap(plan);
    }

    record Snapshot(
            boolean complete,
            String status,
            boolean stateUsable,
            String recommendedMode,
            String nextAction,
            Map<String, Object> metadata) {
    }
}
