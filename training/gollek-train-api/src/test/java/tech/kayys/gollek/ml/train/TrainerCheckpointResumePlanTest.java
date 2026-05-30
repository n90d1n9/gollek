package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerCheckpointResumePlanTest {

    @Test
    void snapshotRecommendsCompleteResumeWhenRequestedWithoutIssues() {
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(true, issues(List.of(), List.of(), List.of()));

        assertTrue(plan.complete());
        assertEquals("complete", plan.status());
        assertTrue(plan.stateUsable());
        assertEquals("complete-resume", plan.recommendedMode());
        assertEquals("continue with restored checkpoint state", plan.nextAction());
        assertEquals(Boolean.FALSE, plan.metadata().get("issueDetected"));
        assertEquals(Boolean.FALSE, plan.metadata().get("primaryIssueAvailable"));
        assertFalse(plan.metadata().containsKey("primaryIssueKind"));
    }

    @Test
    void snapshotRecommendsPartialResumeForNonBlockingIssues() {
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(true, issues(List.of("optimizer"), List.of(), List.of("runtime")));

        assertFalse(plan.complete());
        assertEquals("partial", plan.status());
        assertTrue(plan.stateUsable());
        assertEquals("partial-resume", plan.recommendedMode());
        assertEquals("continue with partial resume or restore missing checkpoint artifacts", plan.nextAction());
        assertEquals(2, plan.metadata().get("issueCount"));
        assertEquals(List.of("missing-artifact", "manifest-entry-missing"), plan.metadata().get("issueKinds"));
        assertEquals(Boolean.FALSE, plan.metadata().get("blockingIssueDetected"));
        assertEquals(Boolean.FALSE, plan.metadata().get("primaryBlockingIssueAvailable"));
    }

    @Test
    void snapshotRecommendsFreshStartForBlockingIssues() {
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(true, issues(List.of(), List.of("model: signature mismatch"), List.of()));

        assertFalse(plan.complete());
        assertEquals("partial", plan.status());
        assertFalse(plan.stateUsable());
        assertEquals("fresh-start", plan.recommendedMode());
        assertEquals(
                "start a fresh run or choose a compatible checkpoint before resuming",
                plan.nextAction());
        assertEquals(Boolean.TRUE, plan.metadata().get("blockingIssueDetected"));
        assertEquals("compatibility-mismatch", plan.metadata().get("primaryBlockingIssueKind"));
        assertEquals("model", plan.metadata().get("primaryBlockingAffectedArtifact"));
    }

    @Test
    void snapshotPreservesIssueContextWhenResumeWasNotRequested() {
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(false, issues(List.of("optimizer"), List.of(), List.of()));

        assertFalse(plan.complete());
        assertEquals("not-requested", plan.status());
        assertTrue(plan.stateUsable());
        assertEquals("not-requested", plan.recommendedMode());
        assertEquals("run without checkpoint resume", plan.nextAction());
        assertEquals(Boolean.FALSE, plan.metadata().get("requested"));
        assertEquals(Boolean.TRUE, plan.metadata().get("issueDetected"));
        assertEquals(List.of("optimizer"), plan.metadata().get("affectedArtifacts"));
    }

    @Test
    void snapshotMetadataIsImmutable() {
        TrainerCheckpointResumePlan.Snapshot plan =
                TrainerCheckpointResumePlan.snapshot(true, issues(List.of("optimizer"), List.of(), List.of()));

        assertThrows(UnsupportedOperationException.class, () -> plan.metadata().clear());
        assertThrows(UnsupportedOperationException.class, () -> issueKinds(plan).clear());
    }

    private static TrainerCheckpointResumeIssueMetadata.Snapshot issues(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        return TrainerCheckpointResumeIssueMetadata.snapshot(
                missingArtifacts,
                compatibilityMismatches,
                manifestEntryMissingArtifacts);
    }

    @SuppressWarnings("unchecked")
    private static List<String> issueKinds(TrainerCheckpointResumePlan.Snapshot plan) {
        return (List<String>) plan.metadata().get("issueKinds");
    }
}
