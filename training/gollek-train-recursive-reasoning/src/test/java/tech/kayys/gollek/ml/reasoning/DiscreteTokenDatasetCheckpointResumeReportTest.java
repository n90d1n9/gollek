package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumeReportTest {
    @Test
    void acceptsMatchingCurrentCheckpointAndPlan() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReport planReport = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetPlanReport defaultResumeReport = plan.report();
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(planReport).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);

        DiscreteTokenDatasetCheckpointResumeReport resumeReport = snapshot.resumeReport(plan);

        assertTrue(resumeReport.ready());
        assertEquals("ready", resumeReport.status());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanChecked());
        assertFalse(resumeReport.policyTracked());
        assertTrue(resumeReport.currentPlanAccepted());
        assertEquals("accepted", resumeReport.currentPlanGateStatus());
        assertEquals(5, resumeReport.gates().size());
        assertEquals(5, resumeReport.gatesById().size());
        assertTrue(resumeReport.blockedGates().isEmpty());
        assertTrue(resumeReport.warningGates().isEmpty());
        assertEquals("accepted", resumeReport.gateSummary().status());
        assertTrue(resumeReport.gateSummary().allAccepted());
        assertFalse(resumeReport.gateSummary().attentionRequired());
        assertFalse(resumeReport.gateSummary().actionRequired());
        assertEquals("info", resumeReport.gateSummary().highestSeverity());
        assertEquals(5, resumeReport.gateSummary().acceptedCount());
        assertEquals(0, resumeReport.gateSummary().blockedCount());
        assertTrue(resumeReport.gateSummary().primaryBlockedGate().isEmpty());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA,
                resumeReport.gateSummary().primaryGateId().orElseThrow());
        assertEquals("manifest", resumeReport.gateSummary().primaryGateCategory().orElseThrow());
        assertEquals("accepted", resumeReport.gateSummary().primaryGateStatus().orElseThrow());
        assertEquals("info", resumeReport.gateSummary().primaryGateSeverity().orElseThrow());
        assertEquals("continue", resumeReport.gateSummary().primaryActionCode().orElseThrow());
        assertEquals(
                "Continue training from the selected checkpoint.",
                resumeReport.gateSummary().primaryActionHint().orElseThrow());
        assertEquals(List.of("continue"), resumeReport.gateSummary().actionCodes());
        assertTrue(resumeReport.gateSummary().requiredActionCodes().isEmpty());
        assertTrue(resumeReport.gateSummary().warningActionCodes().isEmpty());
        assertEquals(1, resumeReport.gateSummary().nextActions().size());
        assertEquals("continue", resumeReport.gateSummary().primaryAction().orElseThrow().code());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA,
                resumeReport.gateSummary().primaryAction().orElseThrow().gateId());
        DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan = resumeReport.gateSummary().actionPlan();
        assertTrue(actionPlan.ready());
        assertTrue(actionPlan.readyWithoutWarnings());
        assertFalse(actionPlan.attentionRequired());
        assertFalse(actionPlan.actionRequired());
        assertEquals(1, actionPlan.actionCount());
        assertEquals(0, actionPlan.requiredActionCount());
        assertEquals(0, actionPlan.warningActionCount());
        assertEquals("continue", actionPlan.primaryActionCode());
        assertEquals(DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA, actionPlan.primaryGateId());
        assertEquals(resumeReport.gateSummary().actionCodes(), actionPlan.actionCodes());
        assertEquals(resumeReport.gateSummary().nextActions(), actionPlan.actions());
        assertEquals("resume action plan accepted: continue", actionPlan.summary());
        actionPlan.requireNoRequiredActions();
        assertEquals(actionPlan, DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(actionPlan.toMetadata()));
        assertEquals(
                actionPlan.primaryAction(),
                DiscreteTokenDatasetCheckpointResumeAction.fromMetadata(actionPlan.primaryAction().toMetadata()));
        assertEquals(actionPlan, resumeReport.actionPlan());
        assertEquals(actionPlan.actions(), resumeReport.nextActions());
        assertEquals(actionPlan.primaryAction(), resumeReport.primaryAction().orElseThrow());
        assertEquals(actionPlan.requiredActions(), resumeReport.requiredActions());
        assertEquals(actionPlan.warningActions(), resumeReport.warningActions());
        assertEquals(actionPlan.actionCodes(), resumeReport.actionCodes());
        assertEquals(actionPlan.requiredActionCodes(), resumeReport.requiredActionCodes());
        assertEquals(actionPlan.warningActionCodes(), resumeReport.warningActionCodes());
        resumeReport.requireNoRequiredActions();
        DiscreteTokenDatasetCheckpointResumeReadinessBadge badge = resumeReport.gateSummary().readinessBadge();
        assertEquals(badge, resumeReport.readinessBadge());
        assertEquals(badge, DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(badge.toMetadata()));
        assertTrue(badge.ready());
        assertFalse(badge.blocked());
        assertFalse(badge.warning());
        assertEquals("Ready", badge.label());
        assertEquals("success", badge.tone());
        assertEquals("check-circle", badge.iconKey());
        assertEquals("continue", badge.actionCode());
        assertEquals("Continue training from the selected checkpoint.", badge.actionHint());
        assertEquals(DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA, badge.primaryGateId());
        assertEquals(
                List.of(
                        DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA,
                        DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_DATASET,
                        DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                        DiscreteTokenDatasetCheckpointResumeGate.RESUME_EXPECTATION,
                        DiscreteTokenDatasetCheckpointResumeGate.CURRENT_PLAN),
                resumeReport.gateSummary().gateIdsByStatus().get("accepted"));
        assertEquals(
                resumeReport.gateSummary().gateIds(),
                resumeReport.gateSummary().gateIdsBySeverity().get("info"));
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.gateSummary().gateIdsByCategory().get("dataset-fingerprint"));
        assertEquals(Map.of("accepted", 5), resumeReport.gateSummary().gateCountByStatus());
        assertEquals(Map.of("info", 5), resumeReport.gateSummary().gateCountBySeverity());
        assertEquals(1, resumeReport.gateSummary().gateCountByCategory().get("dataset-fingerprint"));
        assertEquals(
                List.of(
                        "manifest",
                        "checkpoint-dataset",
                        "dataset-fingerprint",
                        "resume-expectation",
                        "current-dataset"),
                resumeReport.gateSummary().categoryIds());
        assertEquals(resumeReport.gateSummary().categoryIds(), resumeReport.gateSummary().acceptedCategories());
        assertTrue(resumeReport.gateSummary().blockedCategories().isEmpty());
        assertTrue(resumeReport.gateSummary().warningCategories().isEmpty());
        assertTrue(resumeReport.gateSummary().primaryBlockedCategory().isEmpty());
        assertTrue(resumeReport.gateSummary().primaryWarningCategory().isEmpty());
        assertEquals(5, resumeReport.gateSummary().gateCategorySummaries().size());
        DiscreteTokenDatasetCheckpointResumeGateCategorySummary fingerprintCategory =
                resumeReport.gateSummary().requireGateCategorySummary("dataset-fingerprint");
        assertEquals(fingerprintCategory, fingerprintCategory.requireMetadataMatch(fingerprintCategory.toMetadata()));
        assertEquals(
                fingerprintCategory,
                DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromMetadata(
                        fingerprintCategory.toMetadata(),
                        resumeReport.gates()));
        assertEquals(
                "accepted",
                fingerprintCategory.status());
        assertEquals(
                "info",
                fingerprintCategory.severity());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                fingerprintCategory.primaryGate().orElseThrow().id());
        assertEquals(
                "continue",
                fingerprintCategory.primaryActionCode().orElseThrow());
        assertEquals(5, resumeReport.gateSummary().gatesByStatus().get("accepted").size());
        resumeReport.requireAllGatesAccepted();
        assertTrue(resumeReport.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertEquals(
                resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.gate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).orElseThrow());
        assertEquals(
                "accepted",
                resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).status());
        assertFalse(resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                .attentionRequired());
        assertFalse(resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                .actionRequired());
        assertEquals(
                "continue",
                resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                        .actionCode());
        resumeReport.requireGateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT);
        assertTrue(resumeReport.rejectionReasons().isEmpty());
        assertEquals(
                "checkpoint resume ready: run-001 step 12 dataset " + planReport.fingerprint().shortValue(),
                resumeReport.summary());
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("ready", explanation.status());
        assertEquals("resume-ready", explanation.primaryCode());
        assertEquals("ready", explanation.primaryCategory());
        assertEquals(1, explanation.findings().size());
        assertEquals("resume-ready", explanation.toMetadata().get("primaryCode"));
        assertEquals(explanation, DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(explanation.toMetadata()));
        assertEquals(explanation, explanation.requireMetadataMatch(explanation.toMetadata()));
        assertEquals(true, resumeReport.toMetadata().get("ready"));
        assertEquals(true, resumeReport.toMetadata().get("expectationAccepted"));
        assertEquals(true, resumeReport.toMetadata().get("currentPlanChecked"));
        assertEquals(true, resumeReport.toMetadata().get("currentPlanAccepted"));
        assertEquals("accepted", resumeReport.toMetadata().get("currentPlanGateStatus"));
        assertEquals(
                "accepted",
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("status"));
        assertEquals(
                resumeReport.gateSummary().gateIdsByStatus(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("gateIdsByStatus"));
        assertEquals(
                resumeReport.gateSummary().gateCountByStatus(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("gateCountByStatus"));
        assertEquals(
                resumeReport.gateSummary().categoryIds(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("categoryIds"));
        assertEquals(
                false,
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("attentionRequired"));
        assertEquals(
                "info",
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("highestSeverity"));
        assertEquals(
                resumeReport.gateSummary().primaryGateId().orElseThrow(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("primaryGateId"));
        assertEquals(
                resumeReport.gateSummary().primaryActionCode().orElseThrow(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("primaryActionCode"));
        assertEquals(
                resumeReport.gateSummary().actionCodes(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("actionCodes"));
        assertEquals(
                resumeReport.gateSummary().nextActions().stream()
                        .map(DiscreteTokenDatasetCheckpointResumeAction::toMetadata)
                        .toList(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("nextActions"));
        assertEquals(
                actionPlan.toMetadata(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("actionPlan"));
        assertEquals(actionPlan.toMetadata(), resumeReport.toMetadata().get("actionPlan"));
        assertEquals(
                badge.toMetadata(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("readinessBadge"));
        assertEquals(badge.toMetadata(), resumeReport.toMetadata().get("readinessBadge"));
        assertEquals(
                resumeReport.gateSummary().gateCategorySummaries().stream()
                        .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::toMetadata)
                        .toList(),
                ((Map<?, ?>) resumeReport.toMetadata().get("gateSummary")).get("gateCategorySummaries"));
        assertEquals(5, ((List<?>) resumeReport.toMetadata().get("gates")).size());
        assertEquals(false, resumeReport.toMetadata().get("policyTracked"));
        assertEquals("matched", ((Map<?, ?>) resumeReport.toMetadata().get("fingerprintMatch")).get("status"));
        assertEquals(false, ((Map<?, ?>) resumeReport.toMetadata().get("expectation")).get("active"));
        assertEquals(
                defaultResumeReport.toMetadata(),
                ((Map<?, ?>) resumeReport.toMetadata().get("currentPlanReport")));
        assertEquals(manifest.toMetadata(), ((Map<?, ?>) resumeReport.toMetadata().get("checkpoint")));
        resumeReport.requireReady();
    }

    @Test
    void acceptsMatchingCurrentCheckpointAndReport() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(report).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), report);

        assertTrue(resumeReport.ready());
        assertTrue(resumeReport.fingerprintMatched());
    }

    @Test
    void blocksWhenCurrentReportFailsEvenIfCheckpointWasAccepted() {
        DiscreteTokenDatasetPlan warningPlan = warningHeavyPlan();
        DiscreteTokenDatasetPlanReport checkpointReport =
                warningPlan.report(DiscreteTokenDatasetPlanReadinessGate.training());
        DiscreteTokenDatasetPlanReport currentStrictReport =
                warningPlan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(checkpointReport).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        currentStrictReport);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanChecked());
        assertFalse(resumeReport.currentPlanAccepted());
        assertEquals("warning-blocked", resumeReport.currentPlanGateStatus());
        assertTrue(resumeReport.rejectionReasons()
                .contains("current dataset plan was not accepted: warning-blocked"));
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("blocked", explanation.status());
        assertEquals("current-plan-rejected", explanation.primaryCode());
        assertEquals("current-dataset", explanation.primaryCategory());
        assertTrue(explanation.nextSteps().get(0).contains("Fix the current dataset readiness issues"));
        assertEquals(explanation, DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(explanation.toMetadata()));
        assertEquals("warning-blocked", resumeReport.toMetadata().get("currentPlanGateStatus"));
        assertEquals(currentStrictReport.toMetadata(), resumeReport.toMetadata().get("currentPlanReport"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void acceptsMatchingResumeExpectation() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("gram-nqueens")
                        .runId("run-001")
                        .modelFamily("gram")
                        .seed(2026L)
                        .minimumCheckpointStep(12L)
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        plan,
                        expectation);

        assertTrue(resumeReport.ready());
        assertTrue(resumeReport.expectationAccepted());
        assertEquals(expectation.toMetadata(), resumeReport.expectation().toMetadata());
        resumeReport.requireReady();
    }

    @Test
    void blocksWhenResumeExpectationDiffers() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("other-experiment")
                        .runId("run-999")
                        .modelFamily("other-family")
                        .seed(7L)
                        .minimumCheckpointStep(20L)
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        plan,
                        expectation);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertFalse(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("experimentName expected other-experiment")));
        assertTrue(resumeReport.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("checkpointStep expected >= 20")));
        assertEquals(false, resumeReport.toMetadata().get("expectationAccepted"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void blocksWhenCheckpointSchemaIsNotCurrent() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()))
                        .schemaVersion("gollek.discrete-token-checkpoint-manifest.v2")
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), plan);

        assertFalse(resumeReport.ready());
        assertFalse(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().get(0).contains("schema"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void blocksWhenCheckpointDatasetWasRejected() {
        DiscreteTokenDatasetPlanReport report =
                warningHeavyPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(report).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), report);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertFalse(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertFalse(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().contains("checkpoint dataset was not accepted: warning-blocked"));
        assertTrue(resumeReport.rejectionReasons()
                .contains("current dataset plan was not accepted: warning-blocked"));
        assertTrue(resumeReport.summary().contains("warning-blocked"));
    }

    @Test
    void blocksWhenCurrentPlanFingerprintDiffers() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), changedPlan());

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertFalse(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.currentPlanAccepted());
        assertEquals("mismatched", resumeReport.fingerprintMatch().status());
        assertFalse(resumeReport.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertEquals("blocked", resumeReport.gateSummary().status());
        assertFalse(resumeReport.gateSummary().allAccepted());
        assertTrue(resumeReport.gateSummary().attentionRequired());
        assertTrue(resumeReport.gateSummary().actionRequired());
        assertEquals("error", resumeReport.gateSummary().highestSeverity());
        assertEquals(1, resumeReport.gateSummary().blockedCount());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                resumeReport.gateSummary().primaryGateId().orElseThrow());
        assertEquals("dataset-fingerprint", resumeReport.gateSummary().primaryGateCategory().orElseThrow());
        assertEquals("blocked", resumeReport.gateSummary().primaryGateStatus().orElseThrow());
        assertEquals("error", resumeReport.gateSummary().primaryGateSeverity().orElseThrow());
        assertEquals(
                "rebuild-or-select-matching-dataset",
                resumeReport.gateSummary().primaryActionCode().orElseThrow());
        assertTrue(resumeReport.gateSummary().primaryActionHint().orElseThrow().contains("dataset fingerprint"));
        assertEquals(
                List.of("rebuild-or-select-matching-dataset"),
                resumeReport.gateSummary().actionCodes());
        assertEquals(
                List.of("rebuild-or-select-matching-dataset"),
                resumeReport.gateSummary().requiredActionCodes());
        assertTrue(resumeReport.gateSummary().warningActionCodes().isEmpty());
        assertEquals(1, resumeReport.gateSummary().nextActions().size());
        assertTrue(resumeReport.gateSummary().primaryAction().orElseThrow().primary());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                resumeReport.gateSummary().primaryAction().orElseThrow().gateId());
        DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan = resumeReport.gateSummary().actionPlan();
        assertFalse(actionPlan.ready());
        assertFalse(actionPlan.readyWithoutWarnings());
        assertTrue(actionPlan.attentionRequired());
        assertTrue(actionPlan.actionRequired());
        assertEquals(1, actionPlan.actionCount());
        assertEquals(1, actionPlan.requiredActionCount());
        assertEquals(0, actionPlan.warningActionCount());
        assertEquals("rebuild-or-select-matching-dataset", actionPlan.primaryActionCode());
        assertEquals(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT, actionPlan.primaryGateId());
        assertEquals("resume action plan blocked: rebuild-or-select-matching-dataset", actionPlan.summary());
        assertEquals(actionPlan, DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(actionPlan.toMetadata()));
        assertThrows(IllegalStateException.class, actionPlan::requireNoRequiredActions);
        assertEquals(actionPlan, resumeReport.actionPlan());
        assertThrows(IllegalStateException.class, resumeReport::requireNoRequiredActions);
        DiscreteTokenDatasetCheckpointResumeReadinessBadge badge = resumeReport.gateSummary().readinessBadge();
        assertEquals(badge, resumeReport.readinessBadge());
        assertEquals(badge, DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(badge.toMetadata()));
        assertFalse(badge.ready());
        assertTrue(badge.blocked());
        assertFalse(badge.warning());
        assertEquals("Blocked", badge.label());
        assertEquals("danger", badge.tone());
        assertEquals("x-circle", badge.iconKey());
        assertEquals("rebuild-or-select-matching-dataset", badge.actionCode());
        assertTrue(badge.actionHint().contains("dataset fingerprint"));
        assertEquals(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT, badge.primaryGateId());
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.gateSummary().gateIdsByStatus().get("blocked"));
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.gateSummary().gateIdsBySeverity().get("error"));
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.gateSummary().gateIdsByCategory().get("dataset-fingerprint"));
        assertEquals(1, resumeReport.gateSummary().gateCountByStatus().get("blocked"));
        assertEquals(1, resumeReport.gateSummary().gateCountBySeverity().get("error"));
        assertEquals(1, resumeReport.gateSummary().gateCountByCategory().get("dataset-fingerprint"));
        assertEquals(List.of("dataset-fingerprint"), resumeReport.gateSummary().blockedCategories());
        assertEquals("dataset-fingerprint", resumeReport.gateSummary().primaryBlockedCategory().orElseThrow());
        DiscreteTokenDatasetCheckpointResumeGateCategorySummary fingerprintCategory =
                resumeReport.gateSummary().requireGateCategorySummary("dataset-fingerprint");
        assertEquals(
                fingerprintCategory,
                DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromMetadata(
                        fingerprintCategory.toMetadata(),
                        resumeReport.gates()));
        assertEquals(
                "blocked",
                fingerprintCategory.status());
        assertEquals(
                "error",
                fingerprintCategory.severity());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                fingerprintCategory.primaryBlockedGate().orElseThrow().id());
        assertEquals(
                "rebuild-or-select-matching-dataset",
                fingerprintCategory.primaryActionCode().orElseThrow());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                resumeReport.gateSummary().primaryBlockedGate().orElseThrow().id());
        assertThrows(IllegalStateException.class, resumeReport::requireAllGatesAccepted);
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                resumeReport.blockedGates().stream()
                        .map(DiscreteTokenDatasetCheckpointResumeGate::id)
                        .toList());
        assertEquals(
                "error",
                resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).severity());
        assertTrue(resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                .attentionRequired());
        assertTrue(resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                .actionRequired());
        assertEquals(
                "rebuild-or-select-matching-dataset",
                resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                        .actionCode());
        assertTrue(resumeReport.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)
                .actionHint()
                .contains("dataset fingerprint"));
        assertThrows(
                IllegalStateException.class,
                () -> resumeReport.requireGateAccepted(
                        DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertTrue(resumeReport.rejectionReasons().get(0).contains("dataset fingerprint mismatched"));
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("dataset-fingerprint-mismatch", explanation.primaryCode());
        assertEquals("dataset-fingerprint", explanation.primaryCategory());
        assertEquals("error", explanation.findings().get(0).severity());
        assertEquals(explanation, DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(explanation.toMetadata()));
        assertTrue(((List<?>) explanation.findings().get(0).details().get("mismatchReasons"))
                .contains("fingerprint value differs"));
        assertTrue(resumeReport.summary().contains("blocked"));
    }

    @Test
    void rejectsMalformedInputs() {
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(
                        manifest(cleanPlan().report()).build());

        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(null, snapshot.verifyPlan(cleanPlan())));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(snapshot, null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(null, cleanPlan()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(null, cleanPlan()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(snapshot, (DiscreteTokenDatasetPlan) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(snapshot, (DiscreteTokenDatasetPlanReport) null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                        cleanPlan().report(),
                        mapWithNullPolicyValue()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                        cleanPlan().report(),
                        mapWithBlankPolicyKey()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeGate.fromMetadata(Map.of(
                        "id", DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                        "category", "dataset-fingerprint",
                        "accepted", true,
                        "severity", "info",
                        "summary", "bad",
                        "status", "blocked")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeGateSummary.fromGates(List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(Map.of()));
        DiscreteTokenDatasetCheckpointResumeExplanation explanation =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                                snapshot.toMetadata(),
                                cleanPlan())
                        .explanation();
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(
                        replacing(explanation.toMetadata(), "primaryCode", "stale-code")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(
                        replacing(explanation.toMetadata(), "findingCount", 99)));
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest(cleanPlan().report()).build().toMetadata(),
                        cleanPlan());

        assertThrows(UnsupportedOperationException.class, () -> resumeReport.toMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> resumeReport.withPolicyMetadata(Map.of("source", "unit-test"))
                        .policyMetadata()
                        .put("bad", "value"));
    }

    private static Map<String, Object> mapWithNullPolicyValue() {
        java.util.HashMap<String, Object> values = new java.util.HashMap<>();
        values.put("bad", null);
        return values;
    }

    private static Map<String, Object> mapWithBlankPolicyKey() {
        java.util.HashMap<String, Object> values = new java.util.HashMap<>();
        values.put(" ", "bad");
        return values;
    }

    private static Map<String, Object> replacing(Map<String, Object> metadata, String key, Object value) {
        Map<String, Object> copy = new LinkedHashMap<>(metadata);
        copy.put(key, value);
        return copy;
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-nqueens")
                .runId("run-001")
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(1_900_000_000_000L)
                .createdBy("trainer-test");
    }

    private static DiscreteTokenDatasetPlan cleanPlan() {
        return plan(0L);
    }

    private static DiscreteTokenDatasetPlan changedPlan() {
        return plan(42L);
    }

    private static DiscreteTokenDatasetPlan plan(long seed) {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("graph-coloring", 0, 1),
                        knownExample("graph-coloring", 1, 2),
                        knownExample("graph-coloring", 2, 1),
                        knownExample("graph-coloring", 3, 2),
                        knownExample("nqueens", 10, 1),
                        knownExample("nqueens", 11, 2),
                        knownExample("nqueens", 12, 1),
                        knownExample("nqueens", 13, 2)),
                new DiscreteTokenDatasetPlanConfig(
                        0.25d,
                        0.25d,
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        seed,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                        0L,
                        false));
    }

    private static DiscreteTokenDatasetPlan warningHeavyPlan() {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        unknownExample(0, 1),
                        unknownExample(1, 8),
                        unknownExample(2, 2)),
                new DiscreteTokenDatasetPlanConfig(
                        0.0d,
                        0.0d,
                        DiscreteTokenDatasetSplitMode.SEQUENTIAL_FRACTIONS,
                        0L,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                        0L,
                        true));
    }

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
        return example(taskId, index, inputLength, 1);
    }

    private static DiscreteTokenDatasetExample unknownExample(int index, int inputLength) {
        return example("task", index, inputLength, -1);
    }

    private static DiscreteTokenDatasetExample example(
            String taskId,
            int index,
            int inputLength,
            int knownSolutionCount) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                knownSolutionCount,
                Map.of("inputLength", inputLength));
    }
}
