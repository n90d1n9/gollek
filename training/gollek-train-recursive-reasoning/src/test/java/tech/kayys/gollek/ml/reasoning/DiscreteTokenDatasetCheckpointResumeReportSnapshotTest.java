package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumeReportSnapshotTest {
    @Test
    void rehydratesReadyResumeReportSnapshot() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetCheckpointResumeReport report = policy.evaluate(manifest, plan);

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertEquals("ready", snapshot.status());
        assertTrue(snapshot.policyTracked());
        assertTrue(snapshot.currentPlanChecked());
        assertEquals("run-001", snapshot.runId());
        assertEquals(12L, snapshot.checkpointStep());
        assertEquals(report.summary(), snapshot.summary());
        assertEquals(report.toMetadata(), snapshot.toMetadata());
        assertEquals(policy.toMetadata(), snapshot.policyMetadata());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT, snapshot.compatibilityMode());
        assertFalse(snapshot.forceAccepted());
        assertEquals(5, snapshot.gates().size());
        assertEquals("accepted", snapshot.gateSummary().status());
        assertEquals(5, snapshot.gateSummary().acceptedCount());
        assertEquals(0, snapshot.gateSummary().blockedCount());
        assertFalse(snapshot.gateSummary().attentionRequired());
        assertFalse(snapshot.gateSummary().actionRequired());
        assertEquals("info", snapshot.gateSummary().highestSeverity());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA,
                snapshot.gateSummary().primaryGateId().orElseThrow());
        assertEquals("continue", snapshot.gateSummary().primaryActionCode().orElseThrow());
        assertEquals("Ready", snapshot.gateSummary().readinessBadge().label());
        assertEquals("success", snapshot.gateSummary().readinessBadge().tone());
        assertEquals("continue", snapshot.gateSummary().readinessBadge().actionCode());
        assertEquals(
                "Continue training from the selected checkpoint.",
                snapshot.gateSummary().readinessBadge().actionHint());
        assertEquals(List.of("continue"), snapshot.gateSummary().actionCodes());
        assertTrue(snapshot.gateSummary().requiredActionCodes().isEmpty());
        assertTrue(snapshot.gateSummary().warningActionCodes().isEmpty());
        assertEquals("continue", snapshot.gateSummary().primaryAction().orElseThrow().code());
        assertTrue(snapshot.gateSummary().actionPlan().ready());
        assertTrue(snapshot.gateSummary().actionPlan().readyWithoutWarnings());
        assertEquals("continue", snapshot.gateSummary().actionPlan().primaryActionCode());
        assertEquals(snapshot.gateSummary().nextActions(), snapshot.gateSummary().actionPlan().actions());
        assertEquals(snapshot.gateSummary().actionPlan(), snapshot.actionPlan());
        assertEquals(snapshot.gateSummary().readinessBadge(), snapshot.readinessBadge());
        assertEquals(snapshot.gateSummary().nextActions(), snapshot.nextActions());
        assertEquals(snapshot.gateSummary().primaryAction().orElseThrow(), snapshot.primaryAction().orElseThrow());
        assertEquals(snapshot.gateSummary().requiredActions(), snapshot.requiredActions());
        assertEquals(snapshot.gateSummary().warningActions(), snapshot.warningActions());
        assertEquals(snapshot.gateSummary().actionCodes(), snapshot.actionCodes());
        assertEquals(snapshot.gateSummary().requiredActionCodes(), snapshot.requiredActionCodes());
        assertEquals(snapshot.gateSummary().warningActionCodes(), snapshot.warningActionCodes());
        snapshot.requireNoRequiredActions();
        assertEquals(
                snapshot.gateSummary().gateIds(),
                snapshot.gateSummary().gateIdsByStatus().get("accepted"));
        assertEquals(
                snapshot.gateSummary().gateIds(),
                snapshot.gateSummary().gateIdsBySeverity().get("info"));
        assertEquals(Map.of("accepted", 5), snapshot.gateSummary().gateCountByStatus());
        assertEquals(Map.of("info", 5), snapshot.gateSummary().gateCountBySeverity());
        assertEquals(5, snapshot.gateSummary().gateCategorySummaries().size());
        assertEquals(snapshot.gateSummary().categoryIds(), snapshot.gateSummary().acceptedCategories());
        assertTrue(snapshot.gateSummary().blockedCategories().isEmpty());
        assertTrue(snapshot.gateSummary().warningCategories().isEmpty());
        assertEquals(
                "accepted",
                snapshot.gateSummary()
                        .requireGateCategorySummary("dataset-fingerprint")
                        .status());
        assertTrue(snapshot.blockedGates().isEmpty());
        assertTrue(snapshot.warningGates().isEmpty());
        assertTrue(snapshot.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertEquals(
                snapshot.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                snapshot.gate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).orElseThrow());
        assertEquals(report.currentPlanReport().toMetadata(), snapshot.currentPlanReportMetadata());
        snapshot.requireAllGatesAccepted();
        snapshot.requireReady();
    }

    @Test
    void rehydratesBlockedResumeReportSnapshot() {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changed);

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(report.toMetadata());

        assertFalse(snapshot.ready());
        assertEquals("blocked", snapshot.status());
        assertFalse(snapshot.fingerprintMatched());
        assertFalse(snapshot.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertEquals("blocked", snapshot.gateSummary().status());
        assertEquals(1, snapshot.gateSummary().blockedCount());
        assertEquals(1, snapshot.blockedGates().size());
        assertTrue(snapshot.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("dataset fingerprint mismatched")));
        assertThrows(
                IllegalStateException.class,
                () -> snapshot.requireGateAccepted(
                        DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertThrows(IllegalStateException.class, snapshot::requireAllGatesAccepted);
        assertThrows(IllegalStateException.class, snapshot::requireReady);
    }

    @Test
    void rehydratesForceAcceptedResumeReportSnapshot() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.force().evaluate(manifest, changedPlan());

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertTrue(snapshot.forceAccepted());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE, snapshot.compatibilityMode());
        assertFalse(snapshot.fingerprintMatched());
        assertTrue(snapshot.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT));
        assertEquals("warning", snapshot.gateSummary().status());
        assertTrue(snapshot.gateSummary().allAccepted());
        assertTrue(snapshot.gateSummary().attentionRequired());
        assertFalse(snapshot.gateSummary().actionRequired());
        assertEquals("warning", snapshot.gateSummary().highestSeverity());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                snapshot.gateSummary().primaryGateId().orElseThrow());
        assertEquals("dataset-fingerprint", snapshot.gateSummary().primaryGateCategory().orElseThrow());
        assertEquals("warning", snapshot.gateSummary().primaryGateStatus().orElseThrow());
        assertEquals("warning", snapshot.gateSummary().primaryGateSeverity().orElseThrow());
        assertEquals(
                "review-forced-dataset-fingerprint",
                snapshot.gateSummary().primaryActionCode().orElseThrow());
        assertTrue(snapshot.gateSummary().primaryActionHint().orElseThrow().contains("forced dataset fingerprint"));
        assertEquals(
                List.of("review-forced-dataset-fingerprint"),
                snapshot.gateSummary().actionCodes());
        assertTrue(snapshot.gateSummary().requiredActionCodes().isEmpty());
        assertEquals(
                List.of("review-forced-dataset-fingerprint"),
                snapshot.gateSummary().warningActionCodes());
        assertEquals(1, snapshot.gateSummary().nextActions().size());
        assertTrue(snapshot.gateSummary().primaryAction().orElseThrow().warning());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                snapshot.gateSummary().primaryAction().orElseThrow().gateId());
        DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan = snapshot.gateSummary().actionPlan();
        assertTrue(actionPlan.ready());
        assertFalse(actionPlan.readyWithoutWarnings());
        assertTrue(actionPlan.attentionRequired());
        assertFalse(actionPlan.actionRequired());
        assertEquals(1, actionPlan.warningActionCount());
        assertEquals("review-forced-dataset-fingerprint", actionPlan.primaryActionCode());
        assertEquals(
                "resume action plan accepted with warning(s): review-forced-dataset-fingerprint",
                actionPlan.summary());
        actionPlan.requireNoRequiredActions();
        assertEquals(actionPlan, snapshot.actionPlan());
        snapshot.requireNoRequiredActions();
        DiscreteTokenDatasetCheckpointResumeReadinessBadge badge = snapshot.gateSummary().readinessBadge();
        assertEquals(badge, DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(badge.toMetadata()));
        assertFalse(badge.ready());
        assertFalse(badge.blocked());
        assertTrue(badge.warning());
        assertEquals("Warning", badge.label());
        assertEquals("warning", badge.tone());
        assertEquals("alert-triangle", badge.iconKey());
        assertEquals("review-forced-dataset-fingerprint", badge.actionCode());
        assertTrue(badge.actionHint().contains("forced dataset fingerprint"));
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                badge.primaryGateId());
        assertEquals(1, snapshot.gateSummary().warningCount());
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                snapshot.gateSummary().gateIdsByStatus().get("warning"));
        assertEquals(
                List.of(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT),
                snapshot.gateSummary().gateIdsBySeverity().get("warning"));
        assertEquals(1, snapshot.gateSummary().gateCountByStatus().get("warning"));
        assertEquals(1, snapshot.gateSummary().gateCountBySeverity().get("warning"));
        assertEquals(List.of("dataset-fingerprint"), snapshot.gateSummary().warningCategories());
        assertEquals("dataset-fingerprint", snapshot.gateSummary().primaryWarningCategory().orElseThrow());
        assertEquals(
                "warning",
                snapshot.gateSummary()
                        .requireGateCategorySummary("dataset-fingerprint")
                        .status());
        assertEquals(
                "review-forced-dataset-fingerprint",
                snapshot.gateSummary()
                        .requireGateCategorySummary("dataset-fingerprint")
                        .primaryActionCode()
                        .orElseThrow());
        assertEquals(
                "warning",
                snapshot.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).severity());
        assertEquals(
                "review-forced-dataset-fingerprint",
                snapshot.requireGate(DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT).actionCode());
        assertEquals(1, snapshot.warningGates().size());
        assertTrue(snapshot.compatibilityWarnings().stream()
                .anyMatch(warning -> warning.contains("fingerprint mismatch")));
        snapshot.requireReady();
    }

    @Test
    void rehydratesReportWithoutCurrentPlanCheck() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);
        DiscreteTokenDatasetCheckpointResumeReport report =
                new DiscreteTokenDatasetCheckpointResumeReport(
                        checkpoint,
                        checkpoint.verifyReport(manifest.datasetPlanReport()));

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertFalse(snapshot.currentPlanChecked());
        assertTrue(snapshot.currentPlanAccepted());
        assertEquals("not-checked", snapshot.currentPlanGateStatus());
        assertTrue(snapshot.gateAccepted(DiscreteTokenDatasetCheckpointResumeGate.CURRENT_PLAN));
        assertTrue(snapshot.requireGate(DiscreteTokenDatasetCheckpointResumeGate.CURRENT_PLAN)
                .summary()
                .contains("not checked"));
        assertFalse(snapshot.toMetadata().containsKey("currentPlanReport"));
    }

    @Test
    void exposesImmutableViews() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertThrows(UnsupportedOperationException.class, () -> snapshot.policyMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> snapshot.currentPlanReportMetadata().put("bad", "value"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.rejectionReasons().add("bad"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.compatibilityWarnings().add("bad"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.toMetadata().put("bad", "value"));
    }

    @Test
    void rejectsMalformedMetadata() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());
        Map<String, Object> metadata = report.toMetadata();

        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "status", "blocked")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "policyTracked", false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        removing(metadata, "currentPlanReport")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "rejectionReasons", List.of(7))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeAction.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromMetadata(Map.of(), report.gates()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(
                        replacing(report.actionPlan().toMetadata(), "actionCount", 2)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(
                        replacing(report.actionPlan().toMetadata(), "actionCodes", List.of("repair-everything"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(
                        replacing(report.actionPlan().toMetadata(), "primaryActionCode", "repair-everything")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(
                        replacing(report.readinessBadge().toMetadata(), "label", "Warning")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromMetadata(
                        replacing(
                                report.gateSummary()
                                        .requireGateCategorySummary("manifest")
                                        .toMetadata(),
                                "acceptedCount",
                                0),
                        report.gates()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromMetadata(
                        replacing(
                                report.gateSummary()
                                        .requireGateCategorySummary("manifest")
                                        .toMetadata(),
                                "category",
                                "missing-category"),
                        report.gates()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "status", "blocked")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "acceptedCount", 4)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "gateIdsByStatus",
                                Map.of(
                                        "accepted",
                                        List.of(DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA)))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "gateCountByStatus",
                                Map.of("accepted", 4))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "categoryIds",
                                List.of("dataset-fingerprint"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "highestSeverity", "warning")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "attentionRequired", true)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "primaryActionCode", "repair-everything")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "actionCodes", List.of("repair-everything"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "nextActions", List.of(Map.of("code", "bad")))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(metadata, "gateSummary", "actionPlan", Map.of("status", "blocked"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "actionPlan", Map.of("status", "blocked"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "readinessBadge", Map.of("status", "blocked"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "primaryGateId",
                                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "readinessBadge",
                                Map.of(
                                        "status",
                                        "accepted",
                                        "label",
                                        "Warning"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "gateCategorySummaries",
                                List.of(Map.of(
                                        "category",
                                        "dataset-fingerprint",
                                        "status",
                                        "accepted")))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "primaryBlockedCategory",
                                "dataset-fingerprint")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacingNestedMap(
                                metadata,
                                "gateSummary",
                                "primaryBlockedGateId",
                                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT)));
    }

    @Test
    void synthesizesGateSummaryForOlderResumeReports() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());
        Map<String, Object> metadata = report.toMetadata();

        DiscreteTokenDatasetCheckpointResumeReportSnapshot noSummary =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        removing(metadata, "gateSummary"));
        DiscreteTokenDatasetCheckpointResumeReportSnapshot noGatesOrSummary =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        removing(removing(metadata, "gateSummary"), "gates"));

        assertEquals("accepted", noSummary.gateSummary().status());
        assertEquals(5, noSummary.gateSummary().acceptedCount());
        assertEquals("accepted", noGatesOrSummary.gateSummary().status());
        assertEquals(5, noGatesOrSummary.gates().size());
    }

    @Test
    void acceptsGateSummaryWithoutDerivedIndexesFromOlderResumeReports() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());
        Map<String, Object> metadata = report.toMetadata();
        Map<String, Object> oldSummaryMetadata = metadata;
        for (String field : List.of(
                "gateIdsByStatus",
                "gateIdsBySeverity",
                "gateIdsByCategory",
                "gateCountByStatus",
                "gateCountBySeverity",
                "gateCountByCategory",
                "gateCategorySummaries",
                "categoryIds",
                "acceptedCategories",
                "blockedCategories",
                "warningCategories",
                "primaryBlockedCategory",
                "primaryWarningCategory",
                "attentionRequired",
                "actionRequired",
                "highestSeverity",
                "primaryGateId",
                "primaryGateCategory",
                "primaryGateStatus",
                "primaryGateSeverity",
                "primaryActionCode",
                "primaryActionHint",
                "nextActions",
                "actionCodes",
                "requiredActionCodes",
                "warningActionCodes",
                "actionPlan",
                "readinessBadge")) {
            oldSummaryMetadata = removingNestedMap(oldSummaryMetadata, "gateSummary", field);
        }

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(oldSummaryMetadata);

        assertEquals("accepted", snapshot.gateSummary().status());
        assertEquals(
                snapshot.gateSummary().gateIds(),
                snapshot.gateSummary().gateIdsByStatus().get("accepted"));
        assertEquals(Map.of("accepted", 5), snapshot.gateSummary().gateCountByStatus());
        assertEquals(5, snapshot.gateSummary().gateCategorySummaries().size());
        assertEquals(snapshot.gateSummary().categoryIds(), snapshot.gateSummary().acceptedCategories());
        assertFalse(snapshot.gateSummary().attentionRequired());
        assertEquals("info", snapshot.gateSummary().highestSeverity());
        assertEquals("Ready", snapshot.gateSummary().readinessBadge().label());
        assertEquals("continue", snapshot.gateSummary().primaryActionCode().orElseThrow());
        assertEquals(List.of("continue"), snapshot.gateSummary().actionCodes());
        assertEquals("continue", snapshot.gateSummary().actionPlan().primaryActionCode());
        assertEquals("continue", snapshot.actionPlan().primaryActionCode());
    }

    private static Map<String, Object> replacing(Map<String, Object> metadata, String key, Object value) {
        Map<String, Object> copy = new LinkedHashMap<>(metadata);
        copy.put(key, value);
        return copy;
    }

    private static Map<String, Object> removing(Map<String, Object> metadata, String key) {
        Map<String, Object> copy = new LinkedHashMap<>(metadata);
        copy.remove(key);
        return copy;
    }

    private static Map<String, Object> replacingNestedMap(
            Map<String, Object> metadata,
            String key,
            String nestedKey,
            Object nestedValue) {
        Object nested = metadata.get(key);
        if (!(nested instanceof Map<?, ?> nestedMap)) {
            throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
        }
        Map<String, Object> nestedCopy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : nestedMap.entrySet()) {
            nestedCopy.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        nestedCopy.put(nestedKey, nestedValue);
        return replacing(metadata, key, nestedCopy);
    }

    private static Map<String, Object> removingNestedMap(
            Map<String, Object> metadata,
            String key,
            String nestedKey) {
        Object nested = metadata.get(key);
        if (!(nested instanceof Map<?, ?> nestedMap)) {
            throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
        }
        Map<String, Object> nestedCopy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : nestedMap.entrySet()) {
            nestedCopy.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        nestedCopy.remove(nestedKey);
        return replacing(metadata, key, nestedCopy);
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

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                1,
                Map.of("inputLength", inputLength));
    }
}
