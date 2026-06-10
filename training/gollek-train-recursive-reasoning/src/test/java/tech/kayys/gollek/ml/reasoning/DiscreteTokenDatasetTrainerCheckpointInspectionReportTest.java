package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetTrainerCheckpointInspectionReportTest {
    @TempDir
    Path rootDir;

    @Test
    void inspectsSelectionCandidatesAndSelectedCheckpoint() throws IOException {
        writeReadyCheckpoint(rootDir.resolve("ready"), "run-ready", 100L);
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 300L);
        writeBlockedCheckpoint(rootDir.resolve("blocked"), "run-blocked", 400L);

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady();

        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(rootDir, policy);

        assertTrue(report.selectionSatisfied());
        assertTrue(report.selected());
        assertEquals("run-ready", report.requireSelectedCheckpoint().manifest().runId());
        assertEquals(1, report.acceptedCount());
        assertEquals(2, report.rejectedCount());
        assertEquals(3, report.checkpointDecisions().size());
        assertEquals("run-ready", report.acceptedDecisions().get(0).checkpoint().manifest().runId());
        assertTrue(report.rejectedDecisions().stream()
                .flatMap(decision -> decision.rejectionReasons().stream())
                .anyMatch(reason -> reason.contains("resume report is missing")));
        assertTrue(report.summary().contains("selected=run-ready"));

        Map<String, Object> metadata = report.toMetadata();
        assertEquals(true, metadata.get("selectionSatisfied"));
        assertEquals(1, metadata.get("acceptedCount"));
        assertTrue(metadata.containsKey("selectedCheckpoint"));
        assertTrue(metadata.containsKey("lineageGraph"));
        assertEquals(3, ((List<?>) metadata.get("checkpointDecisions")).size());
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void resolvesLineageGraphAndSelectedCheckpointChain() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReport report = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest parent = manifest(report, "run-parent", 100L, 12L)
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("parent"), parent);
        DiscreteTokenDatasetCheckpointManifest child = manifest(report, "run-child", 200L, 18L)
                .lineageFrom(
                        DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(parent),
                        Map.of("source", "resume-test"))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("child"), child);

        DiscreteTokenDatasetTrainerCheckpointInspectionReport inspection =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                        rootDir,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest());

        DiscreteTokenDatasetCheckpointLineageGraph graph = inspection.lineageGraph();
        DiscreteTokenDatasetCheckpointLineageGraph.LineageChain selectedChain =
                inspection.requireSelectedLineageChain();

        assertTrue(graph.healthy());
        assertEquals("healthy", graph.status());
        assertEquals(1.0d, graph.healthScore());
        assertEquals("info", graph.alertLevel());
        assertEquals("No lineage action required.", graph.recommendedAction());
        assertEquals(5, graph.checks().size());
        assertEquals(5, graph.passingCheckCount());
        assertEquals(0, graph.failingCheckCount());
        assertTrue(graph.failingChecks().isEmpty());
        assertTrue(graph.primaryFailingCheck().isEmpty());
        assertTrue(graph.primaryFailingCheckName().isEmpty());
        assertTrue(graph.primaryFailingCheckType().isEmpty());
        assertTrue(graph.primaryFailingCheckCode().isEmpty());
        assertTrue(graph.primaryFailingCheckSeverity().isEmpty());
        assertTrue(graph.primaryFailingCheckAction().isEmpty());
        assertTrue(graph.primaryFailingCheckMessage().isEmpty());
        assertEquals(5, graph.checkSummary().get("total"));
        assertEquals(5, graph.checkSummary().get("passed"));
        assertEquals(0, graph.checkSummary().get("failed"));
        assertEquals(true, graph.checkSummary().get("allPassed"));
        assertEquals("info", graph.checkSummary().get("dominantSeverity"));
        assertEquals("none", graph.checkSummary().get("primaryFailingName"));
        assertEquals("none", graph.checkSummary().get("primaryFailingType"));
        assertEquals("GOLLEK_LINEAGE_HEALTHY", graph.checkSummary().get("primaryFailingCode"));
        assertEquals("info", graph.checkSummary().get("primaryFailingSeverity"));
        assertEquals("No action required.", graph.checkSummary().get("primaryFailingAction"));
        assertFalse(graph.checkSummary().containsKey("primaryFailingCheck"));
        assertEquals(5L, ((Map<?, ?>) graph.checkSummary().get("severityCounts")).get("info"));
        assertEquals(0L, ((Map<?, ?>) graph.checkSummary().get("failingSeverityCounts")).get("critical"));
        graph.requireHealthy();
        assertEquals(2, graph.nodeCount());
        assertEquals(1, graph.roots().size());
        assertTrue(graph.unresolvedNodes().isEmpty());
        assertTrue(graph.cycleRunIds().isEmpty());
        assertEquals("checkpoint-lineage-health", graph.healthMetadata().get("kind"));
        assertEquals("gollek.checkpoint-lineage-health.v1", graph.healthMetadata().get("schemaVersion"));
        DiscreteTokenDatasetCheckpointLineageHealthSnapshot healthySnapshot = graph.healthSnapshot();
        assertEquals(graph.healthMetadata(), healthySnapshot.toMetadata());
        assertEquals("checkpoint-lineage-health", healthySnapshot.kind());
        assertEquals("gollek.checkpoint-lineage-health.v1", healthySnapshot.schemaVersion());
        assertTrue(healthySnapshot.isCurrentSchema());
        healthySnapshot.requireCurrentSchema();
        healthySnapshot.requireHealthy();
        assertFalse(healthySnapshot.hasFailures());
        assertTrue(healthySnapshot.primaryIssueCode().isEmpty());
        assertTrue(healthySnapshot.primaryFailingCheckCode().isEmpty());
        assertEquals(0, graph.healthMetadata().get("issueCount"));
        assertEquals(0, graph.healthMetadata().get("issueDetailCount"));
        assertEquals(0, graph.healthMetadata().get("blockingIssueCount"));
        assertEquals(List.of(), graph.healthMetadata().get("issueCodes"));
        assertEquals(List.of(), graph.healthMetadata().get("issueDetails"));
        assertTrue(graph.primaryIssue().isEmpty());
        assertTrue(graph.primaryIssueCode().isEmpty());
        assertEquals("healthy", graph.healthMetadata().get("status"));
        assertEquals(1.0d, graph.healthMetadata().get("healthScore"));
        assertEquals("info", graph.healthMetadata().get("alertLevel"));
        assertEquals("No lineage action required.", graph.healthMetadata().get("recommendedAction"));
        Map<?, ?> healthyBadge = (Map<?, ?>) graph.healthMetadata().get("healthBadge");
        assertEquals("healthy", healthyBadge.get("status"));
        assertEquals("success", healthyBadge.get("variant"));
        assertEquals("lineage-health-healthy", healthyBadge.get("token"));
        assertEquals("pass", healthyBadge.get("checkStatus"));
        assertEquals("GOLLEK_LINEAGE_HEALTHY", healthyBadge.get("primaryCheckCode"));
        assertEquals("none", healthyBadge.get("primaryCheckType"));
        assertEquals("info", healthyBadge.get("primaryCheckSeverity"));
        assertEquals("No action required.", healthyBadge.get("primaryCheckAction"));
        assertEquals(5, ((List<?>) graph.healthMetadata().get("checks")).size());
        assertEquals(List.of(), graph.healthMetadata().get("failingChecks"));
        assertEquals(5, graph.healthMetadata().get("passingCheckCount"));
        assertEquals(0, graph.healthMetadata().get("failingCheckCount"));
        Map<?, ?> healthyCheckSummary = (Map<?, ?>) graph.healthMetadata().get("checkSummary");
        assertEquals(5, healthyCheckSummary.get("total"));
        assertEquals(5, healthyCheckSummary.get("passed"));
        assertEquals(0, healthyCheckSummary.get("failed"));
        assertEquals(true, healthyCheckSummary.get("allPassed"));
        assertEquals("GOLLEK_LINEAGE_HEALTHY", healthyCheckSummary.get("primaryFailingCode"));
        assertEquals("none", healthyCheckSummary.get("primaryFailingType"));
        assertEquals("info", healthyCheckSummary.get("primaryFailingSeverity"));
        assertEquals("No action required.", healthyCheckSummary.get("primaryFailingAction"));
        assertFalse(healthyCheckSummary.containsKey("primaryFailingCheck"));
        assertFalse(graph.healthMetadata().containsKey("primaryIssueCode"));
        assertFalse(graph.healthMetadata().containsKey("primaryFailingCheck"));
        assertFalse(graph.healthMetadata().containsKey("primaryFailingCheckCode"));
        assertEquals(List.of("run-parent", "run-child"), selectedChain.runIds());
        assertTrue(selectedChain.complete());
        assertEquals(2, selectedChain.depth());
        assertEquals("resume-test", selectedChain.nodes().get(1).lineage().attributes().get("source"));
        assertEquals(selectedChain.toMetadata(), ((Map<?, ?>) inspection.toMetadata().get("selectedLineageChain")));
    }

    @Test
    void reportsMissingLineageParents() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest orphan = manifest(report, "run-orphan", 100L, 18L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-origin",
                        "missing-parent",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("orphan"), orphan);

        DiscreteTokenDatasetCheckpointLineageGraph graph =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                                rootDir,
                                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest())
                        .lineageGraph();
        DiscreteTokenDatasetCheckpointLineageGraph.LineageChain chain = graph.chainForRunId("run-orphan");

        assertFalse(graph.healthy());
        assertEquals("unhealthy", graph.status());
        assertEquals(0.0d, graph.healthScore());
        assertEquals("critical", graph.alertLevel());
        assertEquals(List.of("missing-parent"), graph.missingParentRunIds());
        assertEquals(1, graph.unresolvedNodes().size());
        assertEquals(List.of("missing-parent"), graph.issueTypes());
        assertEquals(1, graph.issueDetailCount());
        assertEquals(1, graph.blockingIssueCount());
        assertEquals(List.of("GOLLEK_LINEAGE_MISSING_PARENT"), graph.issueCodes());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", graph.primaryIssueCode().orElseThrow());
        assertEquals("missing-parent", graph.primaryIssueType().orElseThrow());
        assertTrue(graph.primaryIssueAction().orElseThrow().contains("Restore the parent checkpoint"));
        assertEquals("checkpoint-lineage-health", graph.healthMetadata().get("kind"));
        assertEquals("gollek.checkpoint-lineage-health.v1", graph.healthMetadata().get("schemaVersion"));
        DiscreteTokenDatasetCheckpointLineageHealthSnapshot unhealthySnapshot =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.fromMetadata(graph.healthMetadata());
        assertEquals(graph.healthMetadata(), unhealthySnapshot.toMetadata());
        assertTrue(unhealthySnapshot.isCurrentSchema());
        unhealthySnapshot.requireCurrentSchema();
        assertTrue(unhealthySnapshot.hasFailures());
        assertEquals("unhealthy", unhealthySnapshot.status());
        assertEquals(1, unhealthySnapshot.failingCheckCount());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", unhealthySnapshot.primaryIssueCode().orElseThrow());
        assertEquals("missing-parent", unhealthySnapshot.primaryIssueType().orElseThrow());
        assertTrue(unhealthySnapshot.primaryIssueAction().orElseThrow().contains("Restore the parent checkpoint"));
        assertEquals("parentExists", unhealthySnapshot.primaryFailingCheckName().orElseThrow());
        assertEquals("missing-parent", unhealthySnapshot.primaryFailingCheckType().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", unhealthySnapshot.primaryFailingCheckCode().orElseThrow());
        assertTrue(unhealthySnapshot.primaryFailingCheckAction().orElseThrow()
                .contains("Restore the parent checkpoint"));
        assertThrows(IllegalStateException.class, unhealthySnapshot::requireHealthy);
        assertEquals(5, graph.checks().size());
        assertEquals(4, graph.passingCheckCount());
        assertEquals(1, graph.failingCheckCount());
        Map<?, ?> failingCheck = graph.failingChecks().get(0);
        assertEquals("parentExists", failingCheck.get("name"));
        assertEquals("missing-parent", failingCheck.get("type"));
        assertEquals("critical", failingCheck.get("severity"));
        assertEquals(List.of("missing-parent"), failingCheck.get("runIds"));
        assertEquals(List.of("run-orphan"), failingCheck.get("affectedRunIds"));
        assertEquals(failingCheck, graph.primaryFailingCheck().orElseThrow());
        assertEquals("parentExists", graph.primaryFailingCheckName().orElseThrow());
        assertEquals("missing-parent", graph.primaryFailingCheckType().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", graph.primaryFailingCheckCode().orElseThrow());
        assertEquals("critical", graph.primaryFailingCheckSeverity().orElseThrow());
        assertTrue(graph.primaryFailingCheckAction().orElseThrow().contains("Restore the parent checkpoint"));
        assertTrue(graph.primaryFailingCheckMessage().orElseThrow().contains("missing parent"));
        assertEquals(5, graph.checkSummary().get("total"));
        assertEquals(4, graph.checkSummary().get("passed"));
        assertEquals(1, graph.checkSummary().get("failed"));
        assertEquals(false, graph.checkSummary().get("allPassed"));
        assertEquals("critical", graph.checkSummary().get("dominantSeverity"));
        assertEquals("parentExists", graph.checkSummary().get("primaryFailingName"));
        assertEquals("missing-parent", graph.checkSummary().get("primaryFailingType"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", graph.checkSummary().get("primaryFailingCode"));
        assertEquals("critical", graph.checkSummary().get("primaryFailingSeverity"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingAction"))
                .contains("Restore the parent checkpoint"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingMessage")).contains("missing parent"));
        assertEquals(failingCheck, graph.checkSummary().get("primaryFailingCheck"));
        assertEquals(1L, ((Map<?, ?>) graph.checkSummary().get("failingSeverityCounts")).get("critical"));
        assertEquals(4L, ((Map<?, ?>) graph.checkSummary().get("severityCounts")).get("info"));
        List<?> issueDetails = (List<?>) graph.healthMetadata().get("issueDetails");
        assertEquals(1, issueDetails.size());
        assertEquals(1, graph.healthMetadata().get("issueDetailCount"));
        assertEquals(1, graph.healthMetadata().get("blockingIssueCount"));
        assertEquals(List.of("GOLLEK_LINEAGE_MISSING_PARENT"), graph.healthMetadata().get("issueCodes"));
        assertEquals("unhealthy", graph.healthMetadata().get("status"));
        assertEquals(0.0d, graph.healthMetadata().get("healthScore"));
        assertEquals("critical", graph.healthMetadata().get("alertLevel"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", graph.healthMetadata().get("primaryIssueCode"));
        assertEquals("missing-parent", graph.healthMetadata().get("primaryIssueType"));
        assertTrue(String.valueOf(graph.healthMetadata().get("recommendedAction"))
                .contains("Restore the parent checkpoint"));
        Map<?, ?> unhealthyBadge = (Map<?, ?>) graph.healthMetadata().get("healthBadge");
        assertEquals("unhealthy", unhealthyBadge.get("status"));
        assertEquals("danger", unhealthyBadge.get("variant"));
        assertEquals("lineage-health-unhealthy", unhealthyBadge.get("token"));
        assertEquals("fail", unhealthyBadge.get("checkStatus"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", unhealthyBadge.get("primaryIssueCode"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", unhealthyBadge.get("primaryCheckCode"));
        assertEquals("missing-parent", unhealthyBadge.get("primaryCheckType"));
        assertEquals("critical", unhealthyBadge.get("primaryCheckSeverity"));
        assertTrue(String.valueOf(unhealthyBadge.get("primaryCheckAction"))
                .contains("Restore the parent checkpoint"));
        assertEquals(5, ((List<?>) graph.healthMetadata().get("checks")).size());
        List<?> healthFailingChecks = (List<?>) graph.healthMetadata().get("failingChecks");
        assertEquals(1, healthFailingChecks.size());
        assertEquals(failingCheck, healthFailingChecks.get(0));
        assertEquals(failingCheck, graph.healthMetadata().get("primaryFailingCheck"));
        assertEquals("parentExists", graph.healthMetadata().get("primaryFailingCheckName"));
        assertEquals("missing-parent", graph.healthMetadata().get("primaryFailingCheckType"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", graph.healthMetadata().get("primaryFailingCheckCode"));
        assertEquals("critical", graph.healthMetadata().get("primaryFailingCheckSeverity"));
        assertTrue(String.valueOf(graph.healthMetadata().get("primaryFailingCheckAction"))
                .contains("Restore the parent checkpoint"));
        assertTrue(String.valueOf(graph.healthMetadata().get("primaryFailingCheckMessage")).contains("missing parent"));
        assertEquals(4, graph.healthMetadata().get("passingCheckCount"));
        assertEquals(1, graph.healthMetadata().get("failingCheckCount"));
        Map<?, ?> unhealthyCheckSummary = (Map<?, ?>) graph.healthMetadata().get("checkSummary");
        assertEquals(4, unhealthyCheckSummary.get("passed"));
        assertEquals(1, unhealthyCheckSummary.get("failed"));
        assertEquals(List.of("parentExists"), unhealthyCheckSummary.get("failingNames"));
        assertEquals("parentExists", unhealthyCheckSummary.get("primaryFailingName"));
        assertEquals("missing-parent", unhealthyCheckSummary.get("primaryFailingType"));
        assertEquals("critical", unhealthyCheckSummary.get("primaryFailingSeverity"));
        assertEquals(failingCheck, unhealthyCheckSummary.get("primaryFailingCheck"));
        Map<?, ?> issueDetail = (Map<?, ?>) issueDetails.get(0);
        assertEquals(issueDetail, graph.healthMetadata().get("primaryIssue"));
        assertEquals("missing-parent", issueDetail.get("type"));
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", issueDetail.get("code"));
        assertEquals("error", issueDetail.get("severity"));
        assertEquals(true, issueDetail.get("blocking"));
        assertEquals("run-orphan", issueDetail.get("runId"));
        assertEquals("missing-parent", issueDetail.get("parentRunId"));
        assertTrue(String.valueOf(issueDetail.get("action")).contains("Restore the parent checkpoint"));
        assertThrows(IllegalStateException.class, graph::requireHealthy);
        assertFalse(chain.complete());
        assertEquals("missing-parent", chain.missingParentRunId());
    }

    @Test
    void prioritizesPrimaryIssueAcrossMultipleLineageProblems() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest duplicateA = manifest(report, "run-duplicate", 100L, 12L)
                .build();
        DiscreteTokenDatasetCheckpointManifest duplicateB = manifest(report, "run-duplicate", 200L, 18L)
                .build();
        DiscreteTokenDatasetCheckpointManifest orphan = manifest(report, "run-orphan", 300L, 24L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-origin",
                        "missing-parent",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("duplicate-a"), duplicateA);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("duplicate-b"), duplicateB);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("orphan"), orphan);

        DiscreteTokenDatasetCheckpointLineageGraph graph =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                                rootDir,
                                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest())
                        .lineageGraph();

        assertFalse(graph.healthy());
        assertEquals(List.of("run-duplicate"), graph.duplicateRunIds());
        assertEquals(List.of("missing-parent"), graph.missingParentRunIds());
        assertEquals(List.of("duplicate-run-id", "missing-parent"), graph.issueTypes());
        assertEquals(
                List.of("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", "GOLLEK_LINEAGE_MISSING_PARENT"),
                graph.issueCodes());
        assertEquals(2, graph.issueCount());
        assertEquals(2, graph.issueDetailCount());
        assertEquals(2, graph.blockingIssueCount());
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", graph.primaryIssueCode().orElseThrow());
        assertEquals("duplicate-run-id", graph.primaryIssueType().orElseThrow());
        assertTrue(graph.primaryIssueAction().orElseThrow().contains("unique run id"));
        assertEquals(2, graph.failingCheckCount());
        assertEquals(List.of("uniqueRunIds", "parentExists"), graph.checkSummary().get("failingNames"));
        assertEquals(
                List.of("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", "GOLLEK_LINEAGE_MISSING_PARENT"),
                graph.checkSummary().get("failingCodes"));
        assertEquals("uniqueRunIds", graph.checkSummary().get("primaryFailingName"));
        assertEquals("duplicate-run-id", graph.checkSummary().get("primaryFailingType"));
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", graph.checkSummary().get("primaryFailingCode"));
        assertEquals("critical", graph.checkSummary().get("primaryFailingSeverity"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingAction")).contains("unique run id"));
        assertEquals("uniqueRunIds", graph.primaryFailingCheckName().orElseThrow());
        assertEquals("duplicate-run-id", graph.primaryFailingCheckType().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", graph.primaryFailingCheckCode().orElseThrow());
        assertTrue(graph.primaryFailingCheckAction().orElseThrow().contains("unique run id"));
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", graph.checkSummary().get("primaryIssueCode"));
        assertTrue(String.valueOf(graph.recommendedAction()).contains("unique run id"));
        Map<?, ?> health = graph.healthMetadata();
        assertEquals("checkpoint-lineage-health", health.get("kind"));
        assertEquals("gollek.checkpoint-lineage-health.v1", health.get("schemaVersion"));
        assertEquals(graph.primaryIssue().orElseThrow(), health.get("primaryIssue"));
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", health.get("primaryIssueCode"));
        assertEquals("duplicate-run-id", health.get("primaryIssueType"));
        assertEquals("uniqueRunIds", health.get("primaryFailingCheckName"));
        assertEquals("duplicate-run-id", health.get("primaryFailingCheckType"));
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", health.get("primaryFailingCheckCode"));
        assertTrue(String.valueOf(health.get("primaryFailingCheckAction")).contains("unique run id"));
        assertEquals(
                List.of("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", "GOLLEK_LINEAGE_MISSING_PARENT"),
                health.get("issueCodes"));
        Map<?, ?> badge = (Map<?, ?>) health.get("healthBadge");
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", badge.get("primaryIssueCode"));
        assertEquals("GOLLEK_LINEAGE_DUPLICATE_RUN_ID", badge.get("primaryCheckCode"));
        assertEquals("duplicate-run-id", badge.get("primaryCheckType"));
        assertEquals("critical", badge.get("primaryCheckSeverity"));
        assertTrue(String.valueOf(badge.get("primaryCheckAction")).contains("unique run id"));
    }

    @Test
    void reportsParentIdentityMismatches() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest parent = manifest(report, "run-parent", 100L, 12L)
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("parent"), parent);
        DiscreteTokenDatasetCheckpointManifest child = manifest(report, "run-child", 200L, 18L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-parent",
                        "run-parent",
                        99L,
                        report.fingerprint().value() + "-changed",
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("child"), child);

        DiscreteTokenDatasetCheckpointLineageGraph graph =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                                rootDir,
                                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest())
                        .lineageGraph();
        DiscreteTokenDatasetCheckpointLineageGraph.LineageChain chain = graph.chainForRunId("run-child");

        assertFalse(graph.healthy());
        assertEquals("unhealthy", graph.status());
        assertEquals(0.0d, graph.healthScore());
        assertEquals("critical", graph.alertLevel());
        assertEquals(List.of("run-child"), graph.parentMismatchRunIds());
        assertEquals(List.of("parent-mismatch"), graph.issueTypes());
        assertEquals(1, graph.issueDetailCount());
        assertEquals(1, graph.blockingIssueCount());
        assertEquals(List.of("GOLLEK_LINEAGE_PARENT_MISMATCH"), graph.issueCodes());
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", graph.primaryIssueCode().orElseThrow());
        assertEquals("parent-mismatch", graph.primaryIssueType().orElseThrow());
        assertTrue(graph.primaryIssueAction().orElseThrow().contains("Regenerate the child lineage"));
        assertEquals(5, graph.checks().size());
        assertEquals(4, graph.passingCheckCount());
        assertEquals(1, graph.failingCheckCount());
        Map<?, ?> failingCheck = graph.failingChecks().get(0);
        assertEquals("parentIdentity", failingCheck.get("name"));
        assertEquals("parent-mismatch", failingCheck.get("type"));
        assertEquals("critical", failingCheck.get("severity"));
        assertEquals(List.of("run-child"), failingCheck.get("runIds"));
        assertEquals(List.of("run-child"), failingCheck.get("affectedRunIds"));
        assertEquals(failingCheck, graph.primaryFailingCheck().orElseThrow());
        assertEquals("parentIdentity", graph.primaryFailingCheckName().orElseThrow());
        assertEquals("parent-mismatch", graph.primaryFailingCheckType().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", graph.primaryFailingCheckCode().orElseThrow());
        assertTrue(graph.primaryFailingCheckAction().orElseThrow().contains("Regenerate the child lineage"));
        assertEquals(List.of("parentIdentity"), graph.checkSummary().get("failingNames"));
        assertEquals("parentIdentity", graph.checkSummary().get("primaryFailingName"));
        assertEquals("parent-mismatch", graph.checkSummary().get("primaryFailingType"));
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", graph.checkSummary().get("primaryFailingCode"));
        assertEquals("critical", graph.checkSummary().get("primaryFailingSeverity"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingAction"))
                .contains("Regenerate the child lineage"));
        assertEquals(failingCheck, graph.checkSummary().get("primaryFailingCheck"));
        assertEquals(1L, ((Map<?, ?>) graph.checkSummary().get("failingSeverityCounts")).get("critical"));
        List<?> issueDetails = (List<?>) graph.healthMetadata().get("issueDetails");
        assertEquals(1, issueDetails.size());
        assertEquals(1, graph.healthMetadata().get("issueDetailCount"));
        assertEquals(1, graph.healthMetadata().get("blockingIssueCount"));
        assertEquals(List.of("GOLLEK_LINEAGE_PARENT_MISMATCH"), graph.healthMetadata().get("issueCodes"));
        assertEquals("unhealthy", graph.healthMetadata().get("status"));
        assertEquals(0.0d, graph.healthMetadata().get("healthScore"));
        assertEquals("critical", graph.healthMetadata().get("alertLevel"));
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", graph.healthMetadata().get("primaryIssueCode"));
        assertEquals("parent-mismatch", graph.healthMetadata().get("primaryIssueType"));
        assertTrue(String.valueOf(graph.healthMetadata().get("recommendedAction"))
                .contains("Regenerate the child lineage"));
        Map<?, ?> unhealthyBadge = (Map<?, ?>) graph.healthMetadata().get("healthBadge");
        assertEquals("unhealthy", unhealthyBadge.get("status"));
        assertEquals("danger", unhealthyBadge.get("variant"));
        assertEquals("lineage-health-unhealthy", unhealthyBadge.get("token"));
        assertEquals("fail", unhealthyBadge.get("checkStatus"));
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", unhealthyBadge.get("primaryIssueCode"));
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", unhealthyBadge.get("primaryCheckCode"));
        assertEquals("parent-mismatch", unhealthyBadge.get("primaryCheckType"));
        assertEquals("critical", unhealthyBadge.get("primaryCheckSeverity"));
        assertTrue(String.valueOf(unhealthyBadge.get("primaryCheckAction"))
                .contains("Regenerate the child lineage"));
        assertEquals(1, ((List<?>) graph.healthMetadata().get("failingChecks")).size());
        assertEquals(4, graph.healthMetadata().get("passingCheckCount"));
        assertEquals(1, graph.healthMetadata().get("failingCheckCount"));
        Map<?, ?> issueDetail = (Map<?, ?>) issueDetails.get(0);
        assertEquals(issueDetail, graph.healthMetadata().get("primaryIssue"));
        assertEquals("parent-mismatch", issueDetail.get("type"));
        assertEquals("GOLLEK_LINEAGE_PARENT_MISMATCH", issueDetail.get("code"));
        assertEquals("error", issueDetail.get("severity"));
        assertEquals(true, issueDetail.get("blocking"));
        assertEquals("run-child", issueDetail.get("runId"));
        assertEquals("run-parent", issueDetail.get("parentRunId"));
        assertTrue(String.valueOf(issueDetail.get("action")).contains("Regenerate the child lineage"));
        List<?> reasons = (List<?>) issueDetail.get("reasons");
        assertTrue(reasons.stream().anyMatch(reason -> String.valueOf(reason).contains("checkpointStep expected 99")));
        assertTrue(reasons.stream().anyMatch(reason -> String.valueOf(reason).contains("dataset fingerprint expected")));
        assertEquals(1, graph.unresolvedNodes().size());
        assertFalse(chain.complete());
        assertEquals(List.of("run-child"), chain.parentMismatchRunIds());
        assertTrue(chain.nodes().get(1).parentMismatchReasons().stream()
                .anyMatch(reason -> reason.contains("checkpointStep expected 99")));
        assertTrue(chain.nodes().get(1).parentMismatchReasons().stream()
                .anyMatch(reason -> reason.contains("dataset fingerprint expected")));
    }

    @Test
    void reportsLineageCyclesAsUnhealthyGraph() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest runA = manifest(report, "run-a", 100L, 12L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-a",
                        "run-b",
                        18L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetCheckpointManifest runB = manifest(report, "run-b", 200L, 18L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-a",
                        "run-a",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("run-a"), runA);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("run-b"), runB);

        DiscreteTokenDatasetCheckpointLineageGraph graph =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                                rootDir,
                                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest())
                        .lineageGraph();
        DiscreteTokenDatasetCheckpointLineageGraph.LineageChain chain = graph.chainForRunId("run-a");

        assertFalse(graph.healthy());
        assertEquals(List.of("run-a", "run-b"), graph.cycleRunIds());
        assertEquals(List.of("cycle"), graph.issueTypes());
        assertEquals(2, graph.unresolvedNodes().size());
        assertEquals(List.of("run-a", "run-b"), ((Map<?, ?>) graph.healthMetadata()).get("cycleRunIds"));
        assertFalse(chain.complete());
        assertTrue(chain.cycleDetected());
    }

    @Test
    void prioritizesPrimaryFailingCheckByIssueSeverityOrder() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest runA = manifest(report, "run-a", 100L, 12L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-a",
                        "run-b",
                        18L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetCheckpointManifest runB = manifest(report, "run-b", 200L, 18L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-a",
                        "run-a",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetCheckpointManifest orphan = manifest(report, "run-orphan", 300L, 24L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-a",
                        "missing-parent",
                        12L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("run-a"), runA);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("run-b"), runB);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("orphan"), orphan);

        DiscreteTokenDatasetCheckpointLineageGraph graph =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                                rootDir,
                                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest())
                        .lineageGraph();

        assertFalse(graph.healthy());
        assertEquals(List.of("missing-parent", "cycle"), graph.issueTypes());
        assertEquals(2, graph.issueCount());
        assertEquals(2, graph.failingCheckCount());
        assertEquals("GOLLEK_LINEAGE_CYCLE", graph.primaryIssueCode().orElseThrow());
        assertEquals("cycle", graph.primaryIssueType().orElseThrow());
        assertTrue(graph.primaryIssueAction().orElseThrow().contains("without loops"));
        assertEquals(List.of("parentExists", "acyclic"), graph.checkSummary().get("failingNames"));
        assertEquals(
                List.of("GOLLEK_LINEAGE_MISSING_PARENT", "GOLLEK_LINEAGE_CYCLE"),
                graph.checkSummary().get("failingCodes"));
        assertEquals("acyclic", graph.checkSummary().get("primaryFailingName"));
        assertEquals("cycle", graph.checkSummary().get("primaryFailingType"));
        assertEquals("GOLLEK_LINEAGE_CYCLE", graph.checkSummary().get("primaryFailingCode"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingMessage")).contains("lineage cycle"));
        assertTrue(String.valueOf(graph.checkSummary().get("primaryFailingAction")).contains("without loops"));
        Map<?, ?> primaryFailingCheck = (Map<?, ?>) graph.checkSummary().get("primaryFailingCheck");
        assertEquals("acyclic", primaryFailingCheck.get("name"));
        assertEquals("cycle", primaryFailingCheck.get("type"));
        assertEquals(primaryFailingCheck, graph.primaryFailingCheck().orElseThrow());
        assertEquals("acyclic", graph.primaryFailingCheckName().orElseThrow());
        assertEquals("cycle", graph.primaryFailingCheckType().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_CYCLE", graph.primaryFailingCheckCode().orElseThrow());
        assertEquals("critical", graph.primaryFailingCheckSeverity().orElseThrow());
        assertTrue(graph.primaryFailingCheckAction().orElseThrow().contains("without loops"));
        assertTrue(graph.primaryFailingCheckMessage().orElseThrow().contains("lineage cycle"));
        assertEquals("GOLLEK_LINEAGE_CYCLE", graph.checkSummary().get("primaryIssueCode"));
        Map<?, ?> badge = (Map<?, ?>) graph.healthMetadata().get("healthBadge");
        assertEquals(primaryFailingCheck, graph.healthMetadata().get("primaryFailingCheck"));
        assertEquals("acyclic", graph.healthMetadata().get("primaryFailingCheckName"));
        assertEquals("cycle", graph.healthMetadata().get("primaryFailingCheckType"));
        assertEquals("GOLLEK_LINEAGE_CYCLE", graph.healthMetadata().get("primaryFailingCheckCode"));
        assertTrue(String.valueOf(graph.healthMetadata().get("primaryFailingCheckAction")).contains("without loops"));
        assertEquals("GOLLEK_LINEAGE_CYCLE", badge.get("primaryIssueCode"));
        assertEquals("GOLLEK_LINEAGE_CYCLE", badge.get("primaryCheckCode"));
        assertEquals("cycle", badge.get("primaryCheckType"));
        assertTrue(String.valueOf(badge.get("primaryCheckAction")).contains("without loops"));
    }

    @Test
    void recordsStrictSelectionFailureWithoutHidingCandidates() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 300L);
        Path corruptDir = rootDir.resolve("corrupt");
        Files.createDirectories(corruptDir);
        Files.writeString(
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(corruptDir),
                "{bad-json");

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestReady();
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                inventory.inspectCheckpoints(policy);

        assertFalse(report.selectionSatisfied());
        assertFalse(report.selected());
        assertTrue(report.selectionFailure().orElseThrow().contains("read failure"));
        assertEquals(1, report.acceptedCount());
        assertEquals(0, report.rejectedCount());
        assertThrows(IllegalStateException.class, report::requireSelectedCheckpoint);
        assertEquals(report.summary(), DiscreteTokenDatasetTrainerCheckpointBridge
                .inspectCheckpoints(rootDir, policy)
                .summary());
        assertTrue(report.toMetadata().containsKey("selectionFailure"));
    }

    @Test
    void rejectsMalformedInputs() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 100L);
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);
        DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint =
                inventory.checkpoints().get(0);

        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                        (Path) null,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                        inventory,
                        null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInspectionReport(
                        null,
                        inventory,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady(),
                        Optional.empty(),
                        Optional.empty(),
                        List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInspectionReport(
                        rootDir.resolve("other"),
                        inventory,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady(),
                        Optional.empty(),
                        Optional.empty(),
                        List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInspectionReport(
                        rootDir,
                        inventory,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady(),
                        Optional.empty(),
                        Optional.of(" "),
                        List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision(
                        checkpoint,
                        true,
                        List.of("bad")));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInspectionReport.CheckpointDecision(
                        checkpoint,
                        false,
                        List.of()));
    }

    private static void writeReadyCheckpoint(Path checkpointDir, String runId, long createdAt) throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(checkpointDir, plan, policy);
    }

    private static void writeManifestOnlyCheckpoint(Path checkpointDir, String runId, long createdAt)
            throws IOException {
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(
                checkpointDir,
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt)
                        .build());
    }

    private static void writeBlockedCheckpoint(Path checkpointDir, String runId, long createdAt) throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changedPlan()));
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report,
            String runId,
            long createdAt) {
        return manifest(report, runId, createdAt, 12L);
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report,
            String runId,
            long createdAt,
            long checkpointStep) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-nqueens")
                .runId(runId)
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(checkpointStep)
                .createdAtEpochMillis(createdAt)
                .createdBy("inspection-report-test");
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
