package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportDataHealthTest {

    @Test
    @SuppressWarnings("unchecked")
    void fromMetadataPromotesLoaderAndDistributionHealthIntoTypedAggregate() {
        Map<String, Object> loaderIssue = Map.of(
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "message", "dropLast discarded 3 training samples",
                "action", "disable dropLast or rebalance batch size");
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "warning"),
                Map.entry("dataLoaderPlanHealthHealthy", false),
                Map.entry("dataLoaderPlanHealthGatePassed", true),
                Map.entry("dataLoaderPlanHealthIssueDetected", true),
                Map.entry("dataLoaderPlanHealthIssueCount", 1),
                Map.entry("dataLoaderPlanHealthWarningCount", 1),
                Map.entry("dataLoaderPlanHealthErrorCount", 0),
                Map.entry("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-drop-last-discarded-samples")),
                Map.entry("dataLoaderPlanHealthIssueSeverities", List.of("warning")),
                Map.entry("dataLoaderPlanHealthRecommendedActions", List.of("disable dropLast or rebalance batch size")),
                Map.entry("dataLoaderPlanHealthIssues", List.of(loaderIssue)),
                Map.entry("dataDistributionHealth.available", true),
                Map.entry("dataDistributionHealthStatus", "healthy"),
                Map.entry("dataDistributionHealthHealthy", true),
                Map.entry("dataDistributionHealthGatePassed", true),
                Map.entry("dataDistributionHealthIssueDetected", false),
                Map.entry("dataDistributionHealthIssueCount", 0),
                Map.entry("dataDistributionHealthWarningCount", 0),
                Map.entry("dataDistributionHealthErrorCount", 0),
                Map.entry("dataDistributionHealthIssueCodes", List.of()),
                Map.entry("dataDistributionHealthIssueSeverities", List.of()),
                Map.entry("dataDistributionHealthRecommendedActions", List.of()),
                Map.entry("dataDistributionHealthIssues", List.of()));

        TrainingReportDataHealth health = TrainingReportDataHealth.fromMetadata(metadata);

        assertTrue(health.available());
        assertFalse(health.healthy());
        assertTrue(health.gatePassed());
        assertTrue(health.issueDetected());
        assertEquals(1, health.issueCount());
        assertEquals(1, health.warningCount());
        assertEquals(0, health.errorCount());
        assertEquals("warning", health.loaderPlan().status());
        assertEquals("healthy", health.distribution().status());
        assertEquals(List.of("data-loader-train-drop-last-discarded-samples"), health.issueCodes());
        assertEquals(List.of("disable dropLast or rebalance batch size"), health.recommendedActions());

        Map<String, Object> map = health.toMap();
        assertEquals(Boolean.TRUE, map.get("available"));
        assertEquals(Boolean.FALSE, map.get("healthy"));
        assertEquals(1, map.get("issueCount"));
        assertEquals(loaderIssue.get("code"), ((List<Map<String, Object>>) map.get("issues")).get(0).get("code"));
        assertEquals(loaderIssue.get("code"),
                ((List<Map<String, Object>>) map.get("issueSummaries")).get(0).get("code"));
        assertEquals("dropLast discarded 3 training samples",
                ((List<Map<String, Object>>) map.get("issueSummaries")).get(0).get("message"));
        assertEquals("warning", ((Map<String, Object>) map.get("loaderPlan")).get("status"));
    }

    @Test
    void issueSummaryIncludesSeverityCodeAndAction() {
        TrainingReportDataHealth health = TrainingReportDataHealth.fromMetadata(
                TrainingReportQualityProfileTestFixtures.warningDataHealthMetadata());

        assertEquals(
                "warning data-loader-train-drop-last-discarded-samples"
                        + " - adjust batch size or disable dropLast for small datasets",
                health.issueSummary("fallback"));
    }

    @Test
    void markdownRendersIssueDetailsWithPrefetchEvidence() {
        Map<String, Object> loaderIssue = Map.of(
                "code", "data-loader-train-prefetch-buffer-too-small",
                "severity", "warning",
                "artifact", "train",
                "blocking", false,
                "message", "train loader prefetch buffer can hold only 1 item(s)",
                "action", "increase the prefetch buffer",
                "evidence", Map.of(
                        "trainLoaderPlan.prefetch.enabled", true,
                        "trainLoaderPlan.prefetch.maxBufferedItems", 1,
                        "trainLoaderPlan.prefetch.sourceLoaderType", "example.Loader"));
        TrainingReportDataHealth health = TrainingReportDataHealth.fromMetadata(Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "warning"),
                Map.entry("dataLoaderPlanHealthHealthy", false),
                Map.entry("dataLoaderPlanHealthGatePassed", true),
                Map.entry("dataLoaderPlanHealthIssueDetected", true),
                Map.entry("dataLoaderPlanHealthIssueCount", 1),
                Map.entry("dataLoaderPlanHealthWarningCount", 1),
                Map.entry("dataLoaderPlanHealthErrorCount", 0),
                Map.entry("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-prefetch-buffer-too-small")),
                Map.entry("dataLoaderPlanHealthIssueSeverities", List.of("warning")),
                Map.entry("dataLoaderPlanHealthRecommendedActions", List.of("increase the prefetch buffer")),
                Map.entry("dataLoaderPlanHealthIssues", List.of(loaderIssue))));

        String markdown = TrainingReportDataHealthMarkdown.render(health);

        assertTrue(markdown.contains("### Data Health Issue Details"));
        assertTrue(markdown.contains("`data-loader-train-prefetch-buffer-too-small`"));
        assertTrue(markdown.contains("trainLoaderPlan.prefetch.enabled=true, "
                + "trainLoaderPlan.prefetch.maxBufferedItems=1, "
                + "trainLoaderPlan.prefetch.sourceLoaderType=example.Loader"));
        assertTrue(markdown.contains("train loader prefetch buffer can hold only 1 item(s)"));
        assertTrue(markdown.contains("trainLoaderPlan.prefetch.maxBufferedItems=1"));
        assertTrue(markdown.contains("trainLoaderPlan.prefetch.sourceLoaderType=example.Loader"));
        assertTrue(markdown.contains("| `warning` | `train` | `no` | train loader prefetch"));
    }

    @Test
    void fromMapRoundTripsImmutableNestedSectionsAndAggregates() {
        Map<String, Object> dataHealth = Map.of(
                "loaderPlan",
                Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "error"),
                        Map.entry("healthy", false),
                        Map.entry("gatePassed", false),
                        Map.entry("issueDetected", true),
                        Map.entry("issueCount", 1),
                        Map.entry("warningCount", 0),
                        Map.entry("errorCount", 1),
                        Map.entry("issueCodes", List.of("data-loader-train-plan-unavailable")),
                        Map.entry("issueSeverities", List.of("error")),
                        Map.entry("recommendedActions", List.of("restore train loader plan")),
                        Map.entry("issues", List.of(Map.of("code", "data-loader-train-plan-unavailable")))),
                "distribution",
                Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "warning"),
                        Map.entry("healthy", false),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", true),
                        Map.entry("issueCount", 2),
                        Map.entry("warningCount", 2),
                        Map.entry("errorCount", 0),
                        Map.entry("issueCodes", List.of("data-distribution-class-imbalance")),
                        Map.entry("issueSeverities", List.of("warning")),
                        Map.entry("recommendedActions", List.of("rebalance labels")),
                        Map.entry("issues", List.of(Map.of("code", "data-distribution-class-imbalance")))));

        TrainingReportDataHealth health = TrainingReportDataHealth.fromMap(dataHealth);

        assertTrue(health.available());
        assertFalse(health.healthy());
        assertFalse(health.gatePassed());
        assertEquals(3, health.issueCount());
        assertEquals(2, health.warningCount());
        assertEquals(1, health.errorCount());
        assertEquals(
                List.of("data-loader-train-plan-unavailable", "data-distribution-class-imbalance"),
                health.issueCodes());
        assertThrows(UnsupportedOperationException.class, () -> health.issueCodes().add("extra"));
        assertThrows(UnsupportedOperationException.class, () -> health.issues().get(0).put("extra", true));
        assertThrows(UnsupportedOperationException.class, () -> health.toMap().put("extra", true));
    }

    @Test
    @SuppressWarnings("unchecked")
    void fromMapRoundTripsMachineReadableIssueSummaries() {
        Map<String, Object> loaderIssue = Map.of(
                "code", "data-loader-train-prefetch-buffer-too-small",
                "severity", "warning",
                "artifact", "train",
                "blocking", false,
                "message", "train loader prefetch buffer can hold only 1 item(s)",
                "action", "increase the prefetch buffer",
                "evidence", Map.of(
                        "trainLoaderPlan.prefetch.maxBufferedItems", 1,
                        "trainLoaderPlan.prefetch.enabled", true));
        Map<String, Object> dataHealth = Map.of(
                "loaderPlan",
                Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "warning"),
                        Map.entry("healthy", false),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", true),
                        Map.entry("issueCount", 1),
                        Map.entry("warningCount", 1),
                        Map.entry("errorCount", 0),
                        Map.entry("issueCodes", List.of("data-loader-train-prefetch-buffer-too-small")),
                        Map.entry("issueSeverities", List.of("warning")),
                        Map.entry("recommendedActions", List.of("increase the prefetch buffer")),
                        Map.entry("issues", List.of(loaderIssue))),
                "distribution",
                Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "healthy"),
                        Map.entry("healthy", true),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", false),
                        Map.entry("issueCount", 0),
                        Map.entry("warningCount", 0),
                        Map.entry("errorCount", 0),
                        Map.entry("issues", List.of())));

        TrainingReportDataHealth health = TrainingReportDataHealth.fromMap(dataHealth);
        Map<String, Object> roundTrip = health.toMap();
        List<Map<String, Object>> summaries = (List<Map<String, Object>>) roundTrip.get("issueSummaries");

        assertEquals(1, summaries.size());
        assertEquals("data-loader-train-prefetch-buffer-too-small", summaries.get(0).get("code"));
        assertEquals("warning", summaries.get(0).get("severity"));
        assertEquals("train", summaries.get(0).get("artifact"));
        assertEquals(Boolean.FALSE, summaries.get(0).get("blocking"));
        assertEquals("increase the prefetch buffer", summaries.get(0).get("action"));
        assertEquals(
                "trainLoaderPlan.prefetch.enabled=true, trainLoaderPlan.prefetch.maxBufferedItems=1",
                summaries.get(0).get("evidenceSummary"));
        assertThrows(UnsupportedOperationException.class, () -> summaries.get(0).put("extra", true));
    }

    @Test
    void unknownHealthKeepsGateOpenButMarksSectionsUnavailable() {
        TrainingReportDataHealth health = TrainingReportDataHealth.fromMap(Map.of());

        assertFalse(health.available());
        assertFalse(health.healthy());
        assertTrue(health.gatePassed());
        assertFalse(health.issueDetected());
        assertEquals(0, health.issueCount());
        assertEquals("unknown", health.loaderPlan().status());
        assertEquals("data-loader-plan-health-metadata-missing", health.loaderPlan().skipReason());
    }
}
