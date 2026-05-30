package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerHealthMetadataPublisherTest {

    @Test
    void publishesUnavailableUnknownStatusWithSkipReason() {
        Map<String, Object> metadata = new LinkedHashMap<>();

        TrainerHealthMetadataPublisher.putUnavailable(metadata, "exampleHealth", "not-enabled");

        assertEquals(Boolean.FALSE, metadata.get("exampleHealth.available"));
        assertEquals("not-enabled", metadata.get("exampleHealth.skipReason"));
        assertEquals("unknown", metadata.get("exampleHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("exampleHealthGatePassed"));
    }

    @Test
    void publishesHealthyStatusForEmptyIssueList() {
        Map<String, Object> metadata = new LinkedHashMap<>();

        TrainerHealthMetadataPublisher.put(metadata, "exampleHealth", List.of());

        assertEquals(Boolean.TRUE, metadata.get("exampleHealth.available"));
        assertEquals("healthy", metadata.get("exampleHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("exampleHealthHealthy"));
        assertEquals(Boolean.TRUE, metadata.get("exampleHealthGatePassed"));
        assertEquals(Boolean.FALSE, metadata.get("exampleHealthIssueDetected"));
        assertEquals(0, metadata.get("exampleHealthIssueCount"));
        assertEquals(0, metadata.get("exampleHealthWarningCount"));
        assertEquals(0, metadata.get("exampleHealthErrorCount"));
        assertEquals(List.of(), metadata.get("exampleHealthIssues"));
    }

    @Test
    void publishesWarningAndErrorIssueSummaryWithImmutableCollections() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        List<TrainerHealthIssue> issues = List.of(
                TrainerHealthIssue.warning(
                        "example",
                        "example-warning",
                        "data",
                        "warning message",
                        "inspect warning",
                        Map.of("sampleCount", 3)),
                TrainerHealthIssue.issue(
                        "example",
                        "example-error",
                        TrainerHealthIssue.ERROR,
                        true,
                        "trainer",
                        "error message",
                        "fix error",
                        Map.of("phase", "train")));

        TrainerHealthMetadataPublisher.put(metadata, "exampleHealth", issues);

        assertEquals("error", metadata.get("exampleHealthStatus"));
        assertEquals(Boolean.FALSE, metadata.get("exampleHealthHealthy"));
        assertEquals(Boolean.FALSE, metadata.get("exampleHealthGatePassed"));
        assertEquals(Boolean.TRUE, metadata.get("exampleHealthIssueDetected"));
        assertEquals(2, metadata.get("exampleHealthIssueCount"));
        assertEquals(1, metadata.get("exampleHealthWarningCount"));
        assertEquals(1, metadata.get("exampleHealthErrorCount"));
        assertEquals(List.of("example-warning", "example-error"), metadata.get("exampleHealthIssueCodes"));
        assertEquals(List.of("warning", "error"), metadata.get("exampleHealthIssueSeverities"));
        assertEquals(List.of("inspect warning", "fix error"), metadata.get("exampleHealthRecommendedActions"));
        assertEquals("trainer", healthIssues(metadata).get(1).get("artifact"));
        assertThrows(UnsupportedOperationException.class, () -> healthIssues(metadata).clear());
        assertThrows(UnsupportedOperationException.class, () -> healthIssues(metadata).get(0).clear());
        assertThrows(
                UnsupportedOperationException.class,
                () -> issueCodes(metadata).clear());
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> healthIssues(Map<String, Object> metadata) {
        return (List<Map<String, Object>>) metadata.get("exampleHealthIssues");
    }

    @SuppressWarnings("unchecked")
    private static List<String> issueCodes(Map<String, Object> metadata) {
        return (List<String>) metadata.get("exampleHealthIssueCodes");
    }
}
