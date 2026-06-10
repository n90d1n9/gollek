package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportHealthIssueSummariesTest {

    @Test
    void fromCreatesCompactMachineReadableIssueSummaries() {
        List<Map<String, Object>> summaries = TrainingReportHealthIssueSummaries.from(List.of(Map.of(
                "code", "data-loader-train-prefetch-buffer-too-small",
                "severity", "warning",
                "artifact", "train",
                "blocking", false,
                "message", "prefetch buffer is too small",
                "action", "increase prefetch buffer",
                "evidence", Map.of(
                        "trainLoaderPlan.prefetch.enabled", true,
                        "trainLoaderPlan.prefetch.maxBufferedItems", 1))));

        assertEquals(1, summaries.size());
        Map<String, Object> summary = summaries.get(0);
        assertEquals("data-loader-train-prefetch-buffer-too-small", summary.get("code"));
        assertEquals("warning", summary.get("severity"));
        assertEquals("train", summary.get("artifact"));
        assertEquals(Boolean.FALSE, summary.get("blocking"));
        assertEquals("prefetch buffer is too small", summary.get("message"));
        assertEquals("increase prefetch buffer", summary.get("action"));
        assertEquals("trainLoaderPlan.prefetch.enabled=true, trainLoaderPlan.prefetch.maxBufferedItems=1",
                summary.get("evidenceSummary"));
        assertThrows(UnsupportedOperationException.class, () -> summaries.add(Map.of()));
        assertThrows(UnsupportedOperationException.class, () -> summary.put("extra", true));
    }

    @Test
    void fromReturnsEmptySummaryForMissingIssues() {
        assertEquals(List.of(), TrainingReportHealthIssueSummaries.from(null));
        assertEquals(List.of(), TrainingReportHealthIssueSummaries.from(List.of()));
    }
}
