package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportHealthIssueMarkdownTest {

    @Test
    void renderDetailsIncludesMessageEvidenceAndEscapesTableCells() {
        String markdown = TrainingReportHealthIssueMarkdown.renderDetails(
                "Data | Health",
                List.of(Map.of(
                        "code", "data-loader-train-prefetch-buffer-too-small",
                        "severity", "warning",
                        "artifact", "train|loader",
                        "blocking", false,
                        "message", "buffer | too small",
                        "evidence", Map.of(
                                "buffer", 1,
                                "enabled", true))));

        assertTrue(markdown.contains("### Data | Health"));
        assertTrue(markdown.contains("| Code | Severity | Artifact | Blocking | Message | Evidence |"));
        assertTrue(markdown.contains("`data-loader-train-prefetch-buffer-too-small`"));
        assertTrue(markdown.contains("train\\|loader"));
        assertTrue(markdown.contains("buffer \\| too small"));
        assertTrue(markdown.contains("buffer=1, enabled=true"));
    }

    @Test
    void renderDetailsReturnsBlankWithoutIssues() {
        assertEquals("", TrainingReportHealthIssueMarkdown.renderDetails("Data Health Issue Details", null));
        assertEquals("", TrainingReportHealthIssueMarkdown.renderDetails("Data Health Issue Details", List.of()));
    }
}
