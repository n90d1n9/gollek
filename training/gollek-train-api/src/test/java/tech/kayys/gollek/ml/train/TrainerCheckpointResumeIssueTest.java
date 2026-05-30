package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerCheckpointResumeIssueTest {

    @Test
    void publishesStableMetadataMap() {
        TrainerCheckpointResumeIssue issue = new TrainerCheckpointResumeIssue(
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT,
                " optimizer ",
                " optimizer checkpoint artifact is missing on resume ");

        assertEquals("missing-artifact", issue.kindValue());
        assertEquals("checkpoint-resume-artifact-missing", issue.code());
        assertEquals("warning", issue.severity());
        assertEquals(false, issue.blocking());
        assertEquals("restore the missing artifact or resume without checkpoint state", issue.action());
        assertEquals(
                Map.of(
                        "kind", "missing-artifact",
                        "code", "checkpoint-resume-artifact-missing",
                        "severity", "warning",
                        "blocking", false,
                        "artifact", "optimizer",
                        "message", "optimizer checkpoint artifact is missing on resume",
                        "action", "restore the missing artifact or resume without checkpoint state"),
                issue.toMetadataMap());
    }

    @Test
    void omitsBlankOptionalFieldsAndKeepsMapImmutable() {
        TrainerCheckpointResumeIssue issue = new TrainerCheckpointResumeIssue(
                TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH,
                " ",
                null);

        Map<String, Object> metadata = issue.toMetadataMap();

        assertFalse(metadata.containsKey("artifact"));
        assertFalse(metadata.containsKey("message"));
        assertEquals("compatibility-mismatch", issue.value("kind"));
        assertEquals("error", issue.value("severity"));
        assertEquals(true, issue.blocking());
        assertEquals(true, metadata.get("blocking"));
        assertEquals(null, issue.value("artifact"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("artifact", "model"));
    }
}
