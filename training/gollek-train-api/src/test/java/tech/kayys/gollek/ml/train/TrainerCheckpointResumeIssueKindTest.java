package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class TrainerCheckpointResumeIssueKindTest {

    @Test
    void exposesStablePublishedMetadataValues() {
        assertEquals("missing-artifact", TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.kind());
        assertEquals(
                "checkpoint-resume-artifact-missing",
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.code());
        assertEquals("warning", TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.severity());
        assertEquals(false, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.blocking());
        assertEquals(
                "restore the missing artifact or resume without checkpoint state",
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.action());

        assertEquals("compatibility-mismatch", TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.kind());
        assertEquals(
                "checkpoint-resume-compatibility-mismatch",
                TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.code());
        assertEquals("error", TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.severity());
        assertEquals(true, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.blocking());

        assertEquals("manifest-entry-missing", TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.kind());
        assertEquals(
                "checkpoint-resume-manifest-entry-missing",
                TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.code());
        assertEquals("warning", TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.severity());
        assertEquals(false, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.blocking());
    }

    @Test
    void formatsArtifactMessagesForIssueKindsThatOwnMessageSuffixes() {
        assertEquals(
                "optimizer checkpoint artifact is missing on resume",
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.artifactMessage(" optimizer "));
        assertEquals(
                "checkpoint artifact is missing on resume",
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.artifactMessage(" "));
        assertEquals(
                "runtime checkpoint is missing from checkpoint manifest",
                TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.artifactMessage("runtime"));
        assertEquals(
                "signature mismatch",
                TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.artifactMessage("signature mismatch"));
    }
}
