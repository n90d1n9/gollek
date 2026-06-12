package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TrainingReportDocumentArtifactDescriptorTest {
    private static final String JSON_SHA =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static final String MARKDOWN_SHA =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    @Test
    void exposesStableJsonAndMarkdownArtifactMap() {
        TrainingReportDocumentArtifactDescriptor descriptor = new TrainingReportDocumentArtifactDescriptor(
                Path.of("reports"),
                Path.of("reports/report.json"),
                Path.of("reports/report.md"),
                JSON_SHA,
                MARKDOWN_SHA);

        assertEquals(descriptor.directory().toString(), descriptor.toMap().get("directory"));
        assertEquals(descriptor.jsonFile().toString(), descriptor.toMap().get("jsonFile"));
        assertEquals(descriptor.markdownFile().toString(), descriptor.toMap().get("markdownFile"));
        assertEquals(JSON_SHA, descriptor.toMap().get("jsonSha256"));
        assertEquals(MARKDOWN_SHA, descriptor.toMap().get("markdownSha256"));
    }

    @Test
    void comparesExpectedChecksumsWithNormalizedValues() {
        TrainingReportDocumentArtifactDescriptor descriptor = new TrainingReportDocumentArtifactDescriptor(
                Path.of("reports"),
                Path.of("reports/report.json"),
                Path.of("reports/report.md"),
                JSON_SHA.toUpperCase(),
                MARKDOWN_SHA.toUpperCase());

        TrainingReportDocumentArtifactDescriptor.ChecksumMatch match =
                descriptor.checksumMatch(JSON_SHA.toUpperCase(), MARKDOWN_SHA.toUpperCase());

        assertTrue(match.passed());
        assertTrue(match.jsonMatches());
        assertTrue(match.markdownMatches());
        assertEquals(JSON_SHA, match.expectedJsonSha256());
        assertEquals(MARKDOWN_SHA, match.expectedMarkdownSha256());
    }

    @Test
    void reportsMismatchedChecksums() {
        TrainingReportDocumentArtifactDescriptor descriptor = new TrainingReportDocumentArtifactDescriptor(
                Path.of("reports"),
                Path.of("reports/report.json"),
                Path.of("reports/report.md"),
                JSON_SHA,
                MARKDOWN_SHA);

        TrainingReportDocumentArtifactDescriptor.ChecksumMatch match =
                descriptor.checksumMatch("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", null);

        assertFalse(match.passed());
        assertFalse(match.jsonMatches());
        assertTrue(match.markdownMatches());
    }

    @Test
    void rejectsInvalidShaValues() {
        assertThrows(IllegalArgumentException.class, () -> new TrainingReportDocumentArtifactDescriptor(
                Path.of("reports"),
                Path.of("reports/report.json"),
                Path.of("reports/report.md"),
                "not-a-sha",
                MARKDOWN_SHA));
    }
}
