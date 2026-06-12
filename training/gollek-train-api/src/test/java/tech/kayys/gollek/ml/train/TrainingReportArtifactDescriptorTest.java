package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TrainingReportArtifactDescriptorTest {
    private static final String JSON_SHA = "a".repeat(64);
    private static final String MARKDOWN_SHA = "b".repeat(64);
    private static final String JUNIT_SHA = "c".repeat(64);
    private static final String MANIFEST_SHA = "d".repeat(64);

    @Test
    void describesArtifactsWithoutManifest() {
        TrainingReportArtifactDescriptor descriptor = TrainingReportArtifactDescriptor.withoutManifest(
                Path.of("artifacts"),
                Path.of("artifacts/gate.json"),
                Path.of("artifacts/gate.md"),
                Path.of("artifacts/gate.junit.xml"),
                JSON_SHA.toUpperCase(),
                MARKDOWN_SHA,
                JUNIT_SHA);

        assertFalse(descriptor.hasManifest());
        assertEquals(JSON_SHA, descriptor.jsonSha256());
        assertEquals(false, descriptor.toMap().get("hasManifest"));
        assertFalse(descriptor.toMap().containsKey("manifestFile"));
        assertFalse(descriptor.toMap().containsKey("manifestSha256"));
    }

    @Test
    void matchesExpectedChecksumsWithManifest() {
        TrainingReportArtifactDescriptor descriptor = TrainingReportArtifactDescriptor.withManifest(
                Path.of("artifacts"),
                Path.of("artifacts/gate.json"),
                Path.of("artifacts/gate.md"),
                Path.of("artifacts/gate.junit.xml"),
                Path.of("artifacts/gate.manifest.json"),
                JSON_SHA,
                MARKDOWN_SHA,
                JUNIT_SHA,
                MANIFEST_SHA,
                true);

        TrainingReportArtifactDescriptor.ChecksumMatch match = descriptor.checksumMatch(
                JSON_SHA.toUpperCase(),
                MARKDOWN_SHA,
                JUNIT_SHA,
                MANIFEST_SHA);

        assertTrue(descriptor.hasManifest());
        assertTrue(match.passed());
        assertTrue(match.manifestMatches());
        assertEquals(JSON_SHA, match.expectedJsonSha256());
        assertEquals(MANIFEST_SHA, match.toMap().get("expectedManifestSha256"));
    }

    @Test
    void reportsChecksumMismatches() {
        TrainingReportArtifactDescriptor descriptor = TrainingReportArtifactDescriptor.withManifest(
                Path.of("artifacts"),
                Path.of("artifacts/gate.json"),
                Path.of("artifacts/gate.md"),
                Path.of("artifacts/gate.junit.xml"),
                Path.of("artifacts/gate.manifest.json"),
                JSON_SHA,
                MARKDOWN_SHA,
                JUNIT_SHA,
                MANIFEST_SHA,
                true);

        TrainingReportArtifactDescriptor.ChecksumMatch match = descriptor.checksumMatch(
                "1".repeat(64),
                MARKDOWN_SHA,
                "2".repeat(64),
                MANIFEST_SHA);

        assertFalse(match.passed());
        assertFalse(match.jsonMatches());
        assertTrue(match.markdownMatches());
        assertFalse(match.junitXmlMatches());
        assertTrue(match.manifestMatches());
    }

    @Test
    void rejectsInvalidChecksums() {
        assertThrows(IllegalArgumentException.class, () -> TrainingReportArtifactDescriptor.withoutManifest(
                Path.of("artifacts"),
                Path.of("artifacts/gate.json"),
                Path.of("artifacts/gate.md"),
                Path.of("artifacts/gate.junit.xml"),
                "not-a-sha",
                MARKDOWN_SHA,
                JUNIT_SHA));

        TrainingReportArtifactDescriptor descriptor = TrainingReportArtifactDescriptor.withoutManifest(
                Path.of("artifacts"),
                Path.of("artifacts/gate.json"),
                Path.of("artifacts/gate.md"),
                Path.of("artifacts/gate.junit.xml"),
                JSON_SHA,
                MARKDOWN_SHA,
                JUNIT_SHA);
        assertThrows(IllegalArgumentException.class,
                () -> descriptor.checksumMatch("still-not-a-sha", null, null));
    }
}
