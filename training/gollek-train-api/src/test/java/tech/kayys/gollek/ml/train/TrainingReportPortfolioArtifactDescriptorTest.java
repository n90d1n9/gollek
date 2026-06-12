package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportPortfolioArtifactDescriptorTest {
    private static final String JSON_SHA = "a".repeat(64);
    private static final String MARKDOWN_SHA = "b".repeat(64);
    private static final String LEADERBOARD_SHA = "c".repeat(64);
    private static final String METRICS_SHA = "d".repeat(64);
    private static final String FINDINGS_SHA = "e".repeat(64);

    @Test
    void exposesStableMapForPortfolioArtifactBundle() {
        TrainingReportPortfolioArtifactDescriptor descriptor = descriptor();
        Map<String, Object> map = descriptor.toMap();

        assertEquals(Path.of("portfolio").toAbsolutePath().normalize().toString(), map.get("directory"));
        assertEquals(Path.of("portfolio/export.json").toAbsolutePath().normalize().toString(), map.get("jsonFile"));
        assertEquals(JSON_SHA, map.get("jsonSha256"));
        assertEquals(FINDINGS_SHA, map.get("comparisonFindingsCsvSha256"));
    }

    @Test
    void comparesExpectedChecksumsWithOptionalInputs() {
        TrainingReportPortfolioArtifactDescriptor.ChecksumMatch allMatched = descriptor()
                .checksumMatch(
                        JSON_SHA.toUpperCase(),
                        MARKDOWN_SHA,
                        LEADERBOARD_SHA,
                        METRICS_SHA,
                        FINDINGS_SHA);
        TrainingReportPortfolioArtifactDescriptor.ChecksumMatch optionalMatched = descriptor()
                .checksumMatch(JSON_SHA, null, null, null, null);
        TrainingReportPortfolioArtifactDescriptor.ChecksumMatch mismatched = descriptor()
                .checksumMatch(JSON_SHA, MARKDOWN_SHA, "f".repeat(64), METRICS_SHA, FINDINGS_SHA);

        assertTrue(allMatched.passed());
        assertTrue(optionalMatched.passed());
        assertFalse(mismatched.passed());
        assertFalse(mismatched.leaderboardCsvMatches());
        assertEquals("f".repeat(64), mismatched.toMap().get("expectedLeaderboardCsvSha256"));
    }

    @Test
    void rejectsInvalidShaValues() {
        assertThrows(IllegalArgumentException.class, () -> new TrainingReportPortfolioArtifactDescriptor(
                Path.of("portfolio"),
                Path.of("portfolio/export.json"),
                Path.of("portfolio/export.md"),
                Path.of("portfolio/leaderboard.csv"),
                Path.of("portfolio/metrics.csv"),
                Path.of("portfolio/findings.csv"),
                "not-a-sha",
                MARKDOWN_SHA,
                LEADERBOARD_SHA,
                METRICS_SHA,
                FINDINGS_SHA));
        assertThrows(IllegalArgumentException.class, () -> descriptor().checksumMatch(JSON_SHA, "bad", null, null, null));
    }

    private static TrainingReportPortfolioArtifactDescriptor descriptor() {
        return new TrainingReportPortfolioArtifactDescriptor(
                Path.of("portfolio"),
                Path.of("portfolio/export.json"),
                Path.of("portfolio/export.md"),
                Path.of("portfolio/leaderboard.csv"),
                Path.of("portfolio/metrics.csv"),
                Path.of("portfolio/findings.csv"),
                JSON_SHA,
                MARKDOWN_SHA,
                LEADERBOARD_SHA,
                METRICS_SHA,
                FINDINGS_SHA);
    }
}
