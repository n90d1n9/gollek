package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Stable identity for the five-file trainer portfolio export artifact bundle.
 */
public record TrainingReportPortfolioArtifactDescriptor(
        Path directory,
        Path jsonFile,
        Path markdownFile,
        Path leaderboardCsvFile,
        Path comparisonMetricsCsvFile,
        Path comparisonFindingsCsvFile,
        String jsonSha256,
        String markdownSha256,
        String leaderboardCsvSha256,
        String comparisonMetricsCsvSha256,
        String comparisonFindingsCsvSha256) {
    public TrainingReportPortfolioArtifactDescriptor {
        directory = normalizePath(directory, "directory");
        jsonFile = normalizePath(jsonFile, "jsonFile");
        markdownFile = normalizePath(markdownFile, "markdownFile");
        leaderboardCsvFile = normalizePath(leaderboardCsvFile, "leaderboardCsvFile");
        comparisonMetricsCsvFile = normalizePath(comparisonMetricsCsvFile, "comparisonMetricsCsvFile");
        comparisonFindingsCsvFile = normalizePath(comparisonFindingsCsvFile, "comparisonFindingsCsvFile");
        jsonSha256 = TrainingReportSha256.require(jsonSha256, "jsonSha256");
        markdownSha256 = TrainingReportSha256.require(markdownSha256, "markdownSha256");
        leaderboardCsvSha256 = TrainingReportSha256.require(leaderboardCsvSha256, "leaderboardCsvSha256");
        comparisonMetricsCsvSha256 =
                TrainingReportSha256.require(comparisonMetricsCsvSha256, "comparisonMetricsCsvSha256");
        comparisonFindingsCsvSha256 =
                TrainingReportSha256.require(comparisonFindingsCsvSha256, "comparisonFindingsCsvSha256");
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("directory", directory.toString());
        map.put("jsonFile", jsonFile.toString());
        map.put("markdownFile", markdownFile.toString());
        map.put("leaderboardCsvFile", leaderboardCsvFile.toString());
        map.put("comparisonMetricsCsvFile", comparisonMetricsCsvFile.toString());
        map.put("comparisonFindingsCsvFile", comparisonFindingsCsvFile.toString());
        map.put("jsonSha256", jsonSha256);
        map.put("markdownSha256", markdownSha256);
        map.put("leaderboardCsvSha256", leaderboardCsvSha256);
        map.put("comparisonMetricsCsvSha256", comparisonMetricsCsvSha256);
        map.put("comparisonFindingsCsvSha256", comparisonFindingsCsvSha256);
        return Map.copyOf(map);
    }

    public ChecksumMatch checksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) {
        String normalizedJsonSha = TrainingReportSha256.normalizeOptional(expectedJsonSha256, "expectedJsonSha256");
        String normalizedMarkdownSha =
                TrainingReportSha256.normalizeOptional(expectedMarkdownSha256, "expectedMarkdownSha256");
        String normalizedLeaderboardSha =
                TrainingReportSha256.normalizeOptional(
                        expectedLeaderboardCsvSha256, "expectedLeaderboardCsvSha256");
        String normalizedMetricsSha =
                TrainingReportSha256.normalizeOptional(
                        expectedComparisonMetricsCsvSha256, "expectedComparisonMetricsCsvSha256");
        String normalizedFindingsSha =
                TrainingReportSha256.normalizeOptional(
                        expectedComparisonFindingsCsvSha256, "expectedComparisonFindingsCsvSha256");
        return new ChecksumMatch(
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedLeaderboardSha,
                normalizedMetricsSha,
                normalizedFindingsSha,
                normalizedJsonSha == null || normalizedJsonSha.equals(jsonSha256),
                normalizedMarkdownSha == null || normalizedMarkdownSha.equals(markdownSha256),
                normalizedLeaderboardSha == null || normalizedLeaderboardSha.equals(leaderboardCsvSha256),
                normalizedMetricsSha == null || normalizedMetricsSha.equals(comparisonMetricsCsvSha256),
                normalizedFindingsSha == null || normalizedFindingsSha.equals(comparisonFindingsCsvSha256));
    }

    /**
     * Result of comparing expected portfolio export checksums with current artifact checksums.
     */
    public record ChecksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256,
            boolean jsonMatches,
            boolean markdownMatches,
            boolean leaderboardCsvMatches,
            boolean comparisonMetricsCsvMatches,
            boolean comparisonFindingsCsvMatches) {
        public boolean passed() {
            return jsonMatches
                    && markdownMatches
                    && leaderboardCsvMatches
                    && comparisonMetricsCsvMatches
                    && comparisonFindingsCsvMatches;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonMatches", jsonMatches);
            map.put("markdownMatches", markdownMatches);
            map.put("leaderboardCsvMatches", leaderboardCsvMatches);
            map.put("comparisonMetricsCsvMatches", comparisonMetricsCsvMatches);
            map.put("comparisonFindingsCsvMatches", comparisonFindingsCsvMatches);
            putIfPresent(map, "expectedJsonSha256", expectedJsonSha256);
            putIfPresent(map, "expectedMarkdownSha256", expectedMarkdownSha256);
            putIfPresent(map, "expectedLeaderboardCsvSha256", expectedLeaderboardCsvSha256);
            putIfPresent(map, "expectedComparisonMetricsCsvSha256", expectedComparisonMetricsCsvSha256);
            putIfPresent(map, "expectedComparisonFindingsCsvSha256", expectedComparisonFindingsCsvSha256);
            return Map.copyOf(map);
        }
    }

    private static void putIfPresent(Map<String, Object> map, String key, String value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    private static Path normalizePath(Path value, String fieldName) {
        return Objects.requireNonNull(value, fieldName + " must not be null")
                .toAbsolutePath()
                .normalize();
    }

}
