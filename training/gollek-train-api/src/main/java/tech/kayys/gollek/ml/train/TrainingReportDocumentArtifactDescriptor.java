package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Stable identity for a JSON plus Markdown trainer report artifact pair.
 */
public record TrainingReportDocumentArtifactDescriptor(
        Path directory,
        Path jsonFile,
        Path markdownFile,
        String jsonSha256,
        String markdownSha256) {
    public TrainingReportDocumentArtifactDescriptor {
        directory = normalizePath(directory, "directory");
        jsonFile = normalizePath(jsonFile, "jsonFile");
        markdownFile = normalizePath(markdownFile, "markdownFile");
        jsonSha256 = TrainingReportSha256.require(jsonSha256, "jsonSha256");
        markdownSha256 = TrainingReportSha256.require(markdownSha256, "markdownSha256");
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("directory", directory.toString());
        map.put("jsonFile", jsonFile.toString());
        map.put("markdownFile", markdownFile.toString());
        map.put("jsonSha256", jsonSha256);
        map.put("markdownSha256", markdownSha256);
        return Map.copyOf(map);
    }

    public ChecksumMatch checksumMatch(String expectedJsonSha256, String expectedMarkdownSha256) {
        String normalizedJsonSha = TrainingReportSha256.normalizeOptional(expectedJsonSha256, "expectedJsonSha256");
        String normalizedMarkdownSha =
                TrainingReportSha256.normalizeOptional(expectedMarkdownSha256, "expectedMarkdownSha256");
        return new ChecksumMatch(
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJsonSha == null || normalizedJsonSha.equals(jsonSha256),
                normalizedMarkdownSha == null || normalizedMarkdownSha.equals(markdownSha256));
    }

    /**
     * Result of comparing expected two-file report checksums with current checksums.
     */
    public record ChecksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            boolean jsonMatches,
            boolean markdownMatches) {
        public boolean passed() {
            return jsonMatches && markdownMatches;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonMatches", jsonMatches);
            map.put("markdownMatches", markdownMatches);
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            return Map.copyOf(map);
        }
    }

    private static Path normalizePath(Path value, String fieldName) {
        return Objects.requireNonNull(value, fieldName + " must not be null")
                .toAbsolutePath()
                .normalize();
    }

}
