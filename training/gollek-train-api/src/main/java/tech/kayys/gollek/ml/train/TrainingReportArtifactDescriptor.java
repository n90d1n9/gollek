package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Stable identity for a trainer report artifact bundle used by gates and CI dashboards.
 */
public record TrainingReportArtifactDescriptor(
        Path directory,
        Path jsonFile,
        Path markdownFile,
        Path junitXmlFile,
        Path manifestFile,
        String jsonSha256,
        String markdownSha256,
        String junitXmlSha256,
        String manifestSha256,
        boolean hasManifest) {
    public TrainingReportArtifactDescriptor {
        directory = normalizePath(directory, "directory");
        jsonFile = normalizePath(jsonFile, "jsonFile");
        markdownFile = normalizePath(markdownFile, "markdownFile");
        junitXmlFile = normalizePath(junitXmlFile, "junitXmlFile");
        manifestFile = manifestFile == null ? null : normalizePath(manifestFile, "manifestFile");
        jsonSha256 = TrainingReportSha256.require(jsonSha256, "jsonSha256");
        markdownSha256 = TrainingReportSha256.require(markdownSha256, "markdownSha256");
        junitXmlSha256 = TrainingReportSha256.require(junitXmlSha256, "junitXmlSha256");
        manifestSha256 = TrainingReportSha256.normalizeOptional(manifestSha256, "manifestSha256");
        hasManifest = hasManifest && manifestFile != null && manifestSha256 != null;
    }

    public static TrainingReportArtifactDescriptor withManifest(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path manifestFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            String manifestSha256,
            boolean hasManifest) {
        return new TrainingReportArtifactDescriptor(
                directory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                manifestFile,
                jsonSha256,
                markdownSha256,
                junitXmlSha256,
                manifestSha256,
                hasManifest);
    }

    public static TrainingReportArtifactDescriptor withoutManifest(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256) {
        return new TrainingReportArtifactDescriptor(
                directory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                null,
                jsonSha256,
                markdownSha256,
                junitXmlSha256,
                null,
                false);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("directory", directory.toString());
        map.put("jsonFile", jsonFile.toString());
        map.put("markdownFile", markdownFile.toString());
        map.put("junitXmlFile", junitXmlFile.toString());
        if (manifestFile != null) {
            map.put("manifestFile", manifestFile.toString());
        }
        map.put("jsonSha256", jsonSha256);
        map.put("markdownSha256", markdownSha256);
        map.put("junitXmlSha256", junitXmlSha256);
        if (manifestSha256 != null) {
            map.put("manifestSha256", manifestSha256);
        }
        map.put("hasManifest", hasManifest);
        return Map.copyOf(map);
    }

    public ChecksumMatch checksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        return checksumMatch(expectedJsonSha256, expectedMarkdownSha256, expectedJunitXmlSha256, null);
    }

    public ChecksumMatch checksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedManifestSha256) {
        String normalizedJsonSha = TrainingReportSha256.normalizeOptional(expectedJsonSha256, "expectedJsonSha256");
        String normalizedMarkdownSha =
                TrainingReportSha256.normalizeOptional(expectedMarkdownSha256, "expectedMarkdownSha256");
        String normalizedJunitXmlSha =
                TrainingReportSha256.normalizeOptional(expectedJunitXmlSha256, "expectedJunitXmlSha256");
        String normalizedManifestSha =
                TrainingReportSha256.normalizeOptional(expectedManifestSha256, "expectedManifestSha256");
        return new ChecksumMatch(
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJunitXmlSha,
                normalizedManifestSha,
                normalizedJsonSha == null || normalizedJsonSha.equals(jsonSha256),
                normalizedMarkdownSha == null || normalizedMarkdownSha.equals(markdownSha256),
                normalizedJunitXmlSha == null || normalizedJunitXmlSha.equals(junitXmlSha256),
                normalizedManifestSha == null || normalizedManifestSha.equals(manifestSha256));
    }

    /**
     * Result of comparing expected artifact checksums with a descriptor's current checksums.
     */
    public record ChecksumMatch(
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedManifestSha256,
            boolean jsonMatches,
            boolean markdownMatches,
            boolean junitXmlMatches,
            boolean manifestMatches) {
        public boolean passed() {
            return jsonMatches && markdownMatches && junitXmlMatches && manifestMatches;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonMatches", jsonMatches);
            map.put("markdownMatches", markdownMatches);
            map.put("junitXmlMatches", junitXmlMatches);
            map.put("manifestMatches", manifestMatches);
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            if (expectedJunitXmlSha256 != null) {
                map.put("expectedJunitXmlSha256", expectedJunitXmlSha256);
            }
            if (expectedManifestSha256 != null) {
                map.put("expectedManifestSha256", expectedManifestSha256);
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
