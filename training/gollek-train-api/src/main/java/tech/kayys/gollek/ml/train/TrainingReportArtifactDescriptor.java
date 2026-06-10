package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
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
        jsonSha256 = requireSha256(jsonSha256, "jsonSha256");
        markdownSha256 = requireSha256(markdownSha256, "markdownSha256");
        junitXmlSha256 = requireSha256(junitXmlSha256, "junitXmlSha256");
        manifestSha256 = normalizeOptionalSha256(manifestSha256, "manifestSha256");
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

    private static Path normalizePath(Path value, String fieldName) {
        return Objects.requireNonNull(value, fieldName + " must not be null")
                .toAbsolutePath()
                .normalize();
    }

    private static String normalizeOptionalSha256(String value, String fieldName) {
        return value == null || value.isBlank() ? null : requireSha256(value, fieldName);
    }

    private static String requireSha256(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (!normalized.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(fieldName + " must be a 64-character SHA-256 hex string");
        }
        return normalized;
    }
}
