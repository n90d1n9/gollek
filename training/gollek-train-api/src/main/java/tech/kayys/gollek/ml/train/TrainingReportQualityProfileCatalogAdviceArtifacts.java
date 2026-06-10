package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Persists quality-profile catalog advisory recommendations as reviewable JSON and Markdown artifacts.
 */
public final class TrainingReportQualityProfileCatalogAdviceArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "training-report-quality-profile-catalog-advice.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "training-report-quality-profile-catalog-advice.md";

    private TrainingReportQualityProfileCatalogAdviceArtifacts() {
    }

    public record Options(String jsonFileName, String markdownFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
        }

        public static Options defaults() {
            return new Options(DEFAULT_JSON_FILE_NAME, DEFAULT_MARKDOWN_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            String jsonSha256,
            String markdownSha256,
            TrainingReportQualityProfileCatalogAdvisor.Result result) {
        public ArtifactBundle {
            directory = normalize(directory, "directory");
            jsonFile = normalize(jsonFile, "jsonFile");
            markdownFile = normalize(markdownFile, "markdownFile");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean readyForCi() {
            return result.readyForCi();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("readyForCi", readyForCi());
            map.put("result", result.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Map<String, Object> result,
            String markdown,
            String jsonSha256,
            String markdownSha256) {
        public ArtifactInspection {
            directory = normalize(directory, "directory");
            jsonFile = normalize(jsonFile, "jsonFile");
            markdownFile = normalize(markdownFile, "markdownFile");
            result = TrainingReportMapValues.immutableMap(
                    Objects.requireNonNull(result, "result must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
        }

        public TrainingReportQualityProfileCatalogAdvisor.Result parsedResult() {
            return TrainingReportQualityProfileCatalogAdvisor.Result.fromMap(result);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("result", result);
            map.put("markdown", markdown);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean markdownMatchesJson,
            List<String> failures) {
        public ArtifactVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonSha256 = normalizeChecksum(expectedJsonSha256);
            expectedMarkdownSha256 = normalizeChecksum(expectedMarkdownSha256);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public String message() {
            if (passed()) {
                return "Training report quality profile catalog advice artifacts verified for "
                        + inspection.directory() + ".";
            }
            return "Training report quality profile catalog advice artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("message", message());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("markdownMatchesJson", markdownMatchesJson);
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalogAdvisor.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalogAdvisor.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return writeFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                result);
    }

    public static ArtifactBundle writeFiles(
            Path jsonFile,
            Path markdownFile,
            TrainingReportQualityProfileCatalogAdvisor.Result result) throws IOException {
        TrainingReportQualityProfileCatalogAdvisor.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Path resolvedJsonFile = normalize(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalize(markdownFile, "markdownFile");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, resolvedResult.markdown());
        return bundle(resolvedJsonFile, resolvedMarkdownFile, resolvedResult);
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        TrainingReportQualityProfileCatalogAdvisor.Result result = readResult(jsonFile);
        TrainerCheckpointIO.writeStringAtomically(markdownFile, result.markdown());
        return bundle(jsonFile, markdownFile, result);
    }

    public static ArtifactInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ArtifactInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()));
    }

    public static ArtifactInspection readFiles(Path jsonFile, Path markdownFile) throws IOException {
        Path resolvedJsonFile = normalize(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalize(markdownFile, "markdownFile");
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality profile catalog advice JSON must be an object: " + resolvedJsonFile);
        }
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                TrainingReportMapValues.immutableMap(map),
                markdown,
                TrainingReportArtifactFingerprint.of(resolvedJsonFile).sha256(),
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile).sha256());
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(readFiles(bundle.jsonFile(), bundle.markdownFile()), bundle.jsonSha256(), bundle.markdownSha256());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return verify(read(directory), expectedJsonSha256, expectedMarkdownSha256);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        boolean jsonMatches = normalizedJsonSha == null || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean markdownMatchesJson = inspection.parsedResult().markdown().equals(inspection.markdown());
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!markdownMatchesJson) {
            failures.add("Markdown advice report does not match JSON: " + inspection.markdownFile());
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                jsonMatches,
                markdownMatches,
                markdownMatchesJson,
                failures);
    }

    private static ArtifactBundle bundle(
            Path jsonFile,
            Path markdownFile,
            TrainingReportQualityProfileCatalogAdvisor.Result result) throws IOException {
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);
        return new ArtifactBundle(
                commonDirectory(jsonFile, markdownFile),
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                result);
    }

    private static TrainingReportQualityProfileCatalogAdvisor.Result readResult(Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(Files.readString(normalize(jsonFile, "jsonFile"), StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality profile catalog advice JSON must be an object: " + jsonFile);
        }
        return TrainingReportQualityProfileCatalogAdvisor.Result.fromMap(TrainingReportMapValues.immutableMap(map));
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static Path normalize(Path path, String fieldName) {
        return Objects.requireNonNull(path, fieldName + " must not be null").toAbsolutePath().normalize();
    }

    private static Path commonDirectory(Path jsonFile, Path markdownFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        if (jsonParent != null && jsonParent.equals(markdownParent)) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    private static void requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(java.util.Locale.ROOT);
    }
}
