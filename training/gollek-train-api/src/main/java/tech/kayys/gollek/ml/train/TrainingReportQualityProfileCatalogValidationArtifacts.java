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
 * Persists quality-profile catalog validation results for CI, release reviews, and audit trails.
 */
public final class TrainingReportQualityProfileCatalogValidationArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "training-report-quality-profile-catalog-validation.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "training-report-quality-profile-catalog-validation.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME =
            "training-report-quality-profile-catalog-validation.junit.xml";

    private TrainingReportQualityProfileCatalogValidationArtifacts() {
    }

    public record Options(String jsonFileName, String markdownFileName, String junitXmlFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            junitXmlFileName = normalizeFileName(junitXmlFileName, DEFAULT_JUNIT_XML_FILE_NAME, "junitXmlFileName");
        }

        public static Options defaults() {
            return new Options(DEFAULT_JSON_FILE_NAME, DEFAULT_MARKDOWN_FILE_NAME, DEFAULT_JUNIT_XML_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            TrainingReportQualityProfileCatalogValidator.Result result) {
        public ArtifactBundle {
            directory = normalize(directory, "directory");
            jsonFile = normalize(jsonFile, "jsonFile");
            markdownFile = normalize(markdownFile, "markdownFile");
            junitXmlFile = normalize(junitXmlFile, "junitXmlFile");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean passed() {
            return result.passed();
        }

        public TrainingReportArtifactDescriptor artifact() {
            return TrainingReportArtifactDescriptor.withoutManifest(
                    directory,
                    jsonFile,
                    markdownFile,
                    junitXmlFile,
                    jsonSha256,
                    markdownSha256,
                    junitXmlSha256);
        }

        public Map<String, Object> artifactMap() {
            return artifact().toMap();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>(artifactMap());
            map.put("artifact", artifactMap());
            map.put("passed", passed());
            map.put("result", result.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Map<String, Object> result,
            String markdown,
            String junitXml,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256) {
        public ArtifactInspection {
            directory = normalize(directory, "directory");
            jsonFile = normalize(jsonFile, "jsonFile");
            markdownFile = normalize(markdownFile, "markdownFile");
            junitXmlFile = normalize(junitXmlFile, "junitXmlFile");
            result = TrainingReportMapValues.immutableMap(
                    Objects.requireNonNull(result, "result must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
        }

        public TrainingReportQualityProfileCatalogValidator.Result parsedResult() {
            return TrainingReportQualityProfileCatalogValidator.Result.fromMap(result);
        }

        public TrainingReportArtifactDescriptor artifact() {
            return TrainingReportArtifactDescriptor.withoutManifest(
                    directory,
                    jsonFile,
                    markdownFile,
                    junitXmlFile,
                    jsonSha256,
                    markdownSha256,
                    junitXmlSha256);
        }

        public Map<String, Object> artifactMap() {
            return artifact().toMap();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>(artifactMap());
            map.put("artifact", artifactMap());
            map.put("result", result);
            map.put("markdown", markdown);
            map.put("junitXml", junitXml);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean junitXmlSha256Matches,
            boolean markdownMatchesJson,
            boolean junitXmlMatchesJson,
            List<String> failures) {
        public ArtifactVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonSha256 = normalizeChecksum(expectedJsonSha256);
            expectedMarkdownSha256 = normalizeChecksum(expectedMarkdownSha256);
            expectedJunitXmlSha256 = normalizeChecksum(expectedJunitXmlSha256);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public String message() {
            if (passed()) {
                return "Training report quality profile catalog validation artifacts verified for "
                        + inspection.directory() + ".";
            }
            return "Training report quality profile catalog validation artifact verification failed: "
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
            map.put("artifact", artifactMap());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            if (expectedJunitXmlSha256 != null) {
                map.put("expectedJunitXmlSha256", expectedJunitXmlSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }

        public TrainingReportArtifactDescriptor artifact() {
            return inspection.artifact();
        }

        public Map<String, Object> artifactMap() {
            return artifact().toMap();
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalogValidator.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalogValidator.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return writeFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()),
                result);
    }

    public static ArtifactBundle writeFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            TrainingReportQualityProfileCatalogValidator.Result result) throws IOException {
        TrainingReportQualityProfileCatalogValidator.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Path resolvedJsonFile = normalize(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalize(markdownFile, "markdownFile");
        Path resolvedJunitXmlFile = normalize(junitXmlFile, "junitXmlFile");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, resolvedResult.markdown());
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, resolvedResult.junitXml());
        return bundle(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile, resolvedResult);
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        TrainingReportQualityProfileCatalogValidator.Result result = readResult(jsonFile);
        TrainerCheckpointIO.writeStringAtomically(markdownFile, result.markdown());
        TrainerCheckpointIO.writeStringAtomically(junitXmlFile, result.junitXml());
        return bundle(jsonFile, markdownFile, junitXmlFile, result);
    }

    public static ArtifactInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ArtifactInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = normalize(directory, "directory");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()));
    }

    public static ArtifactInspection readFiles(Path jsonFile, Path markdownFile, Path junitXmlFile) throws IOException {
        Path resolvedJsonFile = normalize(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalize(markdownFile, "markdownFile");
        Path resolvedJunitXmlFile = normalize(junitXmlFile, "junitXmlFile");
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality profile catalog validation JSON must be an object: "
                    + resolvedJsonFile);
        }
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String junitXml = Files.readString(resolvedJunitXmlFile, StandardCharsets.UTF_8);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                TrainingReportMapValues.immutableMap(map),
                markdown,
                junitXml,
                TrainingReportArtifactFingerprint.of(resolvedJsonFile).sha256(),
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile).sha256(),
                TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile).sha256());
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(bundle.jsonFile(), bundle.markdownFile(), bundle.junitXmlFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.junitXmlSha256());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return verify(read(directory), expectedJsonSha256, expectedMarkdownSha256, null);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        TrainingReportArtifactDescriptor.ChecksumMatch checksums = inspection.artifact().checksumMatch(
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
        TrainingReportQualityProfileCatalogValidator.Result result = inspection.parsedResult();
        boolean markdownMatchesJson = result.markdown().equals(inspection.markdown());
        boolean junitXmlMatchesJson = result.junitXml().equals(inspection.junitXml());

        List<String> failures = new ArrayList<>();
        if (!checksums.jsonMatches()) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!checksums.markdownMatches()) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!checksums.junitXmlMatches()) {
            failures.add("JUnit XML checksum mismatch for " + inspection.junitXmlFile());
        }
        if (!markdownMatchesJson) {
            failures.add("Markdown validation report does not match JSON: " + inspection.markdownFile());
        }
        if (!junitXmlMatchesJson) {
            failures.add("JUnit XML validation report does not match JSON: " + inspection.junitXmlFile());
        }
        return new ArtifactVerification(
                inspection,
                checksums.expectedJsonSha256(),
                checksums.expectedMarkdownSha256(),
                checksums.expectedJunitXmlSha256(),
                checksums.jsonMatches(),
                checksums.markdownMatches(),
                checksums.junitXmlMatches(),
                markdownMatchesJson,
                junitXmlMatchesJson,
                failures);
    }

    private static ArtifactBundle bundle(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            TrainingReportQualityProfileCatalogValidator.Result result) throws IOException {
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(junitXmlFile);
        return new ArtifactBundle(
                commonDirectory(jsonFile, markdownFile),
                jsonFile,
                markdownFile,
                junitXmlFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256(),
                result);
    }

    private static TrainingReportQualityProfileCatalogValidator.Result readResult(Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(Files.readString(normalize(jsonFile, "jsonFile"), StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality profile catalog validation JSON must be an object: " + jsonFile);
        }
        return TrainingReportQualityProfileCatalogValidator.Result.fromMap(TrainingReportMapValues.immutableMap(map));
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
