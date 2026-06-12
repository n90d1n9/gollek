package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Persists promotion gate results as CI-uploadable artifacts.
 */
public final class TrainingReportPromotionGateArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "promotion-gate.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "promotion-gate.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "promotion-gate.junit.xml";

    private TrainingReportPromotionGateArtifacts() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName,
            String junitXmlFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            junitXmlFileName = normalizeFileName(junitXmlFileName, DEFAULT_JUNIT_XML_FILE_NAME, "junitXmlFileName");
        }

        public static Options defaults() {
            return new Options(
                    DEFAULT_JSON_FILE_NAME,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    DEFAULT_JUNIT_XML_FILE_NAME);
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
            TrainingReportPromotionGate.Result result) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (jsonSha256 == null || jsonSha256.isBlank()) {
                throw new IllegalArgumentException("jsonSha256 must not be blank");
            }
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
            if (junitXmlSha256 == null || junitXmlSha256.isBlank()) {
                throw new IllegalArgumentException("junitXmlSha256 must not be blank");
            }
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean passed() {
            return result.passed();
        }

        public boolean promotable() {
            return result.promotable();
        }

        public void requirePassed() {
            result.requirePassed();
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
            map.put("promotable", promotable());
            map.put("decision", result.decision().toMap());
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
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            result = Map.copyOf(Objects.requireNonNull(result, "result must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            if (jsonSha256 == null || jsonSha256.isBlank()) {
                throw new IllegalArgumentException("jsonSha256 must not be blank");
            }
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
            if (junitXmlSha256 == null || junitXmlSha256.isBlank()) {
                throw new IllegalArgumentException("junitXmlSha256 must not be blank");
            }
        }

        public boolean passed() {
            Object value = result.get("passed");
            return value instanceof Boolean bool && bool.booleanValue();
        }

        public boolean promotable() {
            Object value = result.get("promotable");
            return value instanceof Boolean bool && bool.booleanValue();
        }

        public String decisionStatus() {
            return stringValue(decision().get("status"), "UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            String candidate = stringValue(decision().get("candidate"), "");
            return candidate.isBlank() ? Optional.empty() : Optional.of(candidate);
        }

        public Map<String, Object> decision() {
            Object decision = result.get("decision");
            if (!(decision instanceof Map<?, ?> map)) {
                return Map.of();
            }
            return immutableStringKeyMap(map);
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
            map.put("promotable", promotable());
            map.put("decisionStatus", decisionStatus());
            decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
            map.put("result", result);
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
            boolean junitXmlWellFormed,
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

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Promotion gate artifacts verified for " + inspection.directory() + ".";
            }
            return "Promotion gate artifact verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("artifact", artifactMap());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            map.put("actualJsonSha256", inspection.jsonSha256());
            map.put("actualMarkdownSha256", inspection.markdownSha256());
            map.put("actualJunitXmlSha256", inspection.junitXmlSha256());
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
            TrainingReportPromotionGate.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPromotionGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        Map<String, Object> resultMap = resolvedResult.toMap();

        TrainerCheckpointIO.writeStringAtomically(jsonFile, TrainerJson.toJson(resultMap) + "\n");
        TrainerCheckpointIO.writeStringAtomically(
                markdownFile,
                TrainingReportPromotionGateArtifactReports.renderMarkdown(resultMap));
        TrainerCheckpointIO.writeStringAtomically(
                junitXmlFile,
                TrainingReportPromotionGateArtifactReports.renderJunitXml(resultMap));
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(junitXmlFile);

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256(),
                resolvedResult);
    }

    public static ArtifactInspection refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactInspection refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        Map<String, Object> result = readResult(jsonFile);

        TrainerCheckpointIO.writeStringAtomically(
                markdownFile,
                TrainingReportPromotionGateArtifactReports.renderMarkdown(result));
        TrainerCheckpointIO.writeStringAtomically(
                junitXmlFile,
                TrainingReportPromotionGateArtifactReports.renderJunitXml(result));

        return readFiles(jsonFile, markdownFile, junitXmlFile);
    }

    public static ArtifactInspection refreshDerivedFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        Map<String, Object> result = readResult(resolvedJsonFile);

        TrainerCheckpointIO.writeStringAtomically(
                resolvedMarkdownFile,
                TrainingReportPromotionGateArtifactReports.renderMarkdown(result));
        TrainerCheckpointIO.writeStringAtomically(
                resolvedJunitXmlFile,
                TrainingReportPromotionGateArtifactReports.renderJunitXml(result));

        return readFiles(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile);
    }

    public static ArtifactInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ArtifactInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()));
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String junitXml = Files.readString(resolvedJunitXmlFile, StandardCharsets.UTF_8);
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Promotion gate JSON must be an object: " + resolvedJsonFile);
        }
        TrainingReportArtifactFingerprint jsonFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                immutableStringKeyMap(map),
                markdown,
                junitXml,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256());
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
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return verify(directory, expectedJsonSha256, expectedMarkdownSha256, expectedJunitXmlSha256, Options.defaults());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            Options options) throws IOException {
        return verify(read(directory, options), expectedJsonSha256, expectedMarkdownSha256, expectedJunitXmlSha256);
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
        boolean junitXmlWellFormed = TrainingReportXml.isWellFormed(inspection.junitXml());
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
        if (!junitXmlWellFormed) {
            failures.add("JUnit XML is not well-formed: " + inspection.junitXmlFile());
        }
        String renderedMarkdown = null;
        String renderedJunitXml = null;
        try {
            renderedMarkdown = TrainingReportPromotionGateArtifactReports.renderMarkdown(inspection.result());
            renderedJunitXml = TrainingReportPromotionGateArtifactReports.renderJunitXml(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Promotion gate JSON cannot be rendered for consistency checks: " + error.getMessage());
        }
        boolean markdownMatchesJson = renderedMarkdown != null && renderedMarkdown.equals(inspection.markdown());
        boolean junitXmlMatchesJson = renderedJunitXml != null && renderedJunitXml.equals(inspection.junitXml());
        if (renderedMarkdown != null && !markdownMatchesJson) {
            failures.add("Markdown report does not match promotion gate JSON: " + inspection.markdownFile());
        }
        if (renderedJunitXml != null && !junitXmlMatchesJson) {
            failures.add("JUnit XML report does not match promotion gate JSON: " + inspection.junitXmlFile());
        }
        return new ArtifactVerification(
                inspection,
                checksums.expectedJsonSha256(),
                checksums.expectedMarkdownSha256(),
                checksums.expectedJunitXmlSha256(),
                checksums.jsonMatches(),
                checksums.markdownMatches(),
                checksums.junitXmlMatches(),
                junitXmlWellFormed,
                markdownMatchesJson,
                junitXmlMatchesJson,
                failures);
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static Path commonDirectory(Path jsonFile, Path markdownFile, Path junitXmlFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        Path junitXmlParent = junitXmlFile.getParent();
        if (jsonParent != null && jsonParent.equals(markdownParent) && jsonParent.equals(junitXmlParent)) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), TrainingReportSnapshots.immutableSnapshot(entry.getValue()));
        }
        return Map.copyOf(values);
    }

    private static Map<String, Object> readResult(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Promotion gate JSON must be an object: " + resolvedJsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }

}
