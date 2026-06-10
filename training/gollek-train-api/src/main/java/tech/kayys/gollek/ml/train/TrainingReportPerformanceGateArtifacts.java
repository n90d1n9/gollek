package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

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
 * Persists trainer performance gate results as JSON, Markdown, and JUnit XML artifacts.
 */
public final class TrainingReportPerformanceGateArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "trainer-performance-gate.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "trainer-performance-gate.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "trainer-performance-gate.junit.xml";

    private TrainingReportPerformanceGateArtifacts() {
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
            TrainingReportPerformanceGate.Result result) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean passed() {
            return result.passed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlSha256);
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
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            result = immutableStringKeyMap(Objects.requireNonNull(result, "result must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
        }

        public boolean passed() {
            return booleanValue(result.get("passed"));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("passed", passed());
            map.put("result", result);
            map.put("markdown", markdown);
            map.put("junitXml", junitXml);
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlSha256);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean junitXmlSha256Matches,
            boolean junitXmlWellFormed,
            boolean markdownMatchesJson,
            boolean junitXmlMatchesJson,
            List<String> failures) {
        public ArtifactVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public String message() {
            if (passed()) {
                return "Trainer performance gate artifacts verified for " + inspection.directory() + ".";
            }
            return "Trainer performance gate artifact verification failed: " + String.join("; ", failures) + ".";
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
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            map.put("failures", failures);
            return Map.copyOf(map);
        }
    }

    public static ArtifactBundle write(Path directory, TrainingReportPerformanceGate.Result result)
            throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPerformanceGate.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
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
            TrainingReportPerformanceGate.Result result) throws IOException {
        TrainingReportPerformanceGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Path resolvedJsonFile = normalizePath(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalizePath(markdownFile, "markdownFile");
        Path resolvedJunitXmlFile = normalizePath(junitXmlFile, "junitXmlFile");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, resolvedResult.markdown());
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, resolvedResult.junitXml());
        return bundle(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile, resolvedResult);
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPerformanceGate.Result result =
                resultFromJsonFile(resolvedDirectory.resolve(resolvedOptions.jsonFileName()));
        return write(resolvedDirectory, result, resolvedOptions);
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

    public static ArtifactInspection readFiles(Path jsonFile, Path markdownFile, Path junitXmlFile) throws IOException {
        Path resolvedJsonFile = normalizePath(jsonFile, "jsonFile");
        Path resolvedMarkdownFile = normalizePath(markdownFile, "markdownFile");
        Path resolvedJunitXmlFile = normalizePath(junitXmlFile, "junitXmlFile");
        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String junitXml = Files.readString(resolvedJunitXmlFile, StandardCharsets.UTF_8);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                readJsonMap(json, resolvedJsonFile),
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

    public static ArtifactVerification verify(Path directory) throws IOException {
        return verify(read(directory), null, null, null);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<String> failures = new ArrayList<>();
        boolean jsonMatches = checksumMatches(expectedJsonSha256, inspection.jsonSha256());
        boolean markdownMatches = checksumMatches(expectedMarkdownSha256, inspection.markdownSha256());
        boolean junitXmlMatches = checksumMatches(expectedJunitXmlSha256, inspection.junitXmlSha256());
        boolean junitXmlWellFormed = TrainingReportXml.isWellFormed(inspection.junitXml());
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!junitXmlMatches) {
            failures.add("JUnit XML checksum mismatch for " + inspection.junitXmlFile());
        }
        if (!junitXmlWellFormed) {
            failures.add("JUnit XML is not well-formed: " + inspection.junitXmlFile());
        }
        TrainingReportPerformanceGate.Result result = null;
        try {
            result = resultFromMap(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Trainer performance gate JSON cannot be rendered: " + error.getMessage());
        }
        boolean markdownMatchesJson = result != null && result.markdown().equals(inspection.markdown());
        boolean junitXmlMatchesJson = result != null && result.junitXml().equals(inspection.junitXml());
        if (result != null && !markdownMatchesJson) {
            failures.add("Markdown report does not match JSON: " + inspection.markdownFile());
        }
        if (result != null && !junitXmlMatchesJson) {
            failures.add("JUnit XML report does not match JSON: " + inspection.junitXmlFile());
        }
        return new ArtifactVerification(
                inspection,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                junitXmlWellFormed,
                markdownMatchesJson,
                junitXmlMatchesJson,
                failures);
    }

    private static ArtifactBundle bundle(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            TrainingReportPerformanceGate.Result result) throws IOException {
        return new ArtifactBundle(
                commonDirectory(jsonFile, markdownFile, junitXmlFile),
                jsonFile,
                markdownFile,
                junitXmlFile,
                TrainingReportArtifactFingerprint.of(jsonFile).sha256(),
                TrainingReportArtifactFingerprint.of(markdownFile).sha256(),
                TrainingReportArtifactFingerprint.of(junitXmlFile).sha256(),
                result);
    }

    private static TrainingReportPerformanceGate.Result resultFromJsonFile(Path jsonFile) throws IOException {
        return resultFromMap(readJsonMap(Files.readString(jsonFile, StandardCharsets.UTF_8), jsonFile));
    }

    private static Map<String, Object> readJsonMap(String json, Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Trainer performance gate JSON must be an object: " + jsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static TrainingReportPerformanceGate.Result resultFromMap(Map<String, Object> map) {
        List<TrainingReportPerformanceGate.Finding> findings = new ArrayList<>();
        Object rawFindings = map.get("findings");
        if (rawFindings instanceof Iterable<?> iterable) {
            for (Object rawFinding : iterable) {
                findings.add(findingFromMap(immutableStringKeyMap(asMap(rawFinding))));
            }
        }
        return new TrainingReportPerformanceGate.Result(
                policyFromMap(mapValue(map, "policy")),
                booleanValue(map.get("available")),
                findings,
                mapValue(map, "acceleration"),
                mapValue(map, "throughput"));
    }

    private static TrainingReportPerformanceGate.Policy policyFromMap(Map<String, Object> map) {
        return new TrainingReportPerformanceGate.Policy(
                booleanValue(map.getOrDefault("failOnAcceleratorFallback", Boolean.TRUE)),
                optionalDouble(map.get("minTrainSamplesPerSecond")).orElse(1.0e-9),
                optionalDouble(map.get("maxValidationToTrainAverageBatchMillisRatio")).orElse(2.0));
    }

    private static TrainingReportPerformanceGate.Finding findingFromMap(Map<String, Object> map) {
        return new TrainingReportPerformanceGate.Finding(
                String.valueOf(map.getOrDefault("code", "")),
                String.valueOf(map.getOrDefault("severity", "warning")),
                String.valueOf(map.getOrDefault("message", "")),
                String.valueOf(map.getOrDefault("action", "")),
                mapValue(map, "evidence"));
    }

    private static Map<?, ?> asMap(Object value) {
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException("value must be an object");
    }

    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value instanceof Map<?, ?> map) {
            return immutableStringKeyMap(map);
        }
        return Map.of();
    }

    private static Path normalizePath(Path path, String fieldName) {
        return Objects.requireNonNull(path, fieldName + " must not be null").toAbsolutePath().normalize();
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
        if (jsonParent != null && jsonParent.equals(markdownFile.getParent()) && jsonParent.equals(junitXmlFile.getParent())) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
    }

    private static boolean checksumMatches(String expected, String actual) {
        String normalizedExpected = normalizeChecksum(expected);
        return normalizedExpected == null || normalizedExpected.equalsIgnoreCase(actual);
    }

    private static String normalizeChecksum(String value) {
        return value == null || value.isBlank() ? null : value.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private static void requireChecksum(String value, String fieldName) {
        if (normalizeChecksum(value) == null) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
    }
}
