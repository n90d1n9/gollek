package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Persists runtime profile budget gate results as CI-uploadable artifacts.
 */
public final class TrainingReportRuntimeProfileBudgetGateArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "runtime-profile-budget-gate.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "runtime-profile-budget-gate.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "runtime-profile-budget-gate.junit.xml";

    private TrainingReportRuntimeProfileBudgetGateArtifacts() {
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
            TrainingReportRuntimeProfileBudgetGate.Result result) {
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
                return "Runtime profile budget gate artifacts verified for " + inspection.directory() + ".";
            }
            return "Runtime profile budget gate artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("message", message());
            map.put("artifact", artifact().toMap());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            map.put("failures", failures);
            return Map.copyOf(map);
        }

        public TrainingReportArtifactDescriptor artifact() {
            return TrainingReportArtifactDescriptor.withoutManifest(
                    inspection.directory(),
                    inspection.jsonFile(),
                    inspection.markdownFile(),
                    inspection.junitXmlFile(),
                    inspection.jsonSha256(),
                    inspection.markdownSha256(),
                    inspection.junitXmlSha256());
        }

        public Map<String, Object> artifactMap() {
            return artifact().toMap();
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportRuntimeProfileBudgetGate.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportRuntimeProfileBudgetGate.Result result,
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
            TrainingReportRuntimeProfileBudgetGate.Result result) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportRuntimeProfileBudgetGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, resolvedResult.markdown());
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, resolvedResult.junitXml());
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        return new ArtifactBundle(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256(),
                resolvedResult);
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        TrainingReportRuntimeProfileBudgetGate.Result result = resultFromJsonFile(jsonFile);
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
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                readJsonMap(json, resolvedJsonFile),
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

    public static ArtifactVerification verify(Path directory) throws IOException {
        return verify(read(directory), null, null, null);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        TrainingReportArtifactDescriptor.ChecksumMatch checksums =
                inspection.artifact().checksumMatch(
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
        TrainingReportRuntimeProfileBudgetGate.Result result = null;
        try {
            result = resultFromMap(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Runtime profile budget gate JSON cannot be rendered: " + error.getMessage());
        }
        boolean markdownMatchesJson = false;
        boolean junitXmlMatchesJson = false;
        if (result != null) {
            markdownMatchesJson = result.markdown().equals(inspection.markdown());
            junitXmlMatchesJson = result.junitXml().equals(inspection.junitXml());
            if (!markdownMatchesJson) {
                failures.add("Markdown report does not match JSON: " + inspection.markdownFile());
            }
            if (!junitXmlMatchesJson) {
                failures.add("JUnit XML report does not match JSON: " + inspection.junitXmlFile());
            }
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

    private static TrainingReportRuntimeProfileBudgetGate.Result resultFromJsonFile(Path jsonFile) throws IOException {
        return resultFromMap(readJsonMap(Files.readString(jsonFile, StandardCharsets.UTF_8), jsonFile));
    }

    private static Map<String, Object> readJsonMap(String json, Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Runtime profile budget gate JSON must be an object: " + jsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static TrainingReportRuntimeProfileBudgetGate.Result resultFromMap(Map<String, Object> map) {
        TrainingReportRuntimeProfileBudgetGate.Policy policy = policyFromMap(mapValue(map, "policy"));
        boolean available = booleanValue(map.get("available"));
        List<TrainingReportRuntimeProfileBudgetGate.Finding> findings = new ArrayList<>();
        Object rawFindings = map.get("findings");
        if (rawFindings instanceof Iterable<?> iterable) {
            for (Object rawFinding : iterable) {
                findings.add(findingFromMap(immutableStringKeyMap(asMap(rawFinding, "finding"))));
            }
        }
        return new TrainingReportRuntimeProfileBudgetGate.Result(
                policy,
                available,
                findings,
                mapValue(map, "runtimeProfile"),
                mapValue(map, "actionPlan"));
    }

    private static TrainingReportRuntimeProfileBudgetGate.Policy policyFromMap(Map<String, Object> map) {
        return new TrainingReportRuntimeProfileBudgetGate.Policy(
                optionalDouble(map.get("maxPrimaryGroupPercent")).orElse(80.0),
                optionalDouble(map.get("maxPrimaryHotspotPercent")).orElse(60.0),
                optionalDouble(map.get("maxPrimaryHotspotTotalMillis")).orElse(Double.POSITIVE_INFINITY),
                optionalDouble(map.get("maxInputBalancePercent")).orElse(60.0),
                optionalDouble(map.get("maxOptimizerBalancePercent")).orElse(50.0),
                optionalDouble(map.get("maxValidationBalancePercent")).orElse(60.0),
                optionalDouble(map.get("maxWallClockOverheadPercent")).orElse(35.0),
                optionalDouble(map.get("maxWallClockOverheadMillis")).orElse(Double.POSITIVE_INFINITY));
    }

    private static TrainingReportRuntimeProfileBudgetGate.Finding findingFromMap(Map<String, Object> map) {
        return new TrainingReportRuntimeProfileBudgetGate.Finding(
                String.valueOf(map.getOrDefault("code", "")),
                String.valueOf(map.getOrDefault("severity", "warning")),
                String.valueOf(map.getOrDefault("message", "")),
                String.valueOf(map.getOrDefault("action", "")),
                mapValue(map, "evidence"));
    }

    private static Map<?, ?> asMap(Object value, String fieldName) {
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException(fieldName + " must be an object");
    }

    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value instanceof Map<?, ?> map) {
            return immutableStringKeyMap(map);
        }
        return Map.of();
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
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Collections.unmodifiableMap(values);
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
