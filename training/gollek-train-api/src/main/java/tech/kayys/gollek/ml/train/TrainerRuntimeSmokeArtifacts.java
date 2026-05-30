package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.longValue;
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
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Persists and verifies trainer runtime smoke artifacts for CI/release checks.
 */
public final class TrainerRuntimeSmokeArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "trainer-runtime-smoke.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "trainer-runtime-smoke.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "trainer-runtime-smoke.junit.xml";

    private TrainerRuntimeSmokeArtifacts() {
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
            TrainerRuntimeSmoke.Result result) {
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

        public boolean smokePassed() {
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
            map.put("smokePassed", smokePassed());
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

        public boolean smokePassed() {
            return booleanValue(result.get("passed"));
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
            map.put("smokePassed", smokePassed());
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
                return "Trainer runtime smoke artifacts verified for " + inspection.directory() + ".";
            }
            return "Trainer runtime smoke artifact verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
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
    }

    public static ArtifactBundle write(Path directory, TrainerRuntimeSmoke.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainerRuntimeSmoke.Result result,
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
            TrainerRuntimeSmoke.Result result) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainerRuntimeSmoke.Result resolvedResult = Objects.requireNonNull(result, "result must not be null");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, TrainerRuntimeSmoke.renderMarkdown(resolvedResult));
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, TrainerRuntimeSmoke.renderJUnitXml(resolvedResult));
        return new ArtifactBundle(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile),
                resolvedResult);
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

    public static ArtifactBundle refresh(Path directory) throws IOException {
        return refresh(directory, Options.defaults());
    }

    public static ArtifactBundle refresh(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return refreshFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()));
    }

    public static ArtifactBundle refreshFiles(
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
        TrainerRuntimeSmoke.Result result = resultFromJsonFile(resolvedJsonFile);
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, TrainerRuntimeSmoke.renderMarkdown(result));
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, TrainerRuntimeSmoke.renderJUnitXml(result));
        return new ArtifactBundle(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile),
                result);
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
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                readJsonMap(json, resolvedJsonFile),
                markdown,
                junitXml,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile));
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
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedJunitXmlSha = normalizeChecksum(expectedJunitXmlSha256);
        boolean jsonMatches = normalizedJsonSha == null || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean junitXmlMatches = normalizedJunitXmlSha == null
                || normalizedJunitXmlSha.equalsIgnoreCase(inspection.junitXmlSha256());
        boolean junitXmlWellFormed = TrainingReportXml.isWellFormed(inspection.junitXml());
        TrainerRuntimeSmoke.Result renderedResult = null;
        List<String> failures = new ArrayList<>();
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
        try {
            renderedResult = resultFromMap(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Trainer runtime smoke JSON cannot be rendered for consistency checks: " + error.getMessage());
        }
        boolean markdownMatchesJson = false;
        boolean junitXmlMatchesJson = false;
        if (renderedResult != null) {
            markdownMatchesJson = TrainerRuntimeSmoke.renderMarkdown(renderedResult).equals(inspection.markdown());
            junitXmlMatchesJson = TrainerRuntimeSmoke.renderJUnitXml(renderedResult).equals(inspection.junitXml());
            if (!markdownMatchesJson) {
                failures.add("Markdown report does not match smoke JSON: " + inspection.markdownFile());
            }
            if (!junitXmlMatchesJson) {
                failures.add("JUnit XML report does not match smoke JSON: " + inspection.junitXmlFile());
            }
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJunitXmlSha,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                junitXmlWellFormed,
                markdownMatchesJson,
                junitXmlMatchesJson,
                failures);
    }

    private static TrainerRuntimeSmoke.Result resultFromJsonFile(Path jsonFile) throws IOException {
        return resultFromMap(readJsonMap(Files.readString(jsonFile, StandardCharsets.UTF_8), jsonFile));
    }

    private static Map<String, Object> readJsonMap(String json, Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Trainer runtime smoke JSON must be an object: " + jsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static TrainerRuntimeSmoke.Result resultFromMap(Map<String, Object> map) {
        Path checkpointDir = Path.of(String.valueOf(map.get("checkpointDir")));
        TrainingSummary firstRun = summaryFromMap(mapValue(map, "firstRun"));
        TrainingSummary resumedRun = summaryFromMap(mapValue(map, "resumedRun"));
        List<TrainerRuntimeSmoke.Check> checks = new ArrayList<>();
        Object rawChecks = map.get("checks");
        if (rawChecks instanceof Iterable<?> iterable) {
            for (Object rawCheck : iterable) {
                checks.add(checkFromMap(immutableStringKeyMap(asMap(rawCheck, "check"))));
            }
        }
        return new TrainerRuntimeSmoke.Result(checkpointDir, firstRun, resumedRun, checks);
    }

    private static TrainingSummary summaryFromMap(Map<String, Object> map) {
        return new TrainingSummary(
                intValue(map.get("epochCount"), 0),
                optionalDouble(map.get("bestValidationLoss")).orElse(Double.NaN),
                intValue(map.get("bestValidationEpoch"), 0),
                nullableDouble(map.get("latestTrainLoss")),
                nullableDouble(map.get("latestValidationLoss")),
                longValue(map.get("durationMs"), 0L),
                mapValue(map, "metadata"));
    }

    private static TrainerRuntimeSmoke.Check checkFromMap(Map<String, Object> map) {
        return new TrainerRuntimeSmoke.Check(
                String.valueOf(map.get("name")),
                booleanValue(map.get("passed")),
                String.valueOf(map.getOrDefault("detail", "")));
    }

    private static Double nullableDouble(Object value) {
        return optionalDouble(value).isPresent() ? optionalDouble(value).getAsDouble() : null;
    }

    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        return immutableStringKeyMap(asMap(source.get(key), key));
    }

    private static Map<?, ?> asMap(Object value, String fieldName) {
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException(fieldName + " must be an object");
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
        return Map.copyOf(values);
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
