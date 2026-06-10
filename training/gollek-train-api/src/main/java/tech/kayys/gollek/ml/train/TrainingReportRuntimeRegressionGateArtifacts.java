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
 * Writes runtime-regression gate results as small CI-uploadable artifact bundles.
 */
public final class TrainingReportRuntimeRegressionGateArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "runtime-regression-gate.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "runtime-regression-gate.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "runtime-regression-gate.junit.xml";
    public static final String DEFAULT_MANIFEST_FILE_NAME = "runtime-regression-gate.manifest.json";

    private TrainingReportRuntimeRegressionGateArtifacts() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName,
            String junitXmlFileName,
            String manifestFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            junitXmlFileName = normalizeFileName(junitXmlFileName, DEFAULT_JUNIT_XML_FILE_NAME, "junitXmlFileName");
            manifestFileName = normalizeFileName(manifestFileName, DEFAULT_MANIFEST_FILE_NAME, "manifestFileName");
        }

        public static Options defaults() {
            return new Options(
                    DEFAULT_JSON_FILE_NAME,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    DEFAULT_JUNIT_XML_FILE_NAME,
                    DEFAULT_MANIFEST_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path manifestFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            String manifestSha256,
            TrainingReportRuntimeRegressionGate.Result result) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            jsonSha256 = requireChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = requireChecksum(markdownSha256, "markdownSha256");
            junitXmlSha256 = requireChecksum(junitXmlSha256, "junitXmlSha256");
            manifestSha256 = requireChecksum(manifestSha256, "manifestSha256");
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean passed() {
            return result.passed();
        }

        public Map<String, Object> artifactManifest() {
            Map<String, Object> manifest = new LinkedHashMap<>();
            manifest.put("schema", "gollek.training.runtime.regression.gate.artifacts.v1");
            manifest.put("passed", passed());
            manifest.put("jsonFile", jsonFile.toString());
            manifest.put("markdownFile", markdownFile.toString());
            manifest.put("junitXmlFile", junitXmlFile.toString());
            manifest.put("manifestFile", manifestFile.toString());
            manifest.put("jsonSha256", jsonSha256);
            manifest.put("markdownSha256", markdownSha256);
            manifest.put("junitXmlSha256", junitXmlSha256);
            manifest.put("findingCount", result.findings().size());
            manifest.put("regressed", result.runtimeRegression().regressed());
            return Map.copyOf(manifest);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>(artifactManifest());
            map.put("manifestSha256", manifestSha256);
            map.put("directory", directory.toString());
            map.put("result", result.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path manifestFile,
            Map<String, Object> result,
            Map<String, Object> manifest,
            String markdown,
            String junitXml,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            String manifestSha256) {
        public ArtifactInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            result = immutableStringKeyMap(Objects.requireNonNull(result, "result must not be null"));
            manifest = manifest == null ? Map.of() : immutableStringKeyMap(manifest);
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            jsonSha256 = requireChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = requireChecksum(markdownSha256, "markdownSha256");
            junitXmlSha256 = requireChecksum(junitXmlSha256, "junitXmlSha256");
            manifestSha256 = normalizeChecksum(manifestSha256);
        }

        public boolean passed() {
            return Boolean.TRUE.equals(result.get("passed"));
        }

        public boolean hasManifest() {
            return !manifest.isEmpty();
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
            boolean manifestSha256Matches,
            boolean junitXmlWellFormed,
            boolean manifestMatchesFiles,
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
                return "Runtime regression gate artifacts verified for " + inspection.directory() + ".";
            }
            return "Runtime regression gate artifact verification failed: "
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
            map.put("manifestSha256Matches", manifestSha256Matches);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("manifestMatchesFiles", manifestMatchesFiles);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            map.put("failures", failures);
            return Map.copyOf(map);
        }

        public TrainingReportArtifactDescriptor artifact() {
            return TrainingReportArtifactDescriptor.withManifest(
                    inspection.directory(),
                    inspection.jsonFile(),
                    inspection.markdownFile(),
                    inspection.junitXmlFile(),
                    inspection.manifestFile(),
                    inspection.jsonSha256(),
                    inspection.markdownSha256(),
                    inspection.junitXmlSha256(),
                    inspection.manifestSha256(),
                    inspection.hasManifest());
        }

        public Map<String, Object> artifactMap() {
            return artifact().toMap();
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportRuntimeRegressionGate.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportRuntimeRegressionGate.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return writeFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()),
                resolvedDirectory.resolve(resolvedOptions.manifestFileName()),
                result);
    }

    public static ArtifactBundle writeFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            TrainingReportRuntimeRegressionGate.Result result) throws IOException {
        Path manifestFile = siblingManifestFile(junitXmlFile);
        return writeFiles(jsonFile, markdownFile, junitXmlFile, manifestFile, result);
    }

    public static ArtifactBundle writeFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path manifestFile,
            TrainingReportRuntimeRegressionGate.Result result) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedManifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportRuntimeRegressionGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        TrainerCheckpointIO.writeStringAtomically(resolvedJsonFile, TrainerJson.toJson(resolvedResult.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, resolvedResult.markdown());
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, resolvedResult.junitXml());
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        ArtifactBundle bundle = new ArtifactBundle(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                resolvedManifestFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256(),
                "0".repeat(64),
                resolvedResult);
        TrainerCheckpointIO.writeStringAtomically(
                resolvedManifestFile,
                TrainerJson.toJson(bundle.artifactManifest()) + "\n");
        TrainingReportArtifactFingerprint manifestFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedManifestFile);
        return new ArtifactBundle(
                bundle.directory(),
                bundle.jsonFile(),
                bundle.markdownFile(),
                bundle.junitXmlFile(),
                bundle.manifestFile(),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.junitXmlSha256(),
                manifestFingerprint.sha256(),
                bundle.result());
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportRuntimeRegressionGate.Result result =
                resultFromJsonFile(resolvedDirectory.resolve(resolvedOptions.jsonFileName()));
        return write(resolvedDirectory, result, resolvedOptions);
    }

    private static Path siblingManifestFile(Path junitXmlFile) {
        Path resolved = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path parent = resolved.getParent();
        return (parent == null ? Path.of(DEFAULT_MANIFEST_FILE_NAME) : parent.resolve(DEFAULT_MANIFEST_FILE_NAME))
                .toAbsolutePath()
                .normalize();
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
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()),
                resolvedDirectory.resolve(resolvedOptions.manifestFileName()));
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile) throws IOException {
        return readFiles(jsonFile, markdownFile, junitXmlFile, siblingManifestFile(junitXmlFile));
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path manifestFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedManifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String junitXml = Files.readString(resolvedJunitXmlFile, StandardCharsets.UTF_8);
        String manifestJson = Files.exists(resolvedManifestFile)
                ? Files.readString(resolvedManifestFile, StandardCharsets.UTF_8)
                : null;
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint = TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        TrainingReportArtifactFingerprint manifestFingerprint = manifestJson == null
                ? null
                : TrainingReportArtifactFingerprint.of(resolvedManifestFile);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                resolvedManifestFile,
                readJsonMap(json, resolvedJsonFile),
                manifestJson == null ? Map.of() : readJsonMap(manifestJson, resolvedManifestFile),
                markdown,
                junitXml,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256(),
                manifestFingerprint == null ? null : manifestFingerprint.sha256());
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(bundle.jsonFile(), bundle.markdownFile(), bundle.junitXmlFile(), bundle.manifestFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.junitXmlSha256(),
                bundle.manifestSha256());
    }

    public static ArtifactVerification verify(Path directory) throws IOException {
        return verify(read(directory), null, null, null);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        return verify(inspection, expectedJsonSha256, expectedMarkdownSha256, expectedJunitXmlSha256, null);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedManifestSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedJunitXmlSha = normalizeChecksum(expectedJunitXmlSha256);
        String normalizedManifestSha = normalizeChecksum(expectedManifestSha256);
        boolean jsonMatches = normalizedJsonSha == null || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean junitXmlMatches = normalizedJunitXmlSha == null
                || normalizedJunitXmlSha.equalsIgnoreCase(inspection.junitXmlSha256());
        boolean manifestMatches = normalizedManifestSha == null
                || normalizedManifestSha.equalsIgnoreCase(String.valueOf(inspection.manifestSha256()));
        boolean junitXmlWellFormed = TrainingReportXml.isWellFormed(inspection.junitXml());
        boolean manifestMatchesFiles = !inspection.hasManifest()
                || manifestMatchesFiles(inspection);
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
        if (!manifestMatches) {
            failures.add("Manifest checksum mismatch for " + inspection.manifestFile());
        }
        if (!junitXmlWellFormed) {
            failures.add("JUnit XML is not well-formed: " + inspection.junitXmlFile());
        }
        if (!manifestMatchesFiles) {
            failures.add("Manifest does not match artifact files: " + inspection.manifestFile());
        }
        TrainingReportRuntimeRegressionGate.Result result = null;
        try {
            result = resultFromMap(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Runtime regression gate JSON cannot be rendered: " + error.getMessage());
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
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJunitXmlSha,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                manifestMatches,
                junitXmlWellFormed,
                manifestMatchesFiles,
                markdownMatchesJson,
                junitXmlMatchesJson,
                failures);
    }

    private static boolean manifestMatchesFiles(ArtifactInspection inspection) {
        Map<String, Object> manifest = inspection.manifest();
        return "gollek.training.runtime.regression.gate.artifacts.v1".equals(String.valueOf(manifest.get("schema")))
                && inspection.jsonFile().toString().equals(String.valueOf(manifest.get("jsonFile")))
                && inspection.markdownFile().toString().equals(String.valueOf(manifest.get("markdownFile")))
                && inspection.junitXmlFile().toString().equals(String.valueOf(manifest.get("junitXmlFile")))
                && inspection.manifestFile().toString().equals(String.valueOf(manifest.get("manifestFile")))
                && inspection.jsonSha256().equalsIgnoreCase(String.valueOf(manifest.get("jsonSha256")))
                && inspection.markdownSha256().equalsIgnoreCase(String.valueOf(manifest.get("markdownSha256")))
                && inspection.junitXmlSha256().equalsIgnoreCase(String.valueOf(manifest.get("junitXmlSha256")));
    }

    private static TrainingReportRuntimeRegressionGate.Result resultFromJsonFile(Path jsonFile) throws IOException {
        return resultFromMap(readJsonMap(Files.readString(jsonFile, StandardCharsets.UTF_8), jsonFile));
    }

    private static Path commonDirectory(Path jsonFile, Path markdownFile, Path junitXmlFile) {
        Path jsonParent = parent(jsonFile);
        if (jsonParent.equals(parent(markdownFile)) && jsonParent.equals(parent(junitXmlFile))) {
            return jsonParent;
        }
        return jsonParent;
    }

    private static Path parent(Path file) {
        Path parent = file.toAbsolutePath().normalize().getParent();
        return parent == null ? file.toAbsolutePath().normalize() : parent;
    }

    private static String normalizeFileName(String value, String fallback, String field) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        if (normalized.contains("/") || normalized.contains("\\") || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(field + " must be a plain file name");
        }
        return normalized;
    }

    private static String requireChecksum(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value.trim();
    }

    private static Map<String, Object> readJsonMap(String json, Path jsonFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Runtime regression gate JSON must be an object: " + jsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static TrainingReportRuntimeRegressionGate.Result resultFromMap(Map<String, Object> map) {
        List<TrainingReportRuntimeRegressionGate.Finding> findings = new ArrayList<>();
        Object rawFindings = map.get("findings");
        if (rawFindings instanceof Iterable<?> iterable) {
            for (Object rawFinding : iterable) {
                findings.add(findingFromMap(immutableStringKeyMap(asMap(rawFinding, "finding"))));
            }
        }
        return new TrainingReportRuntimeRegressionGate.Result(
                Boolean.TRUE.equals(map.get("available")),
                findings,
                TrainingReportRuntimeRegressionSummary.fromMap(mapValue(map, "runtimeRegression")));
    }

    private static TrainingReportRuntimeRegressionGate.Finding findingFromMap(Map<String, Object> map) {
        return new TrainingReportRuntimeRegressionGate.Finding(
                String.valueOf(map.getOrDefault("code", "")),
                String.valueOf(map.getOrDefault("severity", "warning")),
                String.valueOf(map.getOrDefault("message", "")),
                String.valueOf(map.getOrDefault("action", "")),
                mapValue(map, "evidence"));
    }

    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value instanceof Map<?, ?> map) {
            return immutableStringKeyMap(map);
        }
        return Map.of();
    }

    private static Map<?, ?> asMap(Object value, String fieldName) {
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException(fieldName + " must be an object");
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            copy.put(String.valueOf(entry.getKey()), immutableValue(entry.getValue()));
        }
        return Map.copyOf(copy);
    }

    private static Object immutableValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            return immutableStringKeyMap(map);
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> values = new ArrayList<>();
            for (Object item : iterable) {
                values.add(immutableValue(item));
            }
            return List.copyOf(values);
        }
        return value;
    }

    private static String normalizeChecksum(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
