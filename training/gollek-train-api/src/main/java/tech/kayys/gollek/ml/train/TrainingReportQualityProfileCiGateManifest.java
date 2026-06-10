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
 * One-file receipt for every artifact emitted by a quality-profile CI gate run.
 */
public final class TrainingReportQualityProfileCiGateManifest {
    public static final String FORMAT = "gollek.training.quality-profile.ci-gate.manifest.v1";
    public static final String DEFAULT_JSON_FILE_NAME = "quality-profile-ci-gate-manifest.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "quality-profile-ci-gate-manifest.md";

    private TrainingReportQualityProfileCiGateManifest() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
        }

        public static Options defaults() {
            return new Options(DEFAULT_JSON_FILE_NAME, DEFAULT_MARKDOWN_FILE_NAME);
        }
    }

    public record ArtifactEntry(
            String name,
            String kind,
            String reportName,
            Path file,
            String manifestPath,
            long bytes,
            String sha256) {
        public ArtifactEntry {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            if (kind == null || kind.isBlank()) {
                throw new IllegalArgumentException("kind must not be blank");
            }
            name = name.trim();
            kind = kind.trim();
            reportName = reportName == null ? "" : reportName.trim();
            file = Objects.requireNonNull(file, "file must not be null").toAbsolutePath().normalize();
            if (manifestPath == null || manifestPath.isBlank()) {
                throw new IllegalArgumentException("manifestPath must not be blank");
            }
            manifestPath = manifestPath.trim();
            if (bytes < 0L) {
                throw new IllegalArgumentException("bytes must be non-negative");
            }
            sha256 = normalizeRequiredChecksum(sha256, "sha256");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            map.put("kind", kind);
            if (!reportName.isBlank()) {
                map.put("reportName", reportName);
            }
            map.put("file", manifestPath);
            map.put("absoluteFile", file.toString());
            map.put("bytes", bytes);
            map.put("sha256", sha256);
            return Map.copyOf(map);
        }
    }

    public record ManifestBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            String jsonSha256,
            String markdownSha256,
            TrainingReportQualityProfileCiGate.Result result,
            List<ArtifactEntry> artifacts) {
        public ManifestBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            jsonSha256 = normalizeRequiredChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = normalizeRequiredChecksum(markdownSha256, "markdownSha256");
            result = Objects.requireNonNull(result, "result must not be null");
            artifacts = artifacts == null ? List.of() : List.copyOf(artifacts);
        }

        public boolean passed() {
            return result.passed();
        }

        public void requirePassed() {
            result.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("passed", passed());
            map.put("profileId", result.profile().id());
            map.put("artifactCount", artifacts.size());
            map.put("artifacts", artifacts.stream().map(ArtifactEntry::toMap).toList());
            return Map.copyOf(map);
        }
    }

    public record ManifestInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Map<String, Object> manifest,
            String markdown,
            String jsonSha256,
            String markdownSha256) {
        public ManifestInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifest = immutableStringKeyMap(Objects.requireNonNull(manifest, "manifest must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            jsonSha256 = normalizeRequiredChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = normalizeRequiredChecksum(markdownSha256, "markdownSha256");
        }

        public String format() {
            return stringValue(manifest.get("format"), "");
        }

        public boolean passed() {
            return booleanValue(manifest.get("passed"));
        }

        public Optional<String> profileId() {
            String id = stringValue(manifest.get("profileId"), "");
            return id.isBlank() ? Optional.empty() : Optional.of(id);
        }

        public List<Map<String, Object>> artifactMaps() {
            return mapList(manifest.get("artifacts"));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("format", format());
            map.put("passed", passed());
            profileId().ifPresent(id -> map.put("profileId", id));
            map.put("artifactCount", artifactMaps().size());
            map.put("manifest", manifest);
            return Map.copyOf(map);
        }
    }

    public record ManifestVerification(
            ManifestInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean formatValid,
            boolean profileKnown,
            boolean markdownMatchesJson,
            boolean artifactsMatch,
            List<String> failures) {
        public ManifestVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonSha256 = normalizeChecksum(expectedJsonSha256);
            expectedMarkdownSha256 = normalizeChecksum(expectedMarkdownSha256);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public boolean structureValid() {
            return structureFailures().isEmpty();
        }

        public List<String> structureFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.STRUCTURE);
        }

        public List<String> checksumFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.CHECKSUM);
        }

        public List<String> formatFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.FORMAT);
        }

        public List<String> profileFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.PROFILE);
        }

        public List<String> markdownFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.MARKDOWN);
        }

        public List<String> artifactFailures() {
            return categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory.ARTIFACT);
        }

        public List<TrainingReportQualityProfileCiGateManifestFailure> structuredFailures() {
            return TrainingReportQualityProfileCiGateManifestFailures.categorize(failures);
        }

        public Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> failureCountsByCategory() {
            return TrainingReportQualityProfileCiGateManifestFailures.countByCategory(failures);
        }

        public TrainingReportQualityProfileCiGateManifestSummary summary() {
            return TrainingReportQualityProfileCiGateManifestSummary.from(this);
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Quality-profile CI gate manifest verified for " + inspection.directory() + ".";
            }
            return "Quality-profile CI gate manifest verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("formatValid", formatValid);
            map.put("profileKnown", profileKnown);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("structureValid", structureValid());
            map.put("artifactsMatch", artifactsMatch);
            map.put("actualJsonSha256", inspection.jsonSha256());
            map.put("actualMarkdownSha256", inspection.markdownSha256());
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            map.put("failures", failures);
            map.put("structureFailures", structureFailures());
            map.put("checksumFailures", checksumFailures());
            map.put("formatFailures", formatFailures());
            map.put("profileFailures", profileFailures());
            map.put("markdownFailures", markdownFailures());
            map.put("artifactFailures", artifactFailures());
            map.put("structuredFailures", structuredFailures().stream()
                    .map(TrainingReportQualityProfileCiGateManifestFailure::toMap)
                    .toList());
            map.put("failureCountsByCategory",
                    TrainingReportQualityProfileCiGateManifestFailures.countMapByCategory(failures));
            map.put("summary", summary().toMap());
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }

        private List<String> categoryFailures(TrainingReportQualityProfileCiGateManifestFailureCategory category) {
            return TrainingReportQualityProfileCiGateManifestFailures.messages(failures, category);
        }
    }

    public static ManifestBundle write(
            Path directory,
            TrainingReportQualityProfileCiGate.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ManifestBundle write(
            Path directory,
            TrainingReportQualityProfileCiGate.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportQualityProfileCiGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        List<ArtifactEntry> artifacts =
                TrainingReportQualityProfileCiGateManifestArtifacts.collect(resolvedDirectory, resolvedResult);
        Map<String, Object> manifest = manifestMap(resolvedResult, artifacts);

        TrainerCheckpointIO.writeStringAtomically(jsonFile, TrainerJson.toJson(manifest) + "\n");
        TrainerCheckpointIO.writeStringAtomically(
                markdownFile,
                TrainingReportQualityProfileCiGateManifestMarkdown.render(manifest));
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);

        return new ManifestBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                resolvedResult,
                artifacts);
    }

    public static ManifestInspection refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ManifestInspection refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Map<String, Object> manifest = readManifest(jsonFile);
        TrainerCheckpointIO.writeStringAtomically(
                markdownFile,
                TrainingReportQualityProfileCiGateManifestMarkdown.render(manifest));
        return readFiles(jsonFile, markdownFile);
    }

    public static ManifestInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ManifestInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()));
    }

    public static ManifestInspection readFiles(Path jsonFile, Path markdownFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportArtifactFingerprint jsonFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        return new ManifestInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                readManifest(resolvedJsonFile),
                Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8),
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256());
    }

    public static ManifestVerification verify(ManifestBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(readFiles(bundle.jsonFile(), bundle.markdownFile()), bundle.jsonSha256(), bundle.markdownSha256());
    }

    public static ManifestVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return verify(directory, expectedJsonSha256, expectedMarkdownSha256, Options.defaults());
    }

    public static ManifestVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            Options options) throws IOException {
        return verify(read(directory, options), expectedJsonSha256, expectedMarkdownSha256);
    }

    public static ManifestVerification verify(
            ManifestInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        boolean jsonMatches = normalizedJsonSha == null || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean formatValid = FORMAT.equals(inspection.format());
        boolean profileKnown = inspection.profileId()
                .flatMap(TrainingReportQualityProfile::find)
                .isPresent();
        String renderedMarkdown = null;
        List<String> failures = new ArrayList<>();

        if (!jsonMatches) {
            failures.add("Manifest JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Manifest Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!formatValid) {
            failures.add("Manifest format mismatch: " + inspection.format());
        }
        if (!profileKnown) {
            failures.add("Manifest profile is unknown or missing in " + inspection.jsonFile());
        }
        try {
            renderedMarkdown = TrainingReportQualityProfileCiGateManifestMarkdown.render(inspection.manifest());
        } catch (RuntimeException error) {
            failures.add("Manifest JSON cannot be rendered: " + error.getMessage());
        }
        boolean markdownMatchesJson = renderedMarkdown != null && renderedMarkdown.equals(inspection.markdown());
        if (renderedMarkdown != null && !markdownMatchesJson) {
            failures.add("Manifest Markdown does not match manifest JSON: " + inspection.markdownFile());
        }

        failures.addAll(TrainingReportQualityProfileCiGateManifestStructure.verify(inspection));
        List<String> artifactFailures = TrainingReportQualityProfileCiGateManifestArtifacts.verify(inspection);
        boolean artifactsMatch = artifactFailures.isEmpty();
        failures.addAll(artifactFailures);

        return new ManifestVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                jsonMatches,
                markdownMatches,
                formatValid,
                profileKnown,
                markdownMatchesJson,
                artifactsMatch,
                failures);
    }

    private static Map<String, Object> manifestMap(
            TrainingReportQualityProfileCiGate.Result result,
            List<ArtifactEntry> artifacts) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", FORMAT);
        manifest.put("profileId", result.profile().id());
        manifest.put("profile", result.profile().toMap());
        manifest.put("passed", result.passed());
        manifest.put("validationPassed", result.validationPassed());
        manifest.put("promotionPassed", result.promotionPassed());
        manifest.put("artifactsVerified", result.artifactsVerified());
        manifest.put("message", result.message());
        manifest.put("validationCount", result.validations().size());
        manifest.put("failedValidationNames", result.failedValidationNames());
        manifest.put("artifactCount", artifacts.size());
        manifest.put("artifacts", artifacts.stream().map(ArtifactEntry::toMap).toList());
        return Map.copyOf(manifest);
    }

    private static Map<String, Object> readManifest(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality-profile CI gate manifest JSON must be an object: " + resolvedJsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static List<Map<String, Object>> mapList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Map<String, Object>> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                values.add(immutableStringKeyMap(map));
            }
        }
        return List.copyOf(values);
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), TrainingReportSnapshots.immutableSnapshot(entry.getValue()));
        }
        return Map.copyOf(values);
    }

    private static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool.booleanValue();
        }
        if (value instanceof String text) {
            return Boolean.parseBoolean(text);
        }
        return false;
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static Path commonDirectory(Path jsonFile, Path markdownFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        if (jsonParent != null && jsonParent.equals(markdownParent)) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static String normalizeRequiredChecksum(String checksum, String fieldName) {
        String normalized = normalizeChecksum(checksum);
        if (normalized == null) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return normalized;
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }
}
