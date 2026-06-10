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
 * Persists quality-profile catalogs as CI/release artifacts.
 */
public final class TrainingReportQualityProfileArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "training-report-quality-profiles.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "training-report-quality-profiles.md";

    private TrainingReportQualityProfileArtifacts() {
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

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            String jsonSha256,
            String markdownSha256,
            TrainingReportQualityProfileCatalog catalog) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            catalog = Objects.requireNonNull(catalog, "catalog must not be null");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("profileIds", catalog.ids());
            map.put("catalog", catalog.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Map<String, Object> catalog,
            String markdown,
            String jsonSha256,
            String markdownSha256) {
        public ArtifactInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            catalog = immutableStringKeyMap(Objects.requireNonNull(catalog, "catalog must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
        }

        public List<String> profileIds() {
            Object profilesValue = catalog.get("profiles");
            if (!(profilesValue instanceof Iterable<?> iterable)) {
                return List.of();
            }
            List<String> ids = new ArrayList<>();
            for (Object item : iterable) {
                if (item instanceof Map<?, ?> map) {
                    Object id = map.get("id");
                    if (id != null) {
                        ids.add(String.valueOf(id));
                    }
                }
            }
            return List.copyOf(ids);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("profileIds", profileIds());
            map.put("catalog", catalog);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean jsonMatchesCatalog,
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

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Training report quality profile artifacts verified for " + inspection.directory() + ".";
            }
            return "Training report quality profile artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("jsonMatchesCatalog", jsonMatchesCatalog);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("actualJsonSha256", inspection.jsonSha256());
            map.put("actualMarkdownSha256", inspection.markdownSha256());
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

    public static ArtifactBundle write(Path directory) throws IOException {
        return write(directory, TrainingReportQualityProfileCatalog.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalog catalog) throws IOException {
        return write(directory, catalog, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfileCatalog catalog,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportQualityProfileCatalog resolvedCatalog = catalog == null
                ? TrainingReportQualityProfileCatalog.defaults()
                : catalog;
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());

        TrainerCheckpointIO.writeStringAtomically(jsonFile, resolvedCatalog.toJson() + "\n");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, resolvedCatalog.toMarkdown());
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                resolvedCatalog);
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
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        TrainingReportQualityProfileCatalog catalog = readCatalog(jsonFile);
        TrainerCheckpointIO.writeStringAtomically(markdownFile, catalog.toMarkdown());
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);
        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                catalog);
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
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()));
    }

    public static ArtifactInspection readFiles(Path jsonFile, Path markdownFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Training report quality profile catalog JSON must be an object: "
                    + resolvedJsonFile);
        }
        TrainingReportArtifactFingerprint jsonFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                immutableStringKeyMap(map),
                markdown,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256());
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(readFiles(bundle.jsonFile(), bundle.markdownFile()), bundle.jsonSha256(), bundle.markdownSha256());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return verify(directory, expectedJsonSha256, expectedMarkdownSha256, Options.defaults());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            Options options) throws IOException {
        return verify(read(directory, options), expectedJsonSha256, expectedMarkdownSha256);
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
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }

        TrainingReportQualityProfileCatalog catalog = null;
        try {
            catalog = TrainingReportQualityProfileCatalog.fromMap(inspection.catalog());
        } catch (RuntimeException error) {
            failures.add("Quality profile catalog JSON cannot be rendered: " + error.getMessage());
        }

        boolean jsonMatchesCatalog = false;
        boolean markdownMatchesJson = false;
        if (catalog != null) {
            jsonMatchesCatalog = catalog.toMap().equals(inspection.catalog());
            markdownMatchesJson = catalog.toMarkdown().equals(inspection.markdown());
            if (!jsonMatchesCatalog) {
                failures.add("Quality profile catalog JSON does not match its normalized profile definitions: "
                        + inspection.jsonFile());
            }
            if (!markdownMatchesJson) {
                failures.add("Markdown catalog does not match quality profile JSON: " + inspection.markdownFile());
            }
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                jsonMatches,
                markdownMatches,
                jsonMatchesCatalog,
                markdownMatchesJson,
                failures);
    }

    private static TrainingReportQualityProfileCatalog readCatalog(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Training report quality profile catalog JSON must be an object: "
                    + resolvedJsonFile);
        }
        try {
            return TrainingReportQualityProfileCatalog.fromMap(immutableStringKeyMap(map));
        } catch (RuntimeException error) {
            throw new IOException("Training report quality profile catalog JSON cannot be rendered: "
                    + resolvedJsonFile, error);
        }
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static Path commonDirectory(Path jsonFile, Path markdownFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        if (jsonParent != null && jsonParent.equals(markdownParent)) {
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
