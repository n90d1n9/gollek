package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writes comparison action-plan exports as JSON, Markdown, and manifest artifacts.
 */
public final class TrainingReportComparisonActionPlanArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "comparison-action-plan.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "comparison-action-plan.md";
    public static final String DEFAULT_MANIFEST_FILE_NAME = "comparison-action-plan.manifest.json";

    private TrainingReportComparisonActionPlanArtifacts() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName,
            String manifestFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            manifestFileName = normalizeFileName(manifestFileName, DEFAULT_MANIFEST_FILE_NAME, "manifestFileName");
        }

        public static Options defaults() {
            return new Options(DEFAULT_JSON_FILE_NAME, DEFAULT_MARKDOWN_FILE_NAME, DEFAULT_MANIFEST_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path manifestFile,
            Map<String, Object> manifest,
            TrainingReportComparisonActionPlanExport export) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifest = manifest == null ? Map.of() : Map.copyOf(manifest);
            export = Objects.requireNonNull(export, "export must not be null");
        }

        public boolean regressed() {
            return export.regressed();
        }

        public boolean requiresAttention() {
            return export.requiresAttention();
        }

        public Map<String, Object> toMap() {
            return Map.of(
                    "directory", directory.toString(),
                    "jsonFile", jsonFile.toString(),
                    "markdownFile", markdownFile.toString(),
                    "manifestFile", manifestFile.toString(),
                    "manifest", manifest,
                    "regressed", regressed(),
                    "requiresAttention", requiresAttention());
        }
    }

    public record Verification(
            Path manifestFile,
            Map<String, Object> manifest,
            List<String> failures) {
        public Verification {
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifest = manifest == null ? Map.of() : Map.copyOf(manifest);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public boolean hasFailures() {
            return !failures.isEmpty();
        }

        public int failureCount() {
            return failures.size();
        }

        public String summary() {
            if (passed()) {
                return "Comparison action-plan artifacts verified.";
            }
            return "Comparison action-plan artifact verification failed with "
                    + failureCount()
                    + " failure(s).";
        }

        public Map<String, Object> toMap() {
            return Map.of(
                    "manifestFile", manifestFile.toString(),
                    "passed", passed(),
                    "hasFailures", hasFailures(),
                    "failureCount", failureCount(),
                    "summary", summary(),
                    "failures", failures,
                    "manifest", manifest);
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparisonActionPlanExport export) throws IOException {
        return write(directory, export, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparisonActionPlanExport export,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportComparisonActionPlanExport resolvedExport =
                Objects.requireNonNull(export, "export must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path manifestFile = resolvedDirectory.resolve(resolvedOptions.manifestFileName());

        resolvedExport.writeJson(jsonFile);
        resolvedExport.writeMarkdown(markdownFile);
        Map<String, Object> manifest = resolvedExport.artifactManifest(jsonFile, markdownFile);
        TrainerCheckpointIO.writeStringAtomically(manifestFile, TrainerJson.toJson(manifest) + "\n");
        return new ArtifactBundle(resolvedDirectory, jsonFile, markdownFile, manifestFile, manifest, resolvedExport);
    }

    public static Verification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(bundle.manifestFile());
    }

    public static Verification verify(Path manifestFile) throws IOException {
        Path resolvedManifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                .toAbsolutePath()
                .normalize();
        Map<String, Object> manifest = readManifest(resolvedManifestFile);
        List<String> failures = new ArrayList<>();
        verifyManifestSchema(manifest, failures);
        verifyFingerprint("json", manifest.get("json"), failures);
        verifyFingerprint("markdown", manifest.get("markdown"), failures);
        return new Verification(resolvedManifestFile, manifest, failures);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readManifest(Path manifestFile) throws IOException {
        Object parsed = TrainerJsonParser.parse(Files.readString(manifestFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Comparison action-plan artifact manifest must be a JSON object: " + manifestFile);
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static void verifyManifestSchema(Map<String, Object> manifest, List<String> failures) {
        Object schema = manifest.get("schema");
        if (!TrainingReportComparisonActionPlanExport.ARTIFACT_MANIFEST_SCHEMA.equals(schema)) {
            failures.add("Artifact manifest schema mismatch: " + schema);
        }
        Object exportSchema = manifest.get("exportSchema");
        if (!TrainingReportComparisonActionPlanExport.SCHEMA.equals(exportSchema)) {
            failures.add("Artifact manifest export schema mismatch: " + exportSchema);
        }
    }

    private static void verifyFingerprint(String name, Object value, List<String> failures) throws IOException {
        if (!(value instanceof Map<?, ?> map)) {
            failures.add("Artifact manifest missing fingerprint: " + name);
            return;
        }
        try {
            TrainingReportArtifactFingerprint expected = TrainingReportArtifactFingerprint.fromMap(map, name);
            TrainingReportArtifactFingerprint actual = TrainingReportArtifactFingerprint.of(expected.file());
            expected.verifyMatches(name, actual, failures);
        } catch (IllegalArgumentException error) {
            failures.add("Invalid " + name + " fingerprint: " + error.getMessage());
        }
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String resolved = value == null || value.isBlank() ? fallback : value.trim();
        if (resolved.contains("/") || resolved.contains("\\")) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + resolved);
        }
        return resolved;
    }
}
