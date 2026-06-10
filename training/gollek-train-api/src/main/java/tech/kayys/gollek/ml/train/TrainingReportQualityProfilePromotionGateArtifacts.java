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
 * Persists profile-aware promotion gate results for CI uploads.
 */
public final class TrainingReportQualityProfilePromotionGateArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "quality-profile-promotion-gate.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "quality-profile-promotion-gate.md";

    private TrainingReportQualityProfilePromotionGateArtifacts() {
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
            TrainingReportQualityProfilePromotionGate.Result result) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            result = Objects.requireNonNull(result, "result must not be null");
        }

        public boolean passed() {
            return result.passed();
        }

        public boolean promotable() {
            return result.promotable();
        }

        public String profileId() {
            return result.profile().id();
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
            map.put("promotable", promotable());
            map.put("profileId", profileId());
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
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            result = immutableStringKeyMap(Objects.requireNonNull(result, "result must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
        }

        public boolean passed() {
            return booleanValue(result.get("passed"));
        }

        public boolean promotable() {
            return booleanValue(result.get("promotable"));
        }

        public Optional<String> profileId() {
            String id = stringValue(profile().get("id"), "");
            return id.isBlank() ? Optional.empty() : Optional.of(id);
        }

        public Map<String, Object> profile() {
            return mapValue(result, "profile");
        }

        public Map<String, Object> gate() {
            return mapValue(result, "gate");
        }

        public Map<String, Object> decision() {
            return mapValue(result, "decision");
        }

        public String decisionStatus() {
            return stringValue(decision().get("status"), "UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            String candidate = stringValue(decision().get("candidate"), "");
            return candidate.isBlank() ? Optional.empty() : Optional.of(candidate);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("passed", passed());
            map.put("promotable", promotable());
            profileId().ifPresent(id -> map.put("profileId", id));
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
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean profileKnown,
            boolean gatePayloadConsistent,
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
                return "Quality-profile promotion gate artifacts verified for " + inspection.directory() + ".";
            }
            return "Quality-profile promotion gate artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("profileKnown", profileKnown);
            map.put("gatePayloadConsistent", gatePayloadConsistent);
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

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfilePromotionGate.Result result) throws IOException {
        return write(directory, result, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportQualityProfilePromotionGate.Result result,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportQualityProfilePromotionGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Map<String, Object> resultMap = resolvedResult.toMap();

        TrainerCheckpointIO.writeStringAtomically(jsonFile, TrainerJson.toJson(resultMap) + "\n");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, renderMarkdown(resultMap));
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
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
        Map<String, Object> result = readResult(jsonFile);
        TrainerCheckpointIO.writeStringAtomically(markdownFile, renderMarkdown(result));
        return readFiles(jsonFile, markdownFile);
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
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality-profile promotion gate JSON must be an object: " + resolvedJsonFile);
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
                Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8),
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

        boolean profileKnown = inspection.profileId()
                .flatMap(TrainingReportQualityProfile::find)
                .isPresent();
        if (!profileKnown) {
            failures.add("Quality profile is unknown or missing in " + inspection.jsonFile());
        }

        boolean gatePayloadConsistent = gatePayloadConsistent(inspection.result());
        if (!gatePayloadConsistent) {
            failures.add("Quality-profile gate payload is inconsistent with wrapped promotion gate JSON: "
                    + inspection.jsonFile());
        }

        String renderedMarkdown = null;
        try {
            renderedMarkdown = renderMarkdown(inspection.result());
        } catch (RuntimeException error) {
            failures.add("Quality-profile gate JSON cannot be rendered: " + error.getMessage());
        }
        boolean markdownMatchesJson = renderedMarkdown != null && renderedMarkdown.equals(inspection.markdown());
        if (renderedMarkdown != null && !markdownMatchesJson) {
            failures.add("Markdown report does not match quality-profile gate JSON: " + inspection.markdownFile());
        }

        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                jsonMatches,
                markdownMatches,
                profileKnown,
                gatePayloadConsistent,
                markdownMatchesJson,
                failures);
    }

    private static Map<String, Object> readResult(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Quality-profile promotion gate JSON must be an object: " + resolvedJsonFile);
        }
        return immutableStringKeyMap(map);
    }

    private static boolean gatePayloadConsistent(Map<String, Object> result) {
        Map<String, Object> gate = mapValue(result, "gate");
        if (gate.isEmpty()) {
            return false;
        }
        String message = stringValue(result.get("message"), "");
        String gateMessage = stringValue(gate.get("message"), "");
        return booleanValue(result.get("passed")) == booleanValue(gate.get("passed"))
                && booleanValue(result.get("promotable")) == booleanValue(gate.get("promotable"))
                && !gateMessage.isBlank()
                && message.endsWith(gateMessage);
    }

    private static String renderMarkdown(Map<String, Object> result) {
        Map<String, Object> profile = mapValue(result, "profile");
        Map<String, Object> decision = mapValue(result, "decision");
        Map<String, Object> artifacts = mapValue(result, "artifacts");
        Map<String, Object> verification = mapValue(result, "verification");
        Map<String, Object> sourceVerification = mapValue(result, "sourceVerification");

        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Quality Profile Gate");
        appendLine(markdown, "");
        appendLine(markdown, "**Profile:** `" + escapeInline(stringValue(profile.get("id"), "unknown"))
                + "` (" + escapeInline(stringValue(profile.get("displayName"), "unknown")) + ")");
        appendLine(markdown, "**Gate:** `" + (booleanValue(result.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Promotion status:** `" + escapeInline(stringValue(decision.get("status"), "UNKNOWN"))
                + "`");
        appendLine(markdown, "**Promotable:** `" + booleanValue(result.get("promotable")) + "`");
        appendLine(markdown, "**Candidate:** `"
                + escapeInline(stringValue(decision.get("candidate"), "none")) + "`");
        appendLine(markdown, "**Artifact verification:** `"
                + (booleanValue(verification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Source report verification:** `"
                + (booleanValue(sourceVerification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Profile Contract");
        appendLine(markdown, "");
        appendLine(markdown, stringValue(profile.get("description"), "No profile description recorded."));
        appendLine(markdown, "");
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "- Promotion review JSON: `" + escapeInline(stringValue(artifacts.get("jsonFile"), "n/a")) + "`");
        appendLine(markdown, "- Promotion review Markdown: `"
                + escapeInline(stringValue(artifacts.get("markdownFile"), "n/a")) + "`");
        appendLine(markdown, "- JSON SHA-256: `" + escapeInline(shortSha(stringValue(artifacts.get("jsonSha256"), "n/a"))) + "`");
        appendLine(markdown, "- Markdown SHA-256: `"
                + escapeInline(shortSha(stringValue(artifacts.get("markdownSha256"), "n/a"))) + "`");
        appendListSection(markdown, "Decision Reasons", stringList(decision.get("reasons")));
        appendListSection(markdown, "Artifact Verification Failures", stringList(verification.get("failures")));
        appendListSection(markdown, "Source Verification Failures", stringList(sourceVerification.get("failures")));
        appendLine(markdown, "");
        appendLine(markdown, "## Message");
        appendLine(markdown, "");
        appendLine(markdown, stringValue(result.get("message"), "No quality-profile gate message recorded."));
        return markdown.toString();
    }

    private static void appendListSection(StringBuilder markdown, String title, List<String> values) {
        if (values.isEmpty()) {
            return;
        }
        appendLine(markdown, "");
        appendLine(markdown, "## " + title);
        appendLine(markdown, "");
        for (String value : values) {
            appendLine(markdown, "- " + escapeInline(value).replace("\n", " "));
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item != null) {
                String text = String.valueOf(item).trim();
                if (!text.isEmpty()) {
                    values.add(text);
                }
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

    private static String shortSha(String sha256) {
        if (sha256 == null || sha256.isBlank()) {
            return "n/a";
        }
        String normalized = sha256.trim();
        return normalized.length() <= 12 ? normalized : normalized.substring(0, 12);
    }

    private static String escapeInline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
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

    private static void requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }
}
