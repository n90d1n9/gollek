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
import java.util.Optional;

/**
 * Persists promotion reviews as both machine-readable JSON and human-readable Markdown.
 */
public final class TrainingReportPromotionArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "promotion-review.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "promotion-review.md";

    private TrainingReportPromotionArtifacts() {
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
            TrainingReportPortfolio.PromotionDecision decision) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (jsonSha256 == null || jsonSha256.isBlank()) {
                throw new IllegalArgumentException("jsonSha256 must not be blank");
            }
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
            decision = Objects.requireNonNull(decision, "decision must not be null");
        }

        public boolean promotable() {
            return decision.promotable();
        }

        public void requirePromotable() {
            decision.requirePromotable();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("promotable", promotable());
            map.put("decision", decision.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Map<String, Object> review,
            String markdown,
            String jsonSha256,
            String markdownSha256) {
        public ArtifactInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            review = Map.copyOf(Objects.requireNonNull(review, "review must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            if (jsonSha256 == null || jsonSha256.isBlank()) {
                throw new IllegalArgumentException("jsonSha256 must not be blank");
            }
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
        }

        public boolean promotable() {
            Object value = decision().get("promotable");
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
            Object decision = review.get("decision");
            if (!(decision instanceof Map<?, ?> map)) {
                return Map.of();
            }
            return stringKeyMap(map);
        }

        public List<SourceReport> sourceReports() {
            return TrainingReportPromotionArtifacts.sourceReports(this);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("promotable", promotable());
            map.put("decisionStatus", decisionStatus());
            decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
            map.put("sourceReports", sourceReports().stream().map(SourceReport::toMap).toList());
            map.put("review", review);
            return Map.copyOf(map);
        }
    }

    public record SourceReport(
            String role,
            String name,
            Path source,
            Long bytes,
            String sha256) {
        public SourceReport {
            if (role == null || role.isBlank()) {
                throw new IllegalArgumentException("role must not be blank");
            }
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            role = role.trim();
            name = name.trim();
            source = source == null ? null : source.toAbsolutePath().normalize();
            if (bytes != null && bytes.longValue() < 0L) {
                throw new IllegalArgumentException("bytes must be non-negative");
            }
            sha256 = normalizeChecksum(sha256);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role", role);
            map.put("name", name);
            if (source != null) {
                map.put("source", source.toString());
            }
            if (bytes != null) {
                map.put("bytes", bytes);
            }
            if (sha256 != null) {
                map.put("sha256", sha256);
            }
            return Map.copyOf(map);
        }
    }

    public record SourceVerification(
            ArtifactInspection inspection,
            List<SourceReport> reports,
            List<String> failures) {
        public SourceVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            reports = reports == null ? List.of() : List.copyOf(reports);
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
                return "Promotion source reports verified for " + inspection.directory() + ".";
            }
            return "Promotion source report verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("reports", reports.stream().map(SourceReport::toMap).toList());
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
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
                return "Promotion artifacts verified for " + inspection.directory() + ".";
            }
            return "Promotion artifact verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
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
            TrainingReportPortfolio.PromotionReview review) throws IOException {
        return write(directory, review, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolio.PromotionReview review,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPortfolio.PromotionReview resolvedReview =
                Objects.requireNonNull(review, "review must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());

        TrainerCheckpointIO.writeStringAtomically(jsonFile, TrainerJson.toJson(resolvedReview.toMap()) + "\n");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, TrainingReportPromotionMarkdown.render(resolvedReview));
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                resolvedReview.decision());
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
            throw new IOException("Promotion review JSON must be an object: " + resolvedJsonFile);
        }
        TrainingReportArtifactFingerprint jsonFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                stringKeyMap(map),
                markdown,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256());
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(bundle.jsonFile(), bundle.markdownFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256());
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
        boolean jsonMatches = normalizedJsonSha == null
                || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                jsonMatches,
                markdownMatches,
                failures);
    }

    public static List<SourceReport> sourceReports(ArtifactInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        Map<String, SourceReport> reports = new LinkedHashMap<>();
        addSourceReport(
                reports,
                "baseline",
                stringValue(inspection.review().get("baseline"), "baseline"),
                inspection.review().get("baselineReport"));

        Object candidates = inspection.review().get("candidates");
        if (candidates instanceof Iterable<?> iterable) {
            for (Object item : iterable) {
                if (item instanceof Map<?, ?> candidateMap) {
                    Map<String, Object> candidate = stringKeyMap(candidateMap);
                    addSourceReport(
                            reports,
                            "candidate",
                            stringValue(candidate.get("candidate"), "candidate"),
                            candidate.get("candidateReport"));
                }
            }
        }

        if (reports.values().stream().noneMatch(report -> "candidate".equals(report.role()))) {
            Map<String, Object> decision = inspection.decision();
            addSourceReport(
                    reports,
                    "candidate",
                    stringValue(decision.get("candidate"), "candidate"),
                    decision.get("candidateReport"));
        }
        return List.copyOf(reports.values());
    }

    public static SourceVerification verifySourceReports(Path directory) throws IOException {
        return verifySourceReports(read(directory));
    }

    public static SourceVerification verifySourceReports(Path directory, Options options) throws IOException {
        return verifySourceReports(read(directory, options));
    }

    public static SourceVerification verifySourceReports(ArtifactInspection inspection) throws IOException {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<SourceReport> reports = sourceReports(inspection);
        List<String> failures = new ArrayList<>();
        if (reports.isEmpty()) {
            failures.add("No source report provenance was recorded in " + inspection.jsonFile());
        }
        for (SourceReport report : reports) {
            verifySourceReport(report, failures);
        }
        return new SourceVerification(inspection, reports, failures);
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

    private static void addSourceReport(
            Map<String, SourceReport> reports,
            String role,
            String fallbackName,
            Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return;
        }
        Map<String, Object> report = stringKeyMap(map);
        Path source = pathValue(report.get("source"));
        Long bytes = longValue(report.get("sourceBytes"));
        String sha256 = stringValue(report.get("sourceSha256"), null);
        if (source == null && bytes == null && (sha256 == null || sha256.isBlank())) {
            return;
        }
        String name = stringValue(report.get("name"), fallbackName);
        SourceReport sourceReport = new SourceReport(role, name, source, bytes, sha256);
        reports.put(sourceReport.role() + '\u0000' + sourceReport.name(), sourceReport);
    }

    private static void verifySourceReport(SourceReport report, List<String> failures) throws IOException {
        if (report.source() == null) {
            failures.add(report.role() + " source report path is missing for " + report.name());
            return;
        }
        if (!Files.isRegularFile(report.source())) {
            failures.add(report.role() + " source report is missing for " + report.name() + ": " + report.source());
            return;
        }
        if (report.bytes() == null && (report.sha256() == null || report.sha256().isBlank())) {
            failures.add(report.role() + " source report fingerprint is missing for " + report.name());
            return;
        }
        TrainingReportArtifactFingerprint actual = TrainingReportArtifactFingerprint.of(report.source());
        if (report.bytes() != null) {
            if (actual.bytes() != report.bytes().longValue()) {
                failures.add(report.role() + " source report byte size mismatch for " + report.name()
                        + " (expected " + report.bytes() + " bytes, got " + actual.bytes() + " bytes)");
            }
        }
        if (report.sha256() != null && !report.sha256().isBlank()) {
            if (!report.sha256().equalsIgnoreCase(actual.sha256())) {
                failures.add(report.role() + " source report SHA-256 mismatch for " + report.name()
                        + " (expected " + report.sha256() + ", got " + actual.sha256() + ")");
            }
        }
    }

    private static Path pathValue(Object value) {
        String text = stringValue(value, "");
        return text.isBlank() ? null : Path.of(text).toAbsolutePath().normalize();
    }

    private static Long longValue(Object value) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value instanceof String text) {
            try {
                return Long.parseLong(text.trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static Map<String, Object> stringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
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
        return checksum.trim().toLowerCase(java.util.Locale.ROOT);
    }
}
