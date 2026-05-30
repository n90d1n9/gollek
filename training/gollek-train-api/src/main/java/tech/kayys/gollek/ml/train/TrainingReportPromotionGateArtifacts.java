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
        TrainerCheckpointIO.writeStringAtomically(markdownFile, renderMarkdown(resultMap));
        TrainerCheckpointIO.writeStringAtomically(junitXmlFile, renderJunitXml(resultMap));

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(junitXmlFile),
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

        TrainerCheckpointIO.writeStringAtomically(markdownFile, renderMarkdown(result));
        TrainerCheckpointIO.writeStringAtomically(junitXmlFile, renderJunitXml(result));

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

        TrainerCheckpointIO.writeStringAtomically(resolvedMarkdownFile, renderMarkdown(result));
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, renderJunitXml(result));

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
        return new ArtifactInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                immutableStringKeyMap(map),
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
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedJunitXmlSha = normalizeChecksum(expectedJunitXmlSha256);
        boolean jsonMatches = normalizedJsonSha == null
                || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean junitXmlMatches = normalizedJunitXmlSha == null
                || normalizedJunitXmlSha.equalsIgnoreCase(inspection.junitXmlSha256());
        boolean junitXmlWellFormed = isWellFormedXml(inspection.junitXml());
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
        String renderedMarkdown = null;
        String renderedJunitXml = null;
        try {
            renderedMarkdown = renderMarkdown(inspection.result());
            renderedJunitXml = renderJunitXml(inspection.result());
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
            values.put(String.valueOf(entry.getKey()), TrainerMetadataSupport.immutableSnapshot(entry.getValue()));
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

    private static String renderMarkdown(Map<String, Object> result) {
        Objects.requireNonNull(result, "result must not be null");
        Map<String, Object> decision = mapValue(result, "decision");
        Map<String, Object> review = mapValue(result, "review");
        Map<String, Object> artifacts = mapValue(result, "artifacts");
        Map<String, Object> verification = mapValue(result, "verification");
        Map<String, Object> sourceVerification = mapValue(result, "sourceVerification");

        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Promotion Gate");
        appendLine(markdown, "");
        appendLine(markdown, "**Gate:** `" + (booleanValue(result.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Status:** `" + stringValue(decision.get("status"), "UNKNOWN") + "`");
        appendLine(markdown, "**Promotable:** `" + booleanValue(result.get("promotable")) + "`");
        appendLine(markdown, "**Baseline:** `" + escapeInline(stringValue(decision.get("baseline"), "unknown")) + "`");
        String candidate = stringValue(decision.get("candidate"), "");
        appendLine(markdown, "**Candidate:** `" + escapeInline(candidate.isBlank() ? "none" : candidate) + "`");
        appendLine(markdown, "**Artifact verification:** `"
                + (booleanValue(verification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Source report verification:** `"
                + (booleanValue(sourceVerification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 | Verified |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        appendLine(markdown, artifactRow(
                "JSON review",
                stringValue(artifacts.get("jsonFile"), "n/a"),
                stringValue(artifacts.get("jsonSha256"), "n/a"),
                booleanValue(verification.get("jsonSha256Matches"))));
        appendLine(markdown, artifactRow(
                "Markdown review",
                stringValue(artifacts.get("markdownFile"), "n/a"),
                stringValue(artifacts.get("markdownSha256"), "n/a"),
                booleanValue(verification.get("markdownSha256Matches"))));

        List<Map<String, Object>> sourceReports = mapList(sourceVerification.get("reports"));
        if (!sourceReports.isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Source Reports");
            appendLine(markdown, "");
            appendLine(markdown, "| Role | Name | Path | Bytes | SHA-256 |");
            appendLine(markdown, "| --- | --- | --- | ---: | --- |");
            for (Map<String, Object> report : sourceReports) {
                appendLine(markdown, sourceReportRow(report));
            }
        }

        appendLine(markdown, "");
        appendLine(markdown, "## Review");
        appendLine(markdown, "");
        appendLine(markdown, "Baseline: `" + escapeInline(stringValue(review.get("baseline"), "unknown")) + "`");
        appendLine(markdown, "Candidates audited: `" + longValue(review.get("candidateCount"), 0L) + "`");
        appendLine(markdown, "Promotable candidates: `" + longValue(review.get("promotableCount"), 0L) + "`");
        appendLine(markdown, "Held candidates: `" + longValue(review.get("heldCount"), 0L) + "`");
        appendLine(markdown, "");
        appendLine(markdown, stringValue(result.get("message"), "No promotion gate message recorded."));

        appendListSection(markdown, "Reasons", stringList(decision.get("reasons")));
        appendListSection(markdown, "Verification Failures", stringList(verification.get("failures")));
        appendListSection(markdown, "Source Verification Failures", stringList(sourceVerification.get("failures")));
        return markdown.toString();
    }

    private static String renderJunitXml(Map<String, Object> result) {
        Objects.requireNonNull(result, "result must not be null");
        Map<String, Object> decision = mapValue(result, "decision");
        Map<String, Object> artifacts = mapValue(result, "artifacts");
        Map<String, Object> verification = mapValue(result, "verification");
        Map<String, Object> sourceVerification = mapValue(result, "sourceVerification");
        boolean passed = booleanValue(result.get("passed"));
        String markdown = renderMarkdown(result);
        String baseline = stringValue(decision.get("baseline"), "unknown");
        String candidate = stringValue(decision.get("candidate"), "<none>");

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\""
                + (passed ? "0" : "1")
                + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "gate.passed", Boolean.toString(passed));
        property(xml, "promotion.status", stringValue(decision.get("status"), "UNKNOWN"));
        property(xml, "promotion.promotable", Boolean.toString(booleanValue(decision.get("promotable"))));
        property(xml, "promotion.baseline", baseline);
        if (!candidate.isBlank() && !"<none>".equals(candidate)) {
            property(xml, "promotion.candidate", candidate);
        }
        property(xml, "artifacts.json", stringValue(artifacts.get("jsonFile"), ""));
        property(xml, "artifacts.markdown", stringValue(artifacts.get("markdownFile"), ""));
        property(xml, "artifacts.jsonSha256", stringValue(artifacts.get("jsonSha256"), ""));
        property(xml, "artifacts.markdownSha256", stringValue(artifacts.get("markdownSha256"), ""));
        property(xml, "artifacts.verified", Boolean.toString(booleanValue(verification.get("passed"))));
        property(xml, "sourceReports.verified", Boolean.toString(booleanValue(sourceVerification.get("passed"))));
        property(xml, "sourceReports.count", Long.toString(mapList(sourceVerification.get("reports")).size()));
        property(xml, "sourceReports.failures", Long.toString(stringList(sourceVerification.get("failures")).size()));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.promotion\" name=\""
                + escapeXml(baseline + " -> " + candidate)
                + "\" time=\"0\">");
        if (!passed) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType(decision, verification, sourceVerification))
                    + "\" message=\"" + escapeXml(stringValue(result.get("message"), "Promotion gate failed"))
                    + "\">");
            appendLine(xml, escapeText(markdown));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
        appendLine(xml, "  </testcase>");
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static String artifactRow(
            String label,
            String path,
            String sha256,
            boolean verified) {
        return "| " + escapeTable(label)
                + " | `" + escapeTable(path) + "`"
                + " | `" + escapeTable(shortSha(sha256)) + "`"
                + " | `" + (verified ? "yes" : "no") + "` |";
    }

    private static String sourceReportRow(Map<String, Object> report) {
        return "| " + escapeTable(stringValue(report.get("role"), "unknown"))
                + " | `" + escapeTable(stringValue(report.get("name"), "unknown")) + "`"
                + " | `" + escapeTable(stringValue(report.get("source"), "n/a")) + "`"
                + " | `" + escapeTable(stringValue(report.get("bytes"), "n/a")) + "`"
                + " | `" + escapeTable(shortSha(stringValue(report.get("sha256"), "n/a"))) + "` |";
    }

    private static void appendListSection(StringBuilder markdown, String title, List<String> values) {
        if (values.isEmpty()) {
            return;
        }
        appendLine(markdown, "");
        appendLine(markdown, "## " + title);
        appendLine(markdown, "");
        for (String value : values) {
            appendLine(markdown, "- " + escapeListItem(value));
        }
    }

    private static String failureType(
            Map<String, Object> decision,
            Map<String, Object> verification,
            Map<String, Object> sourceVerification) {
        if (!booleanValue(verification.get("passed"))) {
            return "ARTIFACT_VERIFICATION";
        }
        if (!booleanValue(sourceVerification.get("passed"))) {
            return "SOURCE_REPORT_VERIFICATION";
        }
        return stringValue(decision.get("status"), "PROMOTION_GATE");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
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

    private static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool.booleanValue();
        }
        if (value instanceof String text) {
            return Boolean.parseBoolean(text);
        }
        return false;
    }

    private static long longValue(Object value, long fallback) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value instanceof String text) {
            try {
                return Long.parseLong(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
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

    private static String escapeListItem(String value) {
        return escapeInline(value).replace("\n", " ");
    }

    private static String escapeTable(String value) {
        return escapeInline(value).replace("|", "\\|").replace("\n", " ");
    }

    private static String escapeInline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`");
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }

    private static boolean isWellFormedXml(String xml) {
        return TrainingReportXml.isWellFormed(xml);
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static String escapeXml(String value) {
        return escapeText(value).replace("\"", "&quot;").replace("'", "&apos;");
    }

    private static String escapeText(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            switch (ch) {
                case '&' -> escaped.append("&amp;");
                case '<' -> escaped.append("&lt;");
                case '>' -> escaped.append("&gt;");
                default -> {
                    if (isValidXmlChar(ch)) {
                        escaped.append(ch);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static boolean isValidXmlChar(char ch) {
        return ch == 0x9
                || ch == 0xA
                || ch == 0xD
                || (ch >= 0x20 && ch <= 0xD7FF)
                || (ch >= 0xE000 && ch <= 0xFFFD);
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
