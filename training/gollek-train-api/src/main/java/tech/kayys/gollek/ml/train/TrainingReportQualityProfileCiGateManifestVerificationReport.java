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
 * JSON/Markdown/JUnit artifact bundle for manifest verification evidence.
 */
public final class TrainingReportQualityProfileCiGateManifestVerificationReport {
    public static final String FORMAT =
            "gollek.training.quality-profile.ci-gate.manifest.verification.v1";
    public static final String DEFAULT_JSON_FILE_NAME =
            "quality-profile-ci-gate-manifest-verification.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME =
            "quality-profile-ci-gate-manifest-verification.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME =
            "quality-profile-ci-gate-manifest-verification.junit.xml";

    private TrainingReportQualityProfileCiGateManifestVerificationReport() {
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

    public record ReportBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            TrainingReportQualityProfileCiGateManifestJUnitXml.Report junitXmlReport,
            String json,
            String markdown,
            String jsonSha256,
            String markdownSha256,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        public ReportBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlReport = Objects.requireNonNull(junitXmlReport, "junitXmlReport must not be null");
            json = Objects.requireNonNull(json, "json must not be null");
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            jsonSha256 = requireChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = requireChecksum(markdownSha256, "markdownSha256");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public boolean readyForRelease() {
            return verification.summary().readyForRelease();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlReport.junitXmlFile().toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlReport.junitXmlSha256());
            map.put("passed", passed());
            map.put("readyForRelease", readyForRelease());
            map.put("junitXmlContract", TrainingReportQualityProfileCiGateManifestJUnitXmlContract
                    .inspect(junitXmlReport.junitXml(), verification)
                    .toMap());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record ReportInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Map<String, Object> report,
            String json,
            String markdown,
            String junitXml,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256) {
        public ReportInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            report = immutableStringKeyMap(Objects.requireNonNull(report, "report must not be null"));
            json = Objects.requireNonNull(json, "json must not be null");
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            jsonSha256 = requireChecksum(jsonSha256, "jsonSha256");
            markdownSha256 = requireChecksum(markdownSha256, "markdownSha256");
            junitXmlSha256 = requireChecksum(junitXmlSha256, "junitXmlSha256");
        }

        public String format() {
            return stringValue(report.get("format"), "");
        }

        public boolean junitXmlWellFormed() {
            return TrainingReportXml.isWellFormed(junitXml);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("format", format());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlSha256);
            map.put("junitXmlWellFormed", junitXmlWellFormed());
            map.put("report", report);
            return Map.copyOf(map);
        }
    }

    public record ReportVerification(
            ReportInspection inspection,
            boolean formatValid,
            boolean jsonMatchesVerification,
            boolean markdownMatchesVerification,
            boolean junitXmlMatchesVerification,
            boolean junitXmlWellFormed,
            boolean junitXmlContractValid,
            TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection junitXmlContract,
            List<String> failures) {
        public ReportVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            junitXmlContract = Objects.requireNonNull(junitXmlContract, "junitXmlContract must not be null");
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
                return "Quality-profile CI gate manifest verification report verified for "
                        + inspection.directory() + ".";
            }
            return "Quality-profile CI gate manifest verification report failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("formatValid", formatValid);
            map.put("jsonMatchesVerification", jsonMatchesVerification);
            map.put("markdownMatchesVerification", markdownMatchesVerification);
            map.put("junitXmlMatchesVerification", junitXmlMatchesVerification);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("junitXmlContractValid", junitXmlContractValid);
            map.put("junitXmlContract", junitXmlContract.toMap());
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public static String renderJson(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        return TrainerJson.toJson(reportMap(verification)) + "\n";
    }

    public static String renderMarkdown(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportQualityProfileCiGateManifestSummary summary = resolved.summary();
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Quality Profile CI Gate Manifest Verification");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + (summary.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Ready for release:** `" + summary.readyForRelease() + "`");
        appendLine(markdown, "**Profile:** `" + inline(summary.profileId().isBlank() ? "unknown" : summary.profileId()) + "`");
        appendLine(markdown, "**Artifacts:** `" + summary.artifactCount() + "`");
        appendLine(markdown, "**Failures:** `" + summary.failureCount() + "`");
        appendLine(markdown, "**Manifest directory:** `" + inline(summary.directory().toString()) + "`");
        appendLine(markdown, "");
        appendFailureCategories(markdown, summary);
        appendFailures(markdown, resolved);
        appendJunitXmlContract(markdown, resolved);
        return markdown.toString();
    }

    public static ReportBundle write(
            Path outputDirectory,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return write(outputDirectory, verification, Options.defaults());
    }

    public static ReportBundle write(
            Path outputDirectory,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        String json = renderJson(resolvedVerification);
        String markdown = renderMarkdown(resolvedVerification);

        TrainerCheckpointIO.writeStringAtomically(jsonFile, json);
        TrainerCheckpointIO.writeStringAtomically(markdownFile, markdown);
        TrainingReportQualityProfileCiGateManifestJUnitXml.Report junitXmlReport =
                TrainingReportQualityProfileCiGateManifestJUnitXml.write(junitXmlFile, resolvedVerification);
        TrainingReportArtifactFingerprint jsonFingerprint = TrainingReportArtifactFingerprint.of(jsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint = TrainingReportArtifactFingerprint.of(markdownFile);

        return new ReportBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlReport,
                json,
                markdown,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                resolvedVerification);
    }

    public static ReportInspection read(Path outputDirectory) throws IOException {
        return read(outputDirectory, Options.defaults());
    }

    public static ReportInspection read(Path outputDirectory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()));
    }

    public static ReportInspection readFiles(Path jsonFile, Path markdownFile, Path junitXmlFile) throws IOException {
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
        TrainingReportArtifactFingerprint jsonFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJsonFile);
        TrainingReportArtifactFingerprint markdownFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedMarkdownFile);
        TrainingReportArtifactFingerprint junitXmlFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile);
        return new ReportInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                readReport(json, resolvedJsonFile),
                json,
                markdown,
                junitXml,
                jsonFingerprint.sha256(),
                markdownFingerprint.sha256(),
                junitXmlFingerprint.sha256());
    }

    public static ReportVerification verify(
            Path outputDirectory,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return verify(outputDirectory, verification, Options.defaults());
    }

    public static ReportVerification verify(
            Path outputDirectory,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification,
            Options options) throws IOException {
        return verify(read(outputDirectory, options), verification);
    }

    public static ReportVerification verify(
            ReportInspection inspection,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        ReportInspection resolvedInspection = Objects.requireNonNull(inspection, "inspection must not be null");
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        boolean formatValid = FORMAT.equals(resolvedInspection.format());
        boolean jsonMatches = renderJson(resolvedVerification).equals(resolvedInspection.json());
        boolean markdownMatches = renderMarkdown(resolvedVerification).equals(resolvedInspection.markdown());
        boolean junitXmlMatches = TrainingReportQualityProfileCiGateManifestJUnitXml
                .render(resolvedVerification)
                .equals(resolvedInspection.junitXml());
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection junitXmlContract =
                TrainingReportQualityProfileCiGateManifestJUnitXmlContract.inspect(
                        resolvedInspection.junitXml(),
                        resolvedVerification);
        boolean junitXmlWellFormed = junitXmlContract.wellFormed();
        boolean junitXmlContractValid = junitXmlContract.contractValid();
        List<String> failures = new ArrayList<>();
        if (!formatValid) {
            failures.add("Manifest verification report format mismatch: " + resolvedInspection.format());
        }
        if (!jsonMatches) {
            failures.add("Manifest verification report JSON content mismatch for "
                    + resolvedInspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Manifest verification report Markdown content mismatch for "
                    + resolvedInspection.markdownFile());
        }
        if (!junitXmlMatches) {
            failures.add("Manifest verification report JUnit XML content mismatch for "
                    + resolvedInspection.junitXmlFile());
        }
        if (!junitXmlWellFormed) {
            failures.add("Manifest verification report JUnit XML is not well-formed for "
                    + resolvedInspection.junitXmlFile());
        }
        if (!junitXmlContractValid) {
            failures.add("Manifest verification report JUnit XML contract mismatch for "
                    + resolvedInspection.junitXmlFile() + ": "
                    + String.join("; ", junitXmlContract.failures()));
        }
        return new ReportVerification(
                resolvedInspection,
                formatValid,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                junitXmlWellFormed,
                junitXmlContractValid,
                junitXmlContract,
                failures);
    }

    private static Map<String, Object> reportMap(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", FORMAT);
        map.put("summary", resolved.summary().toMap());
        map.put("junitXmlContract", TrainingReportQualityProfileCiGateManifestJUnitXmlContract
                .inspect(TrainingReportQualityProfileCiGateManifestJUnitXml.render(resolved), resolved)
                .toMap());
        map.put("verification", resolved.toMap());
        return Map.copyOf(map);
    }

    private static void appendFailureCategories(
            StringBuilder markdown,
            TrainingReportQualityProfileCiGateManifestSummary summary) {
        appendLine(markdown, "## Failure Categories");
        appendLine(markdown, "");
        if (summary.failedCategories().isEmpty()) {
            appendLine(markdown, "No manifest verification failure categories were reported.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown, "| Category | Count |");
        appendLine(markdown, "| --- | ---: |");
        for (String categoryId : summary.failedCategories()) {
            TrainingReportQualityProfileCiGateManifestFailureCategory category =
                    TrainingReportQualityProfileCiGateManifestFailureCategory.fromId(categoryId)
                            .orElse(TrainingReportQualityProfileCiGateManifestFailureCategory.UNKNOWN);
            appendLine(markdown, "| `" + inline(category.id()) + "` | " + summary.count(category) + " |");
        }
        appendLine(markdown, "");
    }

    private static void appendFailures(
            StringBuilder markdown,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        appendLine(markdown, "## Failures");
        appendLine(markdown, "");
        if (verification.structuredFailures().isEmpty()) {
            appendLine(markdown, "No manifest verification failures were found.");
            return;
        }
        for (TrainingReportQualityProfileCiGateManifestFailure failure : verification.structuredFailures()) {
            appendLine(markdown, "- `" + failure.category().id() + "` " + inline(failure.message()));
        }
    }

    private static void appendJunitXmlContract(
            StringBuilder markdown,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection contract =
                TrainingReportQualityProfileCiGateManifestJUnitXmlContract.inspect(
                        TrainingReportQualityProfileCiGateManifestJUnitXml.render(verification),
                        verification);
        appendLine(markdown, "");
        appendLine(markdown, "## JUnit XML Contract");
        appendLine(markdown, "");
        appendLine(markdown, "**Contract valid:** `" + contract.contractValid() + "`");
        appendLine(markdown, "**Well formed:** `" + contract.wellFormed() + "`");
        appendLine(markdown, "**Expected tests:** `" + contract.counts().expectedTestCount() + "`");
        appendLine(markdown, "**Declared tests:** `" + contract.counts().declaredTestCount() + "`");
        appendLine(markdown, "**Observed testcases:** `" + contract.testcaseCount() + "`");
        appendLine(markdown, "**Property count:** `" + contract.propertyCount() + "`");
        appendLine(markdown, "**Manifest status:** `" + inline(contract.manifestStatus()) + "`");
        appendLine(markdown, "**Manifest ready for release:** `"
                + inline(contract.manifestReadyForRelease()) + "`");
        appendLine(markdown, "**Testcases:** `" + inline(String.join(", ", contract.testcaseNames())) + "`");
        appendLine(markdown, "**Failures:** `" + contract.failures().size() + "`");
        if (!contract.failures().isEmpty()) {
            appendLine(markdown, "");
            for (String failure : contract.failures()) {
                appendLine(markdown, "- " + inline(failure));
            }
        }
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String resolved = value == null || value.isBlank() ? fallback : value.trim();
        if (resolved.contains("/") || resolved.contains("\\")) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path");
        }
        return resolved;
    }

    private static Map<String, Object> readReport(String json, Path jsonFile) throws IOException {
        try {
            Object parsed = TrainerJsonParser.parse(json);
            if (parsed instanceof Map<?, ?> map) {
                return immutableStringKeyMap(map);
            }
            throw new IOException("Expected JSON object manifest verification report at " + jsonFile);
        } catch (RuntimeException error) {
            throw new IOException("Invalid manifest verification report JSON at " + jsonFile
                    + ": " + error.getMessage(), error);
        }
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> raw) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new IllegalArgumentException("Manifest verification report contains a non-string key");
            }
            map.put(key, entry.getValue());
        }
        return Map.copyOf(map);
    }

    private static Path commonDirectory(Path first, Path second, Path third) {
        Path firstParent = parentDirectory(first);
        Path secondParent = parentDirectory(second);
        Path thirdParent = parentDirectory(third);
        if (firstParent.equals(secondParent) && firstParent.equals(thirdParent)) {
            return firstParent;
        }
        return firstParent;
    }

    private static Path commonDirectory(Path first, Path second) {
        Path left = parentDirectory(first);
        Path right = parentDirectory(second);
        if (left != null && left.equals(right)) {
            return left;
        }
        return left;
    }

    private static Path parentDirectory(Path path) {
        Path parent = path == null ? null : path.getParent();
        return parent == null ? Path.of(".").toAbsolutePath().normalize() : parent;
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static String requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return checksum.trim().toLowerCase();
    }

    private static String inline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.replace('\n', ' ').replace('\r', ' ').replace('`', '\'').trim();
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
