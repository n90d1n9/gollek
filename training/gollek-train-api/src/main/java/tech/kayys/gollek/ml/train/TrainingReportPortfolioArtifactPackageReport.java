package tech.kayys.gollek.ml.train;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.helpers.DefaultHandler;

/**
 * Renders and persists CI-friendly reports for portfolio package verification.
 */
public final class TrainingReportPortfolioArtifactPackageReport {
    public static final String FORMAT = "gollek.training.report.portfolio-package-verification.v1";
    public static final int SCHEMA_VERSION = 1;
    public static final String DEFAULT_JSON_FILE_NAME = "portfolio-package-verification.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "portfolio-package-verification.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "portfolio-package-verification.junit.xml";
    private static final List<String> ARTIFACT_VERIFICATION_COMPARISON_FIELDS = List.of(
            "passed",
            "jsonSha256Matches",
            "markdownSha256Matches",
            "leaderboardCsvSha256Matches",
            "comparisonMetricsCsvSha256Matches",
            "comparisonFindingsCsvSha256Matches",
            "actualJsonSha256",
            "actualMarkdownSha256",
            "actualLeaderboardCsvSha256",
            "actualComparisonMetricsCsvSha256",
            "actualComparisonFindingsCsvSha256",
            "expectedJsonSha256",
            "expectedMarkdownSha256",
            "expectedLeaderboardCsvSha256",
            "expectedComparisonMetricsCsvSha256",
            "expectedComparisonFindingsCsvSha256",
            "failures");
    private static final List<String> ARTIFACT_INSPECTION_COMPARISON_FIELDS = List.of(
            "directory",
            "jsonFile",
            "markdownFile",
            "leaderboardCsvFile",
            "comparisonMetricsCsvFile",
            "comparisonFindingsCsvFile",
            "jsonSha256",
            "markdownSha256",
            "leaderboardCsvSha256",
            "comparisonMetricsCsvSha256",
            "comparisonFindingsCsvSha256",
            "entryCount",
            "comparisonMetricCount",
            "comparisonFindingCount",
            "hasComparisons",
            "hasComparisonFindings",
            "export");

    private TrainingReportPortfolioArtifactPackageReport() {
    }

    public record Options(String jsonFileName, String markdownFileName, String junitXmlFileName) {
        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            junitXmlFileName = normalizeFileName(junitXmlFileName, DEFAULT_JUNIT_XML_FILE_NAME, "junitXmlFileName");
        }

        public Options(String jsonFileName, String markdownFileName) {
            this(jsonFileName, markdownFileName, DEFAULT_JUNIT_XML_FILE_NAME);
        }

        public static Options defaults() {
            return new Options(DEFAULT_JSON_FILE_NAME, DEFAULT_MARKDOWN_FILE_NAME, DEFAULT_JUNIT_XML_FILE_NAME);
        }
    }

    public record ReportBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        public ReportBundle {
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
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public String contentFingerprint() {
            return TrainingReportPortfolioArtifactPackageReport.contentFingerprint(verification);
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
            map.put("contentFingerprint", contentFingerprint());
            map.put("passed", passed());
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
            String markdown,
            String junitXml,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256) {
        public ReportInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            report = Map.copyOf(Objects.requireNonNull(report, "report must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
        }

        public String format() {
            Object value = report.get("format");
            return value == null ? "" : String.valueOf(value);
        }

        public int schemaVersion() {
            Integer value = intValue(report.get("schemaVersion"));
            return value == null ? -1 : value;
        }

        public boolean packagePassed() {
            Object value = report.get("passed");
            return value instanceof Boolean bool && bool.booleanValue();
        }

        public String contentFingerprint() {
            return stringValue(report.get("contentFingerprint"));
        }

        public int failureCount() {
            Object value = report.get("failures");
            return value instanceof List<?> failures ? failures.size() : 0;
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
            map.put("format", format());
            map.put("schemaVersion", schemaVersion());
            map.put("packagePassed", packagePassed());
            map.put("contentFingerprint", contentFingerprint());
            map.put("failureCount", failureCount());
            map.put("report", report);
            return Map.copyOf(map);
        }
    }

    public record ReportVerification(
            ReportInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean junitXmlSha256Matches,
            boolean jsonFormatMatches,
            boolean schemaVersionMatches,
            boolean jsonShapeValid,
            boolean contentFingerprintConsistent,
            boolean packageInspectionConsistent,
            boolean artifactInventoryConsistent,
            boolean artifactVerificationConsistent,
            boolean markdownMatchesJson,
            boolean junitXmlWellFormed,
            boolean junitXmlMatchesJson,
            List<String> failures) {
        public ReportVerification {
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
                return "Portfolio package verification report artifacts verified for "
                        + inspection.directory() + ".";
            }
            return "Portfolio package verification report artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("jsonFormatMatches", jsonFormatMatches);
            map.put("schemaVersionMatches", schemaVersionMatches);
            map.put("jsonShapeValid", jsonShapeValid);
            map.put("contentFingerprintConsistent", contentFingerprintConsistent);
            map.put("reportedContentFingerprint", inspection.contentFingerprint());
            if (jsonShapeValid) {
                map.put("computedContentFingerprint", contentFingerprint(inspection));
            }
            map.put("packageInspectionConsistent", packageInspectionConsistent);
            map.put("artifactInventoryConsistent", artifactInventoryConsistent);
            map.put("artifactVerificationConsistent", artifactVerificationConsistent);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
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

    public record ReportPackageConsistency(
            ReportVerification reportVerification,
            TrainingReportPortfolioArtifactManifest.ManifestVerification packageVerification,
            boolean reportMatchesPackage,
            List<String> failures) {
        public ReportPackageConsistency {
            reportVerification = Objects.requireNonNull(reportVerification, "reportVerification must not be null");
            packageVerification = Objects.requireNonNull(packageVerification, "packageVerification must not be null");
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public boolean packagePassed() {
            return packageVerification.passed();
        }

        public boolean gatePassed() {
            return passed() && packagePassed();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public void requireGatePassed() {
            if (!gatePassed()) {
                throw new IllegalStateException(gateMessage());
            }
        }

        public String message() {
            if (passed()) {
                return "Portfolio package verification report matches package "
                        + packageVerification.inspection().directory() + ".";
            }
            return "Portfolio package verification report consistency failed: "
                    + String.join("; ", failures) + ".";
        }

        public String gateMessage() {
            if (gatePassed()) {
                return "Portfolio package verification gate passed for "
                        + packageVerification.inspection().directory() + ".";
            }
            if (!passed()) {
                return message();
            }
            return "Portfolio package verification gate failed because the current package did not pass: "
                    + packageVerification.message();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("gatePassed", gatePassed());
            map.put("packagePassed", packagePassed());
            map.put("reportMatchesPackage", reportMatchesPackage);
            map.put("message", message());
            map.put("gateMessage", gateMessage());
            map.put("failures", failures);
            map.put("reportVerification", reportVerification.toMap());
            map.put("packageVerification", packageVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public static String renderJson(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return TrainerJson.toJson(reportMap(verification));
    }

    public static String contentFingerprint(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return contentFingerprintFromReport(reportPayloadMap(verification));
    }

    public static String contentFingerprint(ReportInspection inspection) {
        return contentFingerprintFromReport(
                Objects.requireNonNull(inspection, "inspection must not be null").report());
    }

    public static String renderMarkdown(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        TrainingReportPortfolioArtifactManifest.ManifestVerification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportPortfolioArtifactManifest.ManifestInspection inspection = resolved.inspection();
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Portfolio Package Verification");
        appendLine(markdown, "");
        appendLine(markdown, "**Verification:** `" + (resolved.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, resolved.message());
        appendLine(markdown, "");
        appendLine(markdown, "## Summary");
        appendLine(markdown, "");
        appendLine(markdown, "| Check | Result |");
        appendLine(markdown, "| --- | --- |");
        appendLine(markdown, summaryRow("Manifest checksum", resolved.manifestSha256Matches()));
        appendLine(markdown, summaryRow("Artifact bytes", resolved.artifactBytesMatch()));
        appendLine(markdown, summaryRow("Artifact checksums", resolved.artifactSha256Match()));
        appendLine(markdown, summaryRow("Package verification", resolved.passed()));
        appendLine(markdown, "");
        appendLine(markdown, "Entries: `" + inspection.entryCount() + "`");
        appendLine(markdown, "Comparison metrics: `" + inspection.comparisonMetricCount() + "`");
        appendLine(markdown, "Comparison findings: `" + inspection.comparisonFindingCount() + "`");
        appendLine(markdown, "Content fingerprint: `" + contentFingerprint(resolved) + "`");
        appendLine(markdown, "");
        appendArtifacts(markdown, inspection);
        if (!resolved.failures().isEmpty()) {
            appendLine(markdown, "## Failures");
            appendLine(markdown, "");
            for (String failure : resolved.failures()) {
                appendLine(markdown, "- " + escapeListItem(failure));
            }
            appendLine(markdown, "");
        }
        return markdown.toString();
    }

    public static String renderJunitXml(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        TrainingReportPortfolioArtifactManifest.ManifestVerification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportPortfolioArtifactManifest.ManifestInspection inspection = resolved.inspection();
        String markdown = renderMarkdown(resolved);
        int failures = (resolved.manifestSha256Matches() ? 0 : 1)
                + (resolved.artifactBytesMatch() ? 0 : 1)
                + (resolved.artifactSha256Match() ? 0 : 1)
                + (resolved.passed() ? 0 : 1);

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.portfolio.package\" tests=\"4\" failures=\""
                + failures + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "package.directory", inspection.directory().toString());
        property(xml, "package.passed", Boolean.toString(resolved.passed()));
        property(xml, "package.entryCount", Integer.toString(inspection.entryCount()));
        property(xml, "package.comparisonMetricCount", Integer.toString(inspection.comparisonMetricCount()));
        property(xml, "package.comparisonFindingCount", Integer.toString(inspection.comparisonFindingCount()));
        property(xml, "package.hasComparisons", Boolean.toString(inspection.hasComparisons()));
        property(xml, "package.hasComparisonFindings", Boolean.toString(inspection.hasComparisonFindings()));
        property(xml, "package.contentFingerprint", contentFingerprint(resolved));
        property(xml, "manifest.file", inspection.manifestFile().toString());
        property(xml, "manifest.sha256", inspection.manifestSha256());
        property(xml, "manifest.checksumMatches", Boolean.toString(resolved.manifestSha256Matches()));
        property(xml, "artifacts.bytesMatch", Boolean.toString(resolved.artifactBytesMatch()));
        property(xml, "artifacts.sha256Match", Boolean.toString(resolved.artifactSha256Match()));
        appendArtifactProperties(xml, inspection);
        appendLine(xml, "  </properties>");

        testcase(
                xml,
                "manifest-checksum",
                resolved.manifestSha256Matches(),
                "MANIFEST_CHECKSUM",
                "Portfolio package manifest checksum mismatch.",
                resolved.failures(),
                markdown);
        testcase(
                xml,
                "artifact-bytes",
                resolved.artifactBytesMatch(),
                "ARTIFACT_BYTES",
                "Portfolio package artifact byte count mismatch.",
                resolved.failures(),
                markdown);
        testcase(
                xml,
                "artifact-checksums",
                resolved.artifactSha256Match(),
                "ARTIFACT_CHECKSUM",
                "Portfolio package artifact checksum mismatch.",
                resolved.failures(),
                markdown);
        testcase(
                xml,
                "complete-package",
                resolved.passed(),
                "PACKAGE_VERIFICATION",
                resolved.message(),
                resolved.failures(),
                markdown);

        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    public static ReportBundle write(
            Path directory,
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) throws IOException {
        return write(directory, verification, Options.defaults());
    }

    public static ReportBundle write(
            Path directory,
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPortfolioArtifactManifest.ManifestVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        TrainerCheckpointIO.writeStringAtomically(jsonFile, renderJson(resolvedVerification) + "\n");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, renderMarkdown(resolvedVerification));
        TrainerCheckpointIO.writeStringAtomically(junitXmlFile, renderJunitXml(resolvedVerification));
        return new ReportBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(junitXmlFile),
                resolvedVerification);
    }

    public static ReportBundle verifyAndWrite(Path packageDirectory) throws IOException {
        return verifyAndWrite(packageDirectory, packageDirectory, Options.defaults());
    }

    public static ReportBundle verifyAndWrite(
            Path packageDirectory,
            Path reportDirectory) throws IOException {
        return verifyAndWrite(packageDirectory, reportDirectory, Options.defaults());
    }

    public static ReportBundle verifyAndWrite(
            Path packageDirectory,
            Path reportDirectory,
            Options options) throws IOException {
        TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                TrainingReportPortfolioArtifactPackage.verify(packageDirectory);
        return write(reportDirectory, verification, options);
    }

    public static ReportInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ReportInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()));
    }

    public static ReportInspection readFiles(
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
            throw new IOException("Portfolio package verification report JSON must be an object: "
                    + resolvedJsonFile);
        }
        return new ReportInspection(
                commonDirectory(resolvedJsonFile, resolvedMarkdownFile, resolvedJunitXmlFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                stringKeyMap(map),
                markdown,
                junitXml,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile));
    }

    public static ReportVerification verify(ReportBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(bundle.jsonFile(), bundle.markdownFile(), bundle.junitXmlFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.junitXmlSha256());
    }

    public static ReportVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256,
                Options.defaults());
    }

    public static ReportVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            Options options) throws IOException {
        return verify(
                read(directory, options),
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static ReportVerification verify(
            ReportInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) {
        ReportInspection resolved = Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedJunitXmlSha = normalizeChecksum(expectedJunitXmlSha256);
        boolean jsonMatches = normalizedJsonSha == null
                || normalizedJsonSha.equalsIgnoreCase(resolved.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(resolved.markdownSha256());
        boolean junitXmlMatches = normalizedJunitXmlSha == null
                || normalizedJunitXmlSha.equalsIgnoreCase(resolved.junitXmlSha256());
        boolean formatMatches = FORMAT.equals(resolved.format());
        boolean schemaVersionMatches = resolved.schemaVersion() == SCHEMA_VERSION;
        boolean junitXmlWellFormed = isWellFormedXml(resolved.junitXml());
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + resolved.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + resolved.markdownFile());
        }
        if (!junitXmlMatches) {
            failures.add("JUnit XML checksum mismatch for " + resolved.junitXmlFile());
        }
        if (!formatMatches) {
            failures.add("Unexpected JSON format '" + resolved.format() + "' in " + resolved.jsonFile());
        }
        boolean jsonShapeValid = validateReportShape(resolved, failures);
        if (!schemaVersionMatches) {
            failures.add("Unexpected JSON schemaVersion '" + resolved.schemaVersion()
                    + "' in " + resolved.jsonFile() + "; expected " + SCHEMA_VERSION);
        }
        boolean markdownMatchesJson = false;
        boolean junitXmlMatchesJson = false;
        boolean contentFingerprintConsistent = false;
        boolean packageInspectionConsistent = false;
        boolean artifactInventoryConsistent = false;
        boolean artifactVerificationConsistent = false;
        if (jsonShapeValid) {
            contentFingerprintConsistent = contentFingerprintConsistent(resolved.report(), failures);
            packageInspectionConsistent = packageInspectionConsistent(resolved.report(), failures);
            artifactInventoryConsistent = artifactInventoryConsistent(resolved.report(), failures);
            artifactVerificationConsistent = artifactVerificationConsistent(resolved.report(), failures);
            markdownMatchesJson = markdownMatchesJson(resolved, failures);
            if (junitXmlWellFormed) {
                junitXmlMatchesJson = junitXmlMatchesJson(resolved, failures);
            }
        }
        if (!junitXmlWellFormed) {
            failures.add("JUnit XML is not well-formed: " + resolved.junitXmlFile());
        }
        return new ReportVerification(
                resolved,
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJunitXmlSha,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                formatMatches,
                schemaVersionMatches,
                jsonShapeValid,
                contentFingerprintConsistent,
                packageInspectionConsistent,
                artifactInventoryConsistent,
                artifactVerificationConsistent,
                markdownMatchesJson,
                junitXmlWellFormed,
                junitXmlMatchesJson,
                failures);
    }

    public static ReportPackageConsistency verifyAgainstPackage(
            ReportBundle reportBundle,
            Path packageDirectory) throws IOException {
        return verifyAgainstPackage(
                verify(reportBundle),
                TrainingReportPortfolioArtifactPackage.verify(packageDirectory));
    }

    public static ReportPackageConsistency verifyAgainstPackage(
            Path reportDirectory,
            Path packageDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return verifyAgainstPackage(
                verify(reportDirectory, expectedJsonSha256, expectedMarkdownSha256, expectedJunitXmlSha256),
                TrainingReportPortfolioArtifactPackage.verify(packageDirectory));
    }

    public static ReportPackageConsistency verifyAgainstPackage(
            ReportVerification reportVerification,
            TrainingReportPortfolioArtifactManifest.ManifestVerification packageVerification) {
        ReportVerification resolvedReport =
                Objects.requireNonNull(reportVerification, "reportVerification must not be null");
        TrainingReportPortfolioArtifactManifest.ManifestVerification resolvedPackage =
                Objects.requireNonNull(packageVerification, "packageVerification must not be null");
        List<String> failures = new ArrayList<>(resolvedReport.failures());
        boolean matchesPackage = reportMatchesPackage(resolvedReport.inspection(), resolvedPackage, failures);
        if (!matchesPackage) {
            failures.add("Verification report does not match current package verification for "
                    + resolvedPackage.inspection().directory());
        }
        return new ReportPackageConsistency(
                resolvedReport,
                resolvedPackage,
                matchesPackage,
                failures);
    }

    private static Map<String, Object> reportMap(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        Map<String, Object> map = reportPayloadMap(verification);
        map.put("contentFingerprint", contentFingerprintFromReport(map));
        return Map.copyOf(map);
    }

    private static Map<String, Object> reportPayloadMap(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        TrainingReportPortfolioArtifactManifest.ManifestVerification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportPortfolioArtifactManifest.ManifestInspection inspection = resolved.inspection();
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("file", inspection.manifestFile().toString());
        manifest.put("sha256", inspection.manifestSha256());
        manifest.put("sha256Matches", resolved.manifestSha256Matches());

        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", FORMAT);
        map.put("schemaVersion", SCHEMA_VERSION);
        map.put("passed", resolved.passed());
        map.put("message", resolved.message());
        map.put("manifest", manifest);
        map.put("artifactBytesMatch", resolved.artifactBytesMatch());
        map.put("artifactSha256Match", resolved.artifactSha256Match());
        map.put("entryCount", inspection.entryCount());
        map.put("comparisonMetricCount", inspection.comparisonMetricCount());
        map.put("comparisonFindingCount", inspection.comparisonFindingCount());
        map.put("hasComparisons", inspection.hasComparisons());
        map.put("hasComparisonFindings", inspection.hasComparisonFindings());
        map.put("failures", resolved.failures());
        map.put("inspection", inspection.toMap());
        resolved.artifactVerificationOptional()
                .ifPresent(artifactVerification -> map.put("artifactVerification", artifactVerification.toMap()));
        return map;
    }

    private static boolean reportMatchesPackage(
            ReportInspection inspection,
            TrainingReportPortfolioArtifactManifest.ManifestVerification packageVerification,
            List<String> failures) {
        Map<String, Object> report = inspection.report();
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        TrainingReportPortfolioArtifactManifest.ManifestInspection packageInspection =
                packageVerification.inspection();
        boolean matches = true;
        matches &= compareField(
                failures,
                "passed",
                booleanValue(report.get("passed")),
                packageVerification.passed());
        matches &= compareField(
                failures,
                "manifest.file",
                stringValue(manifest.get("file")),
                packageInspection.manifestFile().toString());
        matches &= compareField(
                failures,
                "manifest.sha256",
                stringValue(manifest.get("sha256")),
                packageInspection.manifestSha256());
        matches &= compareField(
                failures,
                "manifest.sha256Matches",
                booleanValue(manifest.get("sha256Matches")),
                packageVerification.manifestSha256Matches());
        matches &= compareField(
                failures,
                "artifactBytesMatch",
                booleanValue(report.get("artifactBytesMatch")),
                packageVerification.artifactBytesMatch());
        matches &= compareField(
                failures,
                "artifactSha256Match",
                booleanValue(report.get("artifactSha256Match")),
                packageVerification.artifactSha256Match());
        matches &= compareField(
                failures,
                "entryCount",
                intValue(report.get("entryCount")),
                packageInspection.entryCount());
        matches &= compareField(
                failures,
                "comparisonMetricCount",
                intValue(report.get("comparisonMetricCount")),
                packageInspection.comparisonMetricCount());
        matches &= compareField(
                failures,
                "comparisonFindingCount",
                intValue(report.get("comparisonFindingCount")),
                packageInspection.comparisonFindingCount());
        matches &= compareField(
                failures,
                "hasComparisons",
                booleanValue(report.get("hasComparisons")),
                packageInspection.hasComparisons());
        matches &= compareField(
                failures,
                "hasComparisonFindings",
                booleanValue(report.get("hasComparisonFindings")),
                packageInspection.hasComparisonFindings());
        matches &= compareField(
                failures,
                "contentFingerprint",
                stringValue(report.get("contentFingerprint")),
                contentFingerprint(packageVerification));
        matches &= compareField(
                failures,
                "failures",
                stringList(report.get("failures")),
                packageVerification.failures());
        matches &= artifactVerificationMatchesPackage(report, packageVerification, failures);
        return matches;
    }

    private static boolean artifactVerificationMatchesPackage(
            Map<String, Object> report,
            TrainingReportPortfolioArtifactManifest.ManifestVerification packageVerification,
            List<String> failures) {
        Object reportValue = report.get("artifactVerification");
        var packageArtifactOptional = packageVerification.artifactVerificationOptional();
        if (packageArtifactOptional.isEmpty()) {
            if (reportValue == null) {
                return true;
            }
            failures.add("Verification report field 'artifactVerification' is stale (report=present, package=absent)");
            return false;
        }
        if (!(reportValue instanceof Map<?, ?> map)) {
            failures.add("Verification report field 'artifactVerification' is stale (report=absent, package=present)");
            return false;
        }
        Map<String, Object> reportArtifactVerification = stringKeyMap(map);
        Map<String, Object> packageArtifactVerification = packageArtifactOptional.orElseThrow().toMap();
        boolean matches = true;
        for (String field : ARTIFACT_VERIFICATION_COMPARISON_FIELDS) {
            matches &= compareField(
                    failures,
                    "artifactVerification." + field,
                    reportArtifactVerification.get(field),
                    packageArtifactVerification.get(field));
        }
        Map<String, Object> reportInspection = objectMap(reportArtifactVerification.get("inspection"));
        Map<String, Object> packageInspection = objectMap(packageArtifactVerification.get("inspection"));
        for (String field : ARTIFACT_INSPECTION_COMPARISON_FIELDS) {
            matches &= compareField(
                    failures,
                    "artifactVerification.inspection." + field,
                    reportInspection.get(field),
                    packageInspection.get(field));
        }
        return matches;
    }

    private static boolean compareField(List<String> failures, String name, Object actual, Object expected) {
        if (Objects.equals(actual, expected)) {
            return true;
        }
        failures.add("Verification report field '" + name + "' is stale (report="
                + actual + ", package=" + expected + ")");
        return false;
    }

    private static boolean validateReportShape(ReportInspection inspection, List<String> failures) {
        int initialFailures = failures.size();
        Map<String, Object> report = inspection.report();
        requireString(report, "format", "verification report", failures);
        requireNumber(report, "schemaVersion", "verification report", failures);
        requireString(report, "contentFingerprint", "verification report", failures);
        requireBoolean(report, "passed", "verification report", failures);
        requireString(report, "message", "verification report", failures);
        Map<String, Object> manifest = requireObject(report, "manifest", "verification report", failures);
        if (manifest != null) {
            requireString(manifest, "file", "verification report manifest", failures);
            requireString(manifest, "sha256", "verification report manifest", failures);
            requireBoolean(manifest, "sha256Matches", "verification report manifest", failures);
        }
        requireBoolean(report, "artifactBytesMatch", "verification report", failures);
        requireBoolean(report, "artifactSha256Match", "verification report", failures);
        requireNumber(report, "entryCount", "verification report", failures);
        requireNumber(report, "comparisonMetricCount", "verification report", failures);
        requireNumber(report, "comparisonFindingCount", "verification report", failures);
        requireBoolean(report, "hasComparisons", "verification report", failures);
        requireBoolean(report, "hasComparisonFindings", "verification report", failures);
        requireIterable(report, "failures", "verification report", failures);
        Map<String, Object> reportInspection = requireObject(report, "inspection", "verification report", failures);
        if (reportInspection != null) {
            validateReportInspectionShape(reportInspection, failures);
        }
        if (report.containsKey("artifactVerification")) {
            Map<String, Object> artifactVerification =
                    requireObject(report, "artifactVerification", "verification report", failures);
            if (artifactVerification != null) {
                validateArtifactVerificationShape(artifactVerification, failures);
            }
        }
        return failures.size() == initialFailures;
    }

    private static void validateReportInspectionShape(
            Map<String, Object> inspection,
            List<String> failures) {
        requireString(inspection, "directory", "verification report inspection", failures);
        requireString(inspection, "manifestFile", "verification report inspection", failures);
        requireNumber(inspection, "formatVersion", "verification report inspection", failures);
        requireString(inspection, "generatedAt", "verification report inspection", failures);
        requireString(inspection, "manifestSha256", "verification report inspection", failures);
        requireNumber(inspection, "entryCount", "verification report inspection", failures);
        requireNumber(inspection, "comparisonMetricCount", "verification report inspection", failures);
        requireNumber(inspection, "comparisonFindingCount", "verification report inspection", failures);
        requireBoolean(inspection, "hasComparisons", "verification report inspection", failures);
        requireBoolean(inspection, "hasComparisonFindings", "verification report inspection", failures);
        requireObject(inspection, "artifacts", "verification report inspection", failures);
        requireObject(inspection, "metadata", "verification report inspection", failures);
    }

    private static void validateArtifactVerificationShape(
            Map<String, Object> artifactVerification,
            List<String> failures) {
        requireBoolean(artifactVerification, "passed", "artifact verification", failures);
        requireBoolean(artifactVerification, "jsonSha256Matches", "artifact verification", failures);
        requireBoolean(artifactVerification, "markdownSha256Matches", "artifact verification", failures);
        requireBoolean(artifactVerification, "leaderboardCsvSha256Matches", "artifact verification", failures);
        requireBoolean(artifactVerification, "comparisonMetricsCsvSha256Matches", "artifact verification", failures);
        requireBoolean(artifactVerification, "comparisonFindingsCsvSha256Matches", "artifact verification", failures);
        requireString(artifactVerification, "actualJsonSha256", "artifact verification", failures);
        requireString(artifactVerification, "actualMarkdownSha256", "artifact verification", failures);
        requireString(artifactVerification, "actualLeaderboardCsvSha256", "artifact verification", failures);
        requireString(artifactVerification, "actualComparisonMetricsCsvSha256", "artifact verification", failures);
        requireString(artifactVerification, "actualComparisonFindingsCsvSha256", "artifact verification", failures);
        requireIterable(artifactVerification, "failures", "artifact verification", failures);
        Map<String, Object> inspection = requireObject(
                artifactVerification,
                "inspection",
                "artifact verification",
                failures);
        if (inspection != null) {
            validateArtifactInspectionShape(inspection, failures);
        }
    }

    private static void validateArtifactInspectionShape(
            Map<String, Object> inspection,
            List<String> failures) {
        requireString(inspection, "directory", "artifact inspection", failures);
        requireString(inspection, "jsonFile", "artifact inspection", failures);
        requireString(inspection, "markdownFile", "artifact inspection", failures);
        requireString(inspection, "leaderboardCsvFile", "artifact inspection", failures);
        requireString(inspection, "comparisonMetricsCsvFile", "artifact inspection", failures);
        requireString(inspection, "comparisonFindingsCsvFile", "artifact inspection", failures);
        requireString(inspection, "jsonSha256", "artifact inspection", failures);
        requireString(inspection, "markdownSha256", "artifact inspection", failures);
        requireString(inspection, "leaderboardCsvSha256", "artifact inspection", failures);
        requireString(inspection, "comparisonMetricsCsvSha256", "artifact inspection", failures);
        requireString(inspection, "comparisonFindingsCsvSha256", "artifact inspection", failures);
        requireNumber(inspection, "entryCount", "artifact inspection", failures);
        requireNumber(inspection, "comparisonMetricCount", "artifact inspection", failures);
        requireNumber(inspection, "comparisonFindingCount", "artifact inspection", failures);
        requireBoolean(inspection, "hasComparisons", "artifact inspection", failures);
        requireBoolean(inspection, "hasComparisonFindings", "artifact inspection", failures);
        requireObject(inspection, "export", "artifact inspection", failures);
    }

    private static boolean contentFingerprintConsistent(
            Map<String, Object> report,
            List<String> failures) {
        String expected = stringValue(report.get("contentFingerprint"));
        String actual = contentFingerprintFromReport(report);
        if (expected.equals(actual)) {
            return true;
        }
        failures.add("Verification report field 'contentFingerprint' is inconsistent "
                + "(report=" + expected + ", computed=" + actual + ")");
        return false;
    }

    private static boolean packageInspectionConsistent(
            Map<String, Object> report,
            List<String> failures) {
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        Map<String, Object> inspection = objectMap(report.get("inspection"));
        boolean matches = true;
        matches &= compareInternalField(
                failures,
                "manifest.file",
                manifest.get("file"),
                "inspection.manifestFile",
                inspection.get("manifestFile"));
        matches &= compareInternalField(
                failures,
                "manifest.sha256",
                manifest.get("sha256"),
                "inspection.manifestSha256",
                inspection.get("manifestSha256"));
        matches &= compareInternalField(
                failures,
                "entryCount",
                report.get("entryCount"),
                "inspection.entryCount",
                inspection.get("entryCount"));
        matches &= compareInternalField(
                failures,
                "comparisonMetricCount",
                report.get("comparisonMetricCount"),
                "inspection.comparisonMetricCount",
                inspection.get("comparisonMetricCount"));
        matches &= compareInternalField(
                failures,
                "comparisonFindingCount",
                report.get("comparisonFindingCount"),
                "inspection.comparisonFindingCount",
                inspection.get("comparisonFindingCount"));
        matches &= compareInternalField(
                failures,
                "hasComparisons",
                report.get("hasComparisons"),
                "inspection.hasComparisons",
                inspection.get("hasComparisons"));
        matches &= compareInternalField(
                failures,
                "hasComparisonFindings",
                report.get("hasComparisonFindings"),
                "inspection.hasComparisonFindings",
                inspection.get("hasComparisonFindings"));
        return matches;
    }

    private static boolean compareInternalField(
            List<String> failures,
            String reportField,
            Object reportValue,
            String inspectionField,
            Object inspectionValue) {
        if (Objects.equals(reportValue, inspectionValue)) {
            return true;
        }
        failures.add("Verification report field '" + reportField
                + "' does not match package inspection field '" + inspectionField
                + "' (report=" + reportValue + ", inspection=" + inspectionValue + ")");
        return false;
    }

    private static boolean artifactInventoryConsistent(
            Map<String, Object> report,
            List<String> failures) {
        Object value = report.get("artifactVerification");
        if (!(value instanceof Map<?, ?> map)) {
            return true;
        }
        Map<String, Object> artifactVerification = stringKeyMap(map);
        Map<String, Object> packageInspection = objectMap(report.get("inspection"));
        Map<String, Object> artifacts = objectMap(packageInspection.get("artifacts"));
        Map<String, Object> artifactInspection = objectMap(artifactVerification.get("inspection"));
        boolean matches = true;
        matches &= compareArtifactInventory(
                failures,
                artifacts,
                artifactInspection,
                artifactVerification,
                "json",
                "jsonFile",
                "expectedJsonSha256");
        matches &= compareArtifactInventory(
                failures,
                artifacts,
                artifactInspection,
                artifactVerification,
                "markdown",
                "markdownFile",
                "expectedMarkdownSha256");
        matches &= compareArtifactInventory(
                failures,
                artifacts,
                artifactInspection,
                artifactVerification,
                "leaderboardCsv",
                "leaderboardCsvFile",
                "expectedLeaderboardCsvSha256");
        matches &= compareArtifactInventory(
                failures,
                artifacts,
                artifactInspection,
                artifactVerification,
                "comparisonMetricsCsv",
                "comparisonMetricsCsvFile",
                "expectedComparisonMetricsCsvSha256");
        matches &= compareArtifactInventory(
                failures,
                artifacts,
                artifactInspection,
                artifactVerification,
                "comparisonFindingsCsv",
                "comparisonFindingsCsvFile",
                "expectedComparisonFindingsCsvSha256");
        return matches;
    }

    private static boolean compareArtifactInventory(
            List<String> failures,
            Map<String, Object> artifacts,
            Map<String, Object> artifactInspection,
            Map<String, Object> artifactVerification,
            String artifactName,
            String fileField,
            String expectedShaField) {
        Map<String, Object> artifact = objectMap(artifacts.get(artifactName));
        boolean matches = true;
        matches &= compareArtifactInventoryField(
                failures,
                "inspection.artifacts." + artifactName + ".name",
                artifact.get("name"),
                "expected artifact name '" + artifactName + "'",
                artifactName);
        matches &= compareArtifactInventoryField(
                failures,
                "inspection.artifacts." + artifactName + ".file",
                artifact.get("file"),
                "artifact verification inspection field '" + fileField + "'",
                artifactInspection.get(fileField));
        matches &= compareArtifactInventoryField(
                failures,
                "inspection.artifacts." + artifactName + ".sha256",
                artifact.get("sha256"),
                "artifact verification field '" + expectedShaField + "'",
                artifactVerification.get(expectedShaField));
        return matches;
    }

    private static boolean compareArtifactInventoryField(
            List<String> failures,
            String inventoryField,
            Object inventoryValue,
            String expectedField,
            Object expectedValue) {
        if (Objects.equals(inventoryValue, expectedValue)) {
            return true;
        }
        failures.add("Verification report artifact inventory field '" + inventoryField
                + "' does not match " + expectedField
                + " (inventory=" + inventoryValue + ", expected=" + expectedValue + ")");
        return false;
    }

    private static boolean artifactVerificationConsistent(
            Map<String, Object> report,
            List<String> failures) {
        Object value = report.get("artifactVerification");
        if (!(value instanceof Map<?, ?> map)) {
            return true;
        }
        Map<String, Object> artifactVerification = stringKeyMap(map);
        Map<String, Object> inspection = objectMap(artifactVerification.get("inspection"));
        boolean matches = true;
        matches &= compareArtifactVerificationInternalField(
                failures,
                "actualJsonSha256",
                artifactVerification.get("actualJsonSha256"),
                "jsonSha256",
                inspection.get("jsonSha256"));
        matches &= compareArtifactVerificationInternalField(
                failures,
                "actualMarkdownSha256",
                artifactVerification.get("actualMarkdownSha256"),
                "markdownSha256",
                inspection.get("markdownSha256"));
        matches &= compareArtifactVerificationInternalField(
                failures,
                "actualLeaderboardCsvSha256",
                artifactVerification.get("actualLeaderboardCsvSha256"),
                "leaderboardCsvSha256",
                inspection.get("leaderboardCsvSha256"));
        matches &= compareArtifactVerificationInternalField(
                failures,
                "actualComparisonMetricsCsvSha256",
                artifactVerification.get("actualComparisonMetricsCsvSha256"),
                "comparisonMetricsCsvSha256",
                inspection.get("comparisonMetricsCsvSha256"));
        matches &= compareArtifactVerificationInternalField(
                failures,
                "actualComparisonFindingsCsvSha256",
                artifactVerification.get("actualComparisonFindingsCsvSha256"),
                "comparisonFindingsCsvSha256",
                inspection.get("comparisonFindingsCsvSha256"));
        matches &= compareArtifactVerificationMatchFlag(
                failures,
                artifactVerification,
                "jsonSha256Matches",
                "expectedJsonSha256",
                "actualJsonSha256");
        matches &= compareArtifactVerificationMatchFlag(
                failures,
                artifactVerification,
                "markdownSha256Matches",
                "expectedMarkdownSha256",
                "actualMarkdownSha256");
        matches &= compareArtifactVerificationMatchFlag(
                failures,
                artifactVerification,
                "leaderboardCsvSha256Matches",
                "expectedLeaderboardCsvSha256",
                "actualLeaderboardCsvSha256");
        matches &= compareArtifactVerificationMatchFlag(
                failures,
                artifactVerification,
                "comparisonMetricsCsvSha256Matches",
                "expectedComparisonMetricsCsvSha256",
                "actualComparisonMetricsCsvSha256");
        matches &= compareArtifactVerificationMatchFlag(
                failures,
                artifactVerification,
                "comparisonFindingsCsvSha256Matches",
                "expectedComparisonFindingsCsvSha256",
                "actualComparisonFindingsCsvSha256");
        boolean expectedPassed = stringList(artifactVerification.get("failures")).isEmpty();
        Boolean actualPassed = booleanValue(artifactVerification.get("passed"));
        if (!Objects.equals(actualPassed, expectedPassed)) {
            failures.add("Artifact verification field 'passed' is inconsistent with nested failures "
                    + "(passed=" + actualPassed + ", failuresEmpty=" + expectedPassed + ")");
            matches = false;
        }
        return matches;
    }

    private static boolean compareArtifactVerificationInternalField(
            List<String> failures,
            String artifactField,
            Object artifactValue,
            String inspectionField,
            Object inspectionValue) {
        if (Objects.equals(artifactValue, inspectionValue)) {
            return true;
        }
        failures.add("Artifact verification field '" + artifactField
                + "' does not match inspection field '" + inspectionField
                + "' (artifactVerification=" + artifactValue + ", inspection=" + inspectionValue + ")");
        return false;
    }

    private static boolean compareArtifactVerificationMatchFlag(
            List<String> failures,
            Map<String, Object> artifactVerification,
            String flagField,
            String expectedField,
            String actualField) {
        Object expectedValue = artifactVerification.get(expectedField);
        if (!(expectedValue instanceof String expected) || expected.isBlank()) {
            return true;
        }
        Object actualValue = artifactVerification.get(actualField);
        Boolean actualFlag = booleanValue(artifactVerification.get(flagField));
        boolean expectedFlag = expected.equalsIgnoreCase(String.valueOf(actualValue));
        if (Objects.equals(actualFlag, expectedFlag)) {
            return true;
        }
        failures.add("Artifact verification match flag '" + flagField + "' is inconsistent "
                + "(flag=" + actualFlag + ", expected=" + expected + ", actual=" + actualValue + ")");
        return false;
    }

    private static boolean markdownMatchesJson(ReportInspection inspection, List<String> failures) {
        Map<String, Object> report = inspection.report();
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        String markdown = inspection.markdown();
        boolean passed = Boolean.TRUE.equals(booleanValue(report.get("passed")));
        boolean matches = true;
        matches &= requireMarkdownContains(failures, markdown, "title",
                "# Gollek Training Portfolio Package Verification");
        matches &= requireMarkdownContains(failures, markdown, "verification status",
                "**Verification:** `" + passFail(passed) + "`");
        matches &= requireMarkdownContains(failures, markdown, "manifest checksum",
                summaryRow("Manifest checksum", Boolean.TRUE.equals(booleanValue(manifest.get("sha256Matches")))));
        matches &= requireMarkdownContains(failures, markdown, "artifact bytes",
                summaryRow("Artifact bytes", Boolean.TRUE.equals(booleanValue(report.get("artifactBytesMatch")))));
        matches &= requireMarkdownContains(failures, markdown, "artifact checksums",
                summaryRow("Artifact checksums", Boolean.TRUE.equals(booleanValue(report.get("artifactSha256Match")))));
        matches &= requireMarkdownContains(failures, markdown, "package verification",
                summaryRow("Package verification", passed));
        matches &= requireMarkdownContains(failures, markdown, "entry count",
                "Entries: `" + intValue(report.get("entryCount")) + "`");
        matches &= requireMarkdownContains(failures, markdown, "comparison metric count",
                "Comparison metrics: `" + intValue(report.get("comparisonMetricCount")) + "`");
        matches &= requireMarkdownContains(failures, markdown, "comparison finding count",
                "Comparison findings: `" + intValue(report.get("comparisonFindingCount")) + "`");
        matches &= requireMarkdownContains(failures, markdown, "content fingerprint",
                "Content fingerprint: `" + stringValue(report.get("contentFingerprint")) + "`");
        matches &= artifactMarkdownMatchesJson(failures, markdown, report);
        for (String failure : stringList(report.get("failures"))) {
            matches &= requireMarkdownContains(failures, markdown, "failure", "- " + escapeListItem(failure));
        }
        return matches;
    }

    private static boolean artifactMarkdownMatchesJson(
            List<String> failures,
            String markdown,
            Map<String, Object> report) {
        Map<String, Object> reportInspection = objectMap(report.get("inspection"));
        Map<String, Object> artifacts = objectMap(reportInspection.get("artifacts"));
        boolean matches = true;
        matches &= requireMarkdownContains(failures, markdown, "artifact table header",
                "| Artifact | Path | Bytes | SHA-256 |");
        matches &= requireMarkdownContains(failures, markdown, "artifact table separator",
                "| --- | --- | ---: | --- |");
        for (String artifactName : sortedKeys(artifacts)) {
            matches &= requireMarkdownContains(
                    failures,
                    markdown,
                    "artifact " + artifactName + " row",
                    artifactMarkdownRow(objectMap(artifacts.get(artifactName))));
        }
        return matches;
    }

    private static String artifactMarkdownRow(Map<String, Object> artifact) {
        return "| `" + escapeTable(stringValue(artifact.get("name"))) + "`"
                + " | `" + escapeTable(stringValue(artifact.get("file"))) + "`"
                + " | " + stringValue(artifact.get("bytes"))
                + " | `" + escapeTable(shortSha(stringValue(artifact.get("sha256")))) + "`"
                + " |";
    }

    private static boolean requireMarkdownContains(
            List<String> failures,
            String markdown,
            String field,
            String expected) {
        if (markdown.contains(expected)) {
            return true;
        }
        failures.add("Markdown report does not match JSON field '" + field + "' (expected fragment: "
                + expected + ")");
        return false;
    }

    private static boolean junitXmlMatchesJson(ReportInspection inspection, List<String> failures) {
        Document document = parseXmlDocument(inspection.junitXml());
        if (document == null) {
            return false;
        }
        Map<String, Object> report = inspection.report();
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        Map<String, Object> reportInspection = objectMap(report.get("inspection"));
        Map<String, Object> artifacts = objectMap(reportInspection.get("artifacts"));
        Element root = document.getDocumentElement();
        boolean matches = true;
        matches &= compareReportArtifactField(failures, "name", root.getAttribute("name"),
                "gollek.training.portfolio.package", "JUnit XML");
        matches &= compareReportArtifactField(failures, "tests", root.getAttribute("tests"), "4", "JUnit XML");
        matches &= compareReportArtifactField(
                failures,
                "failures",
                root.getAttribute("failures"),
                Integer.toString(expectedJunitFailureCount(report)),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.passed",
                junitProperty(document, "package.passed"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(report.get("passed")))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.entryCount",
                junitProperty(document, "package.entryCount"),
                String.valueOf(intValue(report.get("entryCount"))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.comparisonMetricCount",
                junitProperty(document, "package.comparisonMetricCount"),
                String.valueOf(intValue(report.get("comparisonMetricCount"))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.comparisonFindingCount",
                junitProperty(document, "package.comparisonFindingCount"),
                String.valueOf(intValue(report.get("comparisonFindingCount"))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.hasComparisons",
                junitProperty(document, "package.hasComparisons"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(report.get("hasComparisons")))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.hasComparisonFindings",
                junitProperty(document, "package.hasComparisonFindings"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(report.get("hasComparisonFindings")))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "package.contentFingerprint",
                junitProperty(document, "package.contentFingerprint"),
                stringValue(report.get("contentFingerprint")),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "manifest.sha256",
                junitProperty(document, "manifest.sha256"),
                stringValue(manifest.get("sha256")),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "manifest.checksumMatches",
                junitProperty(document, "manifest.checksumMatches"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(manifest.get("sha256Matches")))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "artifacts.bytesMatch",
                junitProperty(document, "artifacts.bytesMatch"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(report.get("artifactBytesMatch")))),
                "JUnit XML");
        matches &= compareReportArtifactField(failures, "artifacts.sha256Match",
                junitProperty(document, "artifacts.sha256Match"),
                String.valueOf(Boolean.TRUE.equals(booleanValue(report.get("artifactSha256Match")))),
                "JUnit XML");
        matches &= artifactJunitPropertiesMatchJson(failures, document, artifacts);
        return matches;
    }

    private static boolean artifactJunitPropertiesMatchJson(
            List<String> failures,
            Document document,
            Map<String, Object> artifacts) {
        boolean matches = true;
        for (String artifactName : sortedKeys(artifacts)) {
            Map<String, Object> artifact = objectMap(artifacts.get(artifactName));
            matches &= compareReportArtifactField(
                    failures,
                    "artifact." + artifactName + ".file",
                    junitProperty(document, "artifact." + artifactName + ".file"),
                    stringValue(artifact.get("file")),
                    "JUnit XML");
            matches &= compareReportArtifactField(
                    failures,
                    "artifact." + artifactName + ".bytes",
                    junitProperty(document, "artifact." + artifactName + ".bytes"),
                    stringValue(artifact.get("bytes")),
                    "JUnit XML");
            matches &= compareReportArtifactField(
                    failures,
                    "artifact." + artifactName + ".sha256",
                    junitProperty(document, "artifact." + artifactName + ".sha256"),
                    stringValue(artifact.get("sha256")),
                    "JUnit XML");
        }
        return matches;
    }

    private static int expectedJunitFailureCount(Map<String, Object> report) {
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        return (Boolean.TRUE.equals(booleanValue(manifest.get("sha256Matches"))) ? 0 : 1)
                + (Boolean.TRUE.equals(booleanValue(report.get("artifactBytesMatch"))) ? 0 : 1)
                + (Boolean.TRUE.equals(booleanValue(report.get("artifactSha256Match"))) ? 0 : 1)
                + (Boolean.TRUE.equals(booleanValue(report.get("passed"))) ? 0 : 1);
    }

    private static boolean compareReportArtifactField(
            List<String> failures,
            String field,
            String actual,
            String expected,
            String artifact) {
        if (Objects.equals(actual, expected)) {
            return true;
        }
        failures.add(artifact + " report does not match JSON field '" + field + "' (artifact="
                + actual + ", json=" + expected + ")");
        return false;
    }

    private static String junitProperty(Document document, String name) {
        NodeList nodes = document.getElementsByTagName("property");
        for (int index = 0; index < nodes.getLength(); index++) {
            if (nodes.item(index) instanceof Element element && name.equals(element.getAttribute("name"))) {
                return element.getAttribute("value");
            }
        }
        return "";
    }

    private static void requireString(
            Map<String, Object> values,
            String key,
            String owner,
            List<String> failures) {
        Object value = values.get(key);
        if (!(value instanceof String string) || string.isBlank()) {
            failures.add(owner + " is missing required string field '" + key + "'");
        }
    }

    private static void requireBoolean(
            Map<String, Object> values,
            String key,
            String owner,
            List<String> failures) {
        if (!(values.get(key) instanceof Boolean)) {
            failures.add(owner + " is missing required boolean field '" + key + "'");
        }
    }

    private static void requireNumber(
            Map<String, Object> values,
            String key,
            String owner,
            List<String> failures) {
        if (!(values.get(key) instanceof Number)) {
            failures.add(owner + " is missing required numeric field '" + key + "'");
        }
    }

    private static Map<String, Object> requireObject(
            Map<String, Object> values,
            String key,
            String owner,
            List<String> failures) {
        Object value = values.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            failures.add(owner + " is missing required object field '" + key + "'");
            return null;
        }
        return stringKeyMap(map);
    }

    private static void requireIterable(
            Map<String, Object> values,
            String key,
            String owner,
            List<String> failures) {
        if (!(values.get(key) instanceof Iterable<?>)) {
            failures.add(owner + " is missing required array field '" + key + "'");
        }
    }

    private static void testcase(
            StringBuilder xml,
            String name,
            boolean passed,
            String failureType,
            String message,
            List<String> failures,
            String markdown) {
        appendLine(xml, "  <testcase classname=\"gollek.training.portfolio.package\" name=\""
                + escapeXml(name) + "\" time=\"0\">");
        if (!passed) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType)
                    + "\" message=\"" + escapeXml(message) + "\">");
            appendLine(xml, escapeText(String.join("\n", failures)));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
        appendLine(xml, "  </testcase>");
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static void appendArtifactProperties(
            StringBuilder xml,
            TrainingReportPortfolioArtifactManifest.ManifestInspection inspection) {
        for (TrainingReportPortfolioArtifactManifest.ArtifactEntry artifact : inspection.artifacts().values()) {
            String prefix = "artifact." + artifact.name();
            property(xml, prefix + ".file", artifact.file().toString());
            property(xml, prefix + ".bytes", Long.toString(artifact.bytes()));
            property(xml, prefix + ".sha256", artifact.sha256());
        }
    }

    private static void appendArtifacts(
            StringBuilder markdown,
            TrainingReportPortfolioArtifactManifest.ManifestInspection inspection) {
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | Bytes | SHA-256 |");
        appendLine(markdown, "| --- | --- | ---: | --- |");
        for (TrainingReportPortfolioArtifactManifest.ArtifactEntry artifact : inspection.artifacts().values()) {
            appendLine(markdown, "| `" + escapeTable(artifact.name()) + "`"
                    + " | `" + escapeTable(artifact.file().toString()) + "`"
                    + " | " + artifact.bytes()
                    + " | `" + escapeTable(shortSha(artifact.sha256())) + "`"
                    + " |");
        }
        appendLine(markdown, "");
    }

    private static String contentFingerprintFromReport(Map<String, Object> report) {
        Map<String, Object> manifest = objectMap(report.get("manifest"));
        Map<String, Object> inspection = objectMap(report.get("inspection"));
        Map<String, Object> artifacts = objectMap(inspection.get("artifacts"));
        Map<String, Object> artifactVerification = objectMap(report.get("artifactVerification"));
        Map<String, Object> artifactInspection = objectMap(artifactVerification.get("inspection"));
        StringBuilder canonical = new StringBuilder();

        appendFingerprintField(canonical, "format", report.get("format"));
        appendFingerprintField(canonical, "schemaVersion", report.get("schemaVersion"));
        appendFingerprintField(canonical, "passed", report.get("passed"));
        appendFingerprintField(canonical, "manifest.sha256", manifest.get("sha256"));
        appendFingerprintField(canonical, "manifest.sha256Matches", manifest.get("sha256Matches"));
        appendFingerprintField(canonical, "artifactBytesMatch", report.get("artifactBytesMatch"));
        appendFingerprintField(canonical, "artifactSha256Match", report.get("artifactSha256Match"));
        appendFingerprintField(canonical, "entryCount", report.get("entryCount"));
        appendFingerprintField(canonical, "comparisonMetricCount", report.get("comparisonMetricCount"));
        appendFingerprintField(canonical, "comparisonFindingCount", report.get("comparisonFindingCount"));
        appendFingerprintField(canonical, "hasComparisons", report.get("hasComparisons"));
        appendFingerprintField(canonical, "hasComparisonFindings", report.get("hasComparisonFindings"));
        appendFingerprintList(canonical, "failures", stringList(report.get("failures")));
        appendFingerprintField(canonical, "inspection.formatVersion", inspection.get("formatVersion"));
        appendFingerprintField(canonical, "inspection.generatedAt", inspection.get("generatedAt"));
        appendFingerprintField(canonical, "inspection.manifestSha256", inspection.get("manifestSha256"));
        appendFingerprintMap(canonical, "inspection.metadata", objectMap(inspection.get("metadata")));

        List<String> artifactNames = new ArrayList<>(artifacts.keySet());
        artifactNames.sort(Comparator.naturalOrder());
        appendFingerprintField(canonical, "artifact.count", artifactNames.size());
        for (String artifactName : artifactNames) {
            Map<String, Object> artifact = objectMap(artifacts.get(artifactName));
            appendFingerprintField(canonical, "artifact." + artifactName + ".name", artifact.get("name"));
            appendFingerprintField(canonical, "artifact." + artifactName + ".bytes", artifact.get("bytes"));
            appendFingerprintField(canonical, "artifact." + artifactName + ".sha256", artifact.get("sha256"));
        }

        appendFingerprintField(canonical, "artifactVerification.present", !artifactVerification.isEmpty());
        if (!artifactVerification.isEmpty()) {
            appendFingerprintField(canonical, "artifactVerification.passed", artifactVerification.get("passed"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.jsonSha256Matches",
                    artifactVerification.get("jsonSha256Matches"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.markdownSha256Matches",
                    artifactVerification.get("markdownSha256Matches"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.leaderboardCsvSha256Matches",
                    artifactVerification.get("leaderboardCsvSha256Matches"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.comparisonMetricsCsvSha256Matches",
                    artifactVerification.get("comparisonMetricsCsvSha256Matches"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.comparisonFindingsCsvSha256Matches",
                    artifactVerification.get("comparisonFindingsCsvSha256Matches"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.actualJsonSha256",
                    artifactVerification.get("actualJsonSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.actualMarkdownSha256",
                    artifactVerification.get("actualMarkdownSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.actualLeaderboardCsvSha256",
                    artifactVerification.get("actualLeaderboardCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.actualComparisonMetricsCsvSha256",
                    artifactVerification.get("actualComparisonMetricsCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.actualComparisonFindingsCsvSha256",
                    artifactVerification.get("actualComparisonFindingsCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.expectedJsonSha256",
                    artifactVerification.get("expectedJsonSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.expectedMarkdownSha256",
                    artifactVerification.get("expectedMarkdownSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.expectedLeaderboardCsvSha256",
                    artifactVerification.get("expectedLeaderboardCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.expectedComparisonMetricsCsvSha256",
                    artifactVerification.get("expectedComparisonMetricsCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.expectedComparisonFindingsCsvSha256",
                    artifactVerification.get("expectedComparisonFindingsCsvSha256"));
            appendFingerprintList(
                    canonical,
                    "artifactVerification.failures",
                    stringList(artifactVerification.get("failures")));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.jsonSha256",
                    artifactInspection.get("jsonSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.markdownSha256",
                    artifactInspection.get("markdownSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.leaderboardCsvSha256",
                    artifactInspection.get("leaderboardCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.comparisonMetricsCsvSha256",
                    artifactInspection.get("comparisonMetricsCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.comparisonFindingsCsvSha256",
                    artifactInspection.get("comparisonFindingsCsvSha256"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.entryCount",
                    artifactInspection.get("entryCount"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.comparisonMetricCount",
                    artifactInspection.get("comparisonMetricCount"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.comparisonFindingCount",
                    artifactInspection.get("comparisonFindingCount"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.hasComparisons",
                    artifactInspection.get("hasComparisons"));
            appendFingerprintField(
                    canonical,
                    "artifactVerification.inspection.hasComparisonFindings",
                    artifactInspection.get("hasComparisonFindings"));
        }

        return sha256Hex(canonical.toString());
    }

    private static void appendFingerprintMap(
            StringBuilder canonical,
            String key,
            Map<String, Object> values) {
        List<String> names = new ArrayList<>(values.keySet());
        names.sort(Comparator.naturalOrder());
        appendFingerprintField(canonical, key + ".count", names.size());
        for (String name : names) {
            appendFingerprintField(canonical, key + "." + name, values.get(name));
        }
    }

    private static void appendFingerprintList(
            StringBuilder canonical,
            String key,
            List<String> values) {
        appendFingerprintField(canonical, key + ".count", values.size());
        for (int index = 0; index < values.size(); index++) {
            appendFingerprintField(canonical, key + "." + index, values.get(index));
        }
    }

    private static void appendFingerprintField(StringBuilder canonical, String key, Object value) {
        String normalized = value == null ? "" : String.valueOf(value);
        canonical.append(key.length())
                .append(':')
                .append(key)
                .append('=')
                .append(normalized.length())
                .append(':')
                .append(normalized)
                .append('\n');
    }

    private static String sha256Hex(String value) {
        MessageDigest digest = newSha256Digest();
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        digest.update(bytes, 0, bytes.length);
        byte[] hashed = digest.digest();
        StringBuilder result = new StringBuilder(hashed.length * 2);
        for (byte item : hashed) {
            result.append(Character.forDigit((item >>> 4) & 0x0f, 16));
            result.append(Character.forDigit(item & 0x0f, 16));
        }
        return result.toString();
    }

    private static MessageDigest newSha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 digest is not available", error);
        }
    }

    private static String summaryRow(String label, boolean passed) {
        return "| " + escapeTable(label) + " | `" + passFail(passed) + "` |";
    }

    private static String passFail(boolean passed) {
        return passed ? "PASS" : "FAIL";
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

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
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

    private static Map<String, Object> stringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
    }

    private static List<String> sortedKeys(Map<String, Object> values) {
        List<String> names = new ArrayList<>(values.keySet());
        names.sort(Comparator.naturalOrder());
        return names;
    }

    private static Map<String, Object> objectMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        return stringKeyMap(map);
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static Boolean booleanValue(Object value) {
        return value instanceof Boolean bool ? bool : null;
    }

    private static Integer intValue(Object value) {
        return value instanceof Number number ? number.intValue() : null;
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof List<?> values)) {
            return List.of();
        }
        List<String> strings = new ArrayList<>(values.size());
        for (Object item : values) {
            strings.add(String.valueOf(item));
        }
        return List.copyOf(strings);
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

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(Locale.ROOT);
    }

    private static boolean isWellFormedXml(String xml) {
        return parseXmlDocument(xml) != null;
    }

    private static Document parseXmlDocument(String xml) {
        return TrainingReportXml.parseDocument(xml);
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
}
