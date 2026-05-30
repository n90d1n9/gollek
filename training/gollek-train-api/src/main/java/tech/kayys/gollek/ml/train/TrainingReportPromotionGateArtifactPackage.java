package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * One-call packaging API for promotion gate artifacts plus their provenance manifest.
 */
public final class TrainingReportPromotionGateArtifactPackage {
    public static final String DEFAULT_VERIFICATION_REPORT_FILE_NAME = "promotion-gate-package-verification.json";
    public static final String DEFAULT_VERIFICATION_MARKDOWN_FILE_NAME = "promotion-gate-package-verification.md";
    public static final String DEFAULT_VERIFICATION_JUNIT_XML_FILE_NAME =
            "promotion-gate-package-verification.junit.xml";
    public static final String DEFAULT_VERIFICATION_INDEX_FILE_NAME =
            "promotion-gate-package-verification.index.json";
    public static final String DEFAULT_VERIFICATION_INDEX_RECEIPT_FILE_NAME =
            "promotion-gate-package-verification.index.receipt.json";
    public static final String DEFAULT_VERIFICATION_INDEX_PACKAGE_AUDIT_FILE_NAME =
            "promotion-gate-package-verification.index.package-audit.json";
    public static final String DEFAULT_VERIFICATION_REPORT_BUNDLE_RECEIPT_FILE_NAME =
            "promotion-gate-package-verification.reports.receipt.json";
    public static final String DEFAULT_VERIFICATION_EVIDENCE_FILE_NAME =
            "promotion-gate-package-verification.evidence.json";
    public static final String DEFAULT_VERIFICATION_EVIDENCE_RECEIPT_FILE_NAME =
            "promotion-gate-package-verification.evidence.receipt.json";
    public static final String VERIFICATION_INDEX_FORMAT =
            "gollek.training.promotion.package.verification.index.v1";
    public static final String VERIFICATION_INDEX_RECEIPT_FORMAT =
            "gollek.training.promotion.package.verification.index.receipt.v1";
    public static final String VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT =
            "gollek.training.promotion.package.verification.index.package-audit.v1";
    public static final String VERIFICATION_REPORT_FORMAT =
            "gollek.training.promotion.package.verification.report.v1";
    public static final String VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT =
            "gollek.training.promotion.package.verification.reports.receipt.v1";
    public static final String VERIFICATION_EVIDENCE_FORMAT =
            "gollek.training.promotion.package.verification.evidence.v1";
    public static final String VERIFICATION_EVIDENCE_RECEIPT_FORMAT =
            "gollek.training.promotion.package.verification.evidence.receipt.v1";

    private static final String REVIEW_JSON_ARTIFACT = "reviewJson";
    private static final String REVIEW_MARKDOWN_ARTIFACT = "reviewMarkdown";
    private static final String GATE_JSON_ARTIFACT = "json";
    private static final String GATE_MARKDOWN_ARTIFACT = "markdown";
    private static final String GATE_JUNIT_XML_ARTIFACT = "junitXml";
    private static final String SOURCE_REPORT_ARTIFACT_PREFIX = "sourceReport.";

    private TrainingReportPromotionGateArtifactPackage() {
    }

    public record Options(
            TrainingReportPromotionArtifacts.Options review,
            TrainingReportPromotionGateArtifacts.Options artifacts,
            TrainingReportPromotionGateArtifactManifest.Options manifest) {
        public Options {
            review = review == null ? TrainingReportPromotionArtifacts.Options.defaults() : review;
            artifacts = artifacts == null ? TrainingReportPromotionGateArtifacts.Options.defaults() : artifacts;
            manifest = manifest == null ? TrainingReportPromotionGateArtifactManifest.Options.defaults() : manifest;
        }

        public Options(
                TrainingReportPromotionGateArtifacts.Options artifacts,
                TrainingReportPromotionGateArtifactManifest.Options manifest) {
            this(TrainingReportPromotionArtifacts.Options.defaults(), artifacts, manifest);
        }

        public static Options defaults() {
            return new Options(
                    TrainingReportPromotionArtifacts.Options.defaults(),
                    TrainingReportPromotionGateArtifacts.Options.defaults(),
                    TrainingReportPromotionGateArtifactManifest.Options.defaults());
        }
    }

    public record PackageBundle(
            Path directory,
            TrainingReportPromotionArtifacts.ArtifactBundle review,
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest) {
        public PackageBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            review = Objects.requireNonNull(review, "review must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
        }

        public boolean passed() {
            return artifacts.passed();
        }

        public boolean promotable() {
            return artifacts.promotable();
        }

        public void requirePassed() {
            artifacts.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("review", review.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("manifest", manifest.toMap());
            return Map.copyOf(map);
        }
    }

    public record PackageInspection(
            Path directory,
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            TrainingReportPromotionArtifacts.ArtifactInspection review,
            TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts) {
        public PackageInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
            review = Objects.requireNonNull(review, "review must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
        }

        public boolean passed() {
            return artifacts.passed();
        }

        public boolean promotable() {
            return artifacts.promotable();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("manifest", manifest.toMap());
            map.put("review", review.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("sourceReportSnapshots", sourceReportSnapshots().stream()
                    .map(SourceReportSnapshot::toMap)
                    .toList());
            return Map.copyOf(map);
        }

        public List<SourceReportSnapshot> sourceReportSnapshots() {
            return TrainingReportPromotionGateArtifactPackage.sourceReportSnapshots(this);
        }
    }

    public record SourceReportSnapshot(
            TrainingReportPromotionArtifacts.SourceReport sourceReport,
            TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact) {
        public SourceReportSnapshot {
            sourceReport = Objects.requireNonNull(sourceReport, "sourceReport must not be null");
            artifact = Objects.requireNonNull(artifact, "artifact must not be null");
        }

        public String role() {
            return sourceReport.role();
        }

        public String name() {
            return sourceReport.name();
        }

        public Path snapshotFile() {
            return artifact.file();
        }

        public boolean manifestBytesMatchSource() {
            return sourceReport.bytes() != null && sourceReport.bytes().longValue() == artifact.bytes();
        }

        public boolean manifestSha256MatchesSource() {
            return sourceReport.sha256() != null && sourceReport.sha256().equalsIgnoreCase(artifact.sha256());
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role", role());
            map.put("name", name());
            map.put("snapshotArtifact", artifact.name());
            map.put("snapshotFile", snapshotFile().toString());
            map.put("snapshotBytes", artifact.bytes());
            map.put("snapshotSha256", artifact.sha256());
            map.put("manifestBytesMatchSource", manifestBytesMatchSource());
            map.put("manifestSha256MatchesSource", manifestSha256MatchesSource());
            map.put("sourceReport", sourceReport.toMap());
            return Map.copyOf(map);
        }
    }

    public record SourceSnapshotVerification(
            PackageInspection inspection,
            List<SourceReportSnapshot> snapshots,
            List<String> failures) {
        public SourceSnapshotVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            snapshots = snapshots == null ? List.of() : List.copyOf(snapshots);
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

        public List<String> expectedSourceReportArtifactNames() {
            return TrainingReportPromotionGateArtifactPackage.expectedSourceReportArtifactNames(inspection);
        }

        public List<String> presentSourceReportArtifactNames() {
            return TrainingReportPromotionGateArtifactPackage.presentSourceReportArtifactNames(inspection);
        }

        public List<String> missingSourceReportArtifactNames() {
            return TrainingReportPromotionGateArtifactPackage.missingSourceReportArtifactNames(inspection);
        }

        public List<String> unexpectedSourceReportArtifactNames() {
            return TrainingReportPromotionGateArtifactPackage.unexpectedSourceReportArtifactNames(inspection);
        }

        public String message() {
            if (passed()) {
                return "Promotion gate package source report snapshots verified for "
                        + inspection.directory() + ".";
            }
            return "Promotion gate package source report snapshot verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("snapshots", snapshots.stream().map(SourceReportSnapshot::toMap).toList());
            map.put("expectedSourceReportArtifacts", expectedSourceReportArtifactNames());
            map.put("presentSourceReportArtifacts", presentSourceReportArtifactNames());
            map.put("missingSourceReportArtifacts", missingSourceReportArtifactNames());
            map.put("unexpectedSourceReportArtifacts", unexpectedSourceReportArtifactNames());
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public record PackageVerification(
            PackageInspection inspection,
            TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification,
            SourceSnapshotVerification sourceSnapshotVerification,
            List<String> failures) {
        public PackageVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            manifestVerification = Objects.requireNonNull(manifestVerification, "manifestVerification must not be null");
            sourceSnapshotVerification = Objects.requireNonNull(
                    sourceSnapshotVerification,
                    "sourceSnapshotVerification must not be null");
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
                return "Promotion gate package verified for " + inspection.directory() + ".";
            }
            return "Promotion gate package verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("failures", failures);
            map.put("manifestVerification", manifestVerification.toMap());
            map.put("sourceSnapshotVerification", sourceSnapshotVerification.toMap());
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public record PackageRefresh(
            Path directory,
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts,
            PackageVerification verification,
            VerificationReportBundle reports,
            VerificationReportBundleVerification reportVerification) {
        public PackageRefresh {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
            reports = Objects.requireNonNull(reports, "reports must not be null");
            reportVerification = Objects.requireNonNull(reportVerification, "reportVerification must not be null");
        }

        public boolean passed() {
            return verification.passed() && reports.passed() && reportVerification.passed();
        }

        public boolean promotable() {
            return verification.inspection().promotable();
        }

        public void requirePassed() {
            verification.requirePassed();
            reports.requirePassed();
            reportVerification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("manifest", manifest.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("verification", verification.toMap());
            map.put("reports", reports.toMap());
            map.put("reportVerification", reportVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReport(
            Path reportFile,
            String reportSha256,
            PackageVerification verification) {
        public VerificationReport {
            reportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (reportSha256 == null || reportSha256.isBlank()) {
                throw new IllegalArgumentException("reportSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("reportFile", reportFile.toString());
            map.put("reportSha256", reportSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReportInspection(
            Path reportFile,
            String reportSha256,
            Map<String, Object> report) {
        public VerificationReportInspection {
            reportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (reportSha256 == null || reportSha256.isBlank()) {
                throw new IllegalArgumentException("reportSha256 must not be blank");
            }
            report = immutableMap(Objects.requireNonNull(report, "report must not be null"));
        }

        public String format() {
            return stringValue(report, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(report.get("passed"));
        }

        public Path packageDirectory() {
            return objectValue(report, "inspection")
                    .flatMap(inspection -> pathValue(inspection, "directory"))
                    .orElse(null);
        }

        public Path manifestFile() {
            return objectValue(report, "manifestVerification")
                    .flatMap(manifest -> objectValue(manifest, "inspection"))
                    .flatMap(inspection -> pathValue(inspection, "manifestFile"))
                    .orElse(null);
        }

        public String manifestSha256() {
            return objectValue(report, "manifestVerification")
                    .flatMap(manifest -> stringValue(manifest, "actualManifestSha256"))
                    .orElse(null);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("reportFile", reportFile.toString());
            map.put("reportSha256", reportSha256);
            map.put("format", format());
            map.put("passed", passed());
            if (packageDirectory() != null) {
                map.put("packageDirectory", packageDirectory().toString());
            }
            if (manifestFile() != null) {
                map.put("manifestFile", manifestFile().toString());
            }
            if (manifestSha256() != null) {
                map.put("manifestSha256", manifestSha256());
            }
            map.put("report", report);
            return Map.copyOf(map);
        }
    }

    public record VerificationReportVerification(
            VerificationReportInspection inspection,
            String expectedReportSha256,
            boolean reportSha256Matches,
            boolean schemaValid,
            boolean packageRevalidated,
            PackageVerification packageVerification,
            List<String> failures) {
        public VerificationReportVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReportSha256 = expectedReportSha256 == null || expectedReportSha256.isBlank()
                    ? null
                    : expectedReportSha256.trim();
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
                return "Promotion gate package verification report verified for " + inspection.reportFile() + ".";
            }
            return "Promotion gate package verification report failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("reportSha256Matches", reportSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("packageRevalidated", packageRevalidated);
            map.put("actualReportSha256", inspection.reportSha256());
            if (expectedReportSha256 != null) {
                map.put("expectedReportSha256", expectedReportSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (packageVerification != null) {
                map.put("packageVerification", packageVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerificationMarkdownReport(
            Path markdownFile,
            String markdownSha256,
            PackageVerification verification) {
        public VerificationMarkdownReport {
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("markdownFile", markdownFile.toString());
            map.put("markdownSha256", markdownSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationJUnitXmlReport(
            Path junitXmlFile,
            String junitXmlSha256,
            PackageVerification verification) {
        public VerificationJUnitXmlReport {
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (junitXmlSha256 == null || junitXmlSha256.isBlank()) {
                throw new IllegalArgumentException("junitXmlSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("junitXmlSha256", junitXmlSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundle(
            Path directory,
            VerificationReport json,
            VerificationMarkdownReport markdown,
            VerificationJUnitXmlReport junitXml) {
        public VerificationReportBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            json = Objects.requireNonNull(json, "json must not be null");
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            if (!json.verification().equals(markdown.verification())
                    || !json.verification().equals(junitXml.verification())) {
                throw new IllegalArgumentException("verification reports must describe the same package verification");
            }
        }

        public PackageVerification verification() {
            return json.verification();
        }

        public boolean passed() {
            return json.passed() && markdown.passed() && junitXml.passed();
        }

        public void requirePassed() {
            verification().requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("passed", passed());
            map.put("json", json.toMap());
            map.put("markdown", markdown.toMap());
            map.put("junitXml", junitXml.toMap());
            map.put("verification", verification().toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundleInspection(
            Path directory,
            VerificationReportInspection json,
            Path markdownFile,
            String markdownSha256,
            Path junitXmlFile,
            String junitXmlSha256) {
        public VerificationReportBundleInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            json = Objects.requireNonNull(json, "json must not be null");
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (markdownSha256 == null || markdownSha256.isBlank()) {
                throw new IllegalArgumentException("markdownSha256 must not be blank");
            }
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (junitXmlSha256 == null || junitXmlSha256.isBlank()) {
                throw new IllegalArgumentException("junitXmlSha256 must not be blank");
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("json", json.toMap());
            map.put("markdownFile", markdownFile.toString());
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("junitXmlSha256", junitXmlSha256);
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundleVerification(
            VerificationReportBundleInspection inspection,
            String expectedJsonReportSha256,
            boolean jsonReportVerified,
            boolean markdownMatchesRendered,
            boolean junitXmlMatchesRendered,
            VerificationReportVerification jsonVerification,
            List<String> failures) {
        public VerificationReportBundleVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonReportSha256 = expectedJsonReportSha256 == null || expectedJsonReportSha256.isBlank()
                    ? null
                    : expectedJsonReportSha256.trim();
            jsonVerification = Objects.requireNonNull(jsonVerification, "jsonVerification must not be null");
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
                return "Promotion gate package verification report bundle verified for "
                        + inspection.directory() + ".";
            }
            return "Promotion gate package verification report bundle failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonReportVerified", jsonReportVerified);
            map.put("markdownMatchesRendered", markdownMatchesRendered);
            map.put("junitXmlMatchesRendered", junitXmlMatchesRendered);
            if (expectedJsonReportSha256 != null) {
                map.put("expectedJsonReportSha256", expectedJsonReportSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            map.put("jsonVerification", jsonVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundleReceipt(
            Path receiptFile,
            String receiptSha256,
            VerificationReportBundleVerification verification) {
        public VerificationReportBundleReceipt {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundleReceiptInspection(
            Path receiptFile,
            String receiptSha256,
            Map<String, Object> receipt) {
        public VerificationReportBundleReceiptInspection {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            receipt = immutableMap(Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path reportDirectory() {
            return pathValue(receipt, "reportDirectory").orElse(null);
        }

        public Path jsonReportFile() {
            return pathValue(receipt, "jsonReportFile").orElse(null);
        }

        public String jsonReportSha256() {
            return stringValue(receipt, "jsonReportSha256").orElse(null);
        }

        public Path markdownFile() {
            return pathValue(receipt, "markdownFile").orElse(null);
        }

        public String markdownSha256() {
            return stringValue(receipt, "markdownSha256").orElse(null);
        }

        public Path junitXmlFile() {
            return pathValue(receipt, "junitXmlFile").orElse(null);
        }

        public String junitXmlSha256() {
            return stringValue(receipt, "junitXmlSha256").orElse(null);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("format", format());
            map.put("passed", passed());
            if (reportDirectory() != null) {
                map.put("reportDirectory", reportDirectory().toString());
            }
            if (jsonReportFile() != null) {
                map.put("jsonReportFile", jsonReportFile().toString());
            }
            if (jsonReportSha256() != null) {
                map.put("jsonReportSha256", jsonReportSha256());
            }
            if (markdownFile() != null) {
                map.put("markdownFile", markdownFile().toString());
            }
            if (markdownSha256() != null) {
                map.put("markdownSha256", markdownSha256());
            }
            if (junitXmlFile() != null) {
                map.put("junitXmlFile", junitXmlFile().toString());
            }
            if (junitXmlSha256() != null) {
                map.put("junitXmlSha256", junitXmlSha256());
            }
            map.put("receipt", receipt);
            return Map.copyOf(map);
        }
    }

    public record VerificationReportBundleReceiptVerification(
            VerificationReportBundleReceiptInspection inspection,
            String expectedReceiptSha256,
            boolean receiptSha256Matches,
            boolean schemaValid,
            boolean reportBundleRevalidated,
            VerificationReportBundleVerification reportBundleVerification,
            List<String> failures) {
        public VerificationReportBundleReceiptVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReceiptSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                    ? null
                    : expectedReceiptSha256.trim();
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
                return "Promotion gate package verification report bundle receipt verified for "
                        + inspection.receiptFile() + ".";
            }
            return "Promotion gate package verification report bundle receipt failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("receiptSha256Matches", receiptSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("reportBundleRevalidated", reportBundleRevalidated);
            map.put("actualReceiptSha256", inspection.receiptSha256());
            if (expectedReceiptSha256 != null) {
                map.put("expectedReceiptSha256", expectedReceiptSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (reportBundleVerification != null) {
                map.put("reportBundleVerification", reportBundleVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerificationIndex(
            Path indexFile,
            String indexSha256) {
        public VerificationIndex {
            indexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (indexSha256 == null || indexSha256.isBlank()) {
                throw new IllegalArgumentException("indexSha256 must not be blank");
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("indexFile", indexFile.toString());
            map.put("indexSha256", indexSha256);
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexInspection(
            Path indexFile,
            String indexSha256,
            Map<String, Object> index) {
        public VerificationIndexInspection {
            indexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (indexSha256 == null || indexSha256.isBlank()) {
                throw new IllegalArgumentException("indexSha256 must not be blank");
            }
            index = immutableMap(Objects.requireNonNull(index, "index must not be null"));
        }

        public String format() {
            return stringValue(index, "format").orElse("UNKNOWN");
        }

        public Path packageDirectory() {
            return pathValue(index, "packageDirectory").orElse(null);
        }

        public Path reportDirectory() {
            return pathValue(index, "reportDirectory").orElse(null);
        }

        public boolean passed() {
            return Boolean.TRUE.equals(index.get("passed"));
        }

        public boolean promotable() {
            return Boolean.TRUE.equals(index.get("promotable"));
        }

        public String decisionStatus() {
            return stringValue(index, "decisionStatus").orElse("UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            return stringValue(index, "decisionCandidate");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("indexFile", indexFile.toString());
            map.put("indexSha256", indexSha256);
            map.put("format", format());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("decisionStatus", decisionStatus());
            decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
            if (packageDirectory() != null) {
                map.put("packageDirectory", packageDirectory().toString());
            }
            if (reportDirectory() != null) {
                map.put("reportDirectory", reportDirectory().toString());
            }
            map.put("index", index);
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexVerification(
            VerificationIndexInspection inspection,
            String expectedIndexSha256,
            boolean indexSha256Matches,
            boolean schemaValid,
            boolean referencedSha256Match,
            List<String> failures) {
        public VerificationIndexVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedIndexSha256 = expectedIndexSha256 == null || expectedIndexSha256.isBlank()
                    ? null
                    : expectedIndexSha256.trim();
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
                return "Promotion gate package verification index verified for " + inspection.indexFile() + ".";
            }
            return "Promotion gate package verification index failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("indexSha256Matches", indexSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("referencedSha256Match", referencedSha256Match);
            map.put("actualIndexSha256", inspection.indexSha256());
            if (expectedIndexSha256 != null) {
                map.put("expectedIndexSha256", expectedIndexSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexReceipt(
            Path receiptFile,
            String receiptSha256,
            VerificationIndexVerification verification) {
        public VerificationIndexReceipt {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexReceiptInspection(
            Path receiptFile,
            String receiptSha256,
            Map<String, Object> receipt) {
        public VerificationIndexReceiptInspection {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            receipt = immutableMap(Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path indexFile() {
            return pathValue(receipt, "indexFile").orElse(null);
        }

        public String indexSha256() {
            return stringValue(receipt, "indexSha256").orElse(null);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("format", format());
            map.put("passed", passed());
            if (indexFile() != null) {
                map.put("indexFile", indexFile().toString());
            }
            if (indexSha256() != null) {
                map.put("indexSha256", indexSha256());
            }
            map.put("receipt", receipt);
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexReceiptVerification(
            VerificationIndexReceiptInspection inspection,
            String expectedReceiptSha256,
            boolean receiptSha256Matches,
            boolean schemaValid,
            boolean indexRevalidated,
            VerificationIndexVerification indexVerification,
            List<String> failures) {
        public VerificationIndexReceiptVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReceiptSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                    ? null
                    : expectedReceiptSha256.trim();
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
                return "Promotion gate package verification index receipt verified for "
                        + inspection.receiptFile() + ".";
            }
            return "Promotion gate package verification index receipt failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("receiptSha256Matches", receiptSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("indexRevalidated", indexRevalidated);
            map.put("actualReceiptSha256", inspection.receiptSha256());
            if (expectedReceiptSha256 != null) {
                map.put("expectedReceiptSha256", expectedReceiptSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (indexVerification != null) {
                map.put("indexVerification", indexVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexPackageAudit(
            VerificationIndexVerification indexVerification,
            PackageVerification packageVerification,
            List<String> failures) {
        public VerificationIndexPackageAudit {
            indexVerification = Objects.requireNonNull(indexVerification, "indexVerification must not be null");
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
                return "Promotion gate package verification index package audit passed for "
                        + indexVerification.inspection().indexFile() + ".";
            }
            return "Promotion gate package verification index package audit failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("failures", failures);
            map.put("indexVerification", indexVerification.toMap());
            if (packageVerification != null) {
                map.put("packageVerification", packageVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexPackageAuditReport(
            Path reportFile,
            String reportSha256,
            VerificationIndexPackageAudit audit) {
        public VerificationIndexPackageAuditReport {
            reportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (reportSha256 == null || reportSha256.isBlank()) {
                throw new IllegalArgumentException("reportSha256 must not be blank");
            }
            audit = Objects.requireNonNull(audit, "audit must not be null");
        }

        public boolean passed() {
            return audit.passed();
        }

        public void requirePassed() {
            audit.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("reportFile", reportFile.toString());
            map.put("reportSha256", reportSha256);
            map.put("passed", passed());
            map.put("audit", audit.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexPackageAuditReportInspection(
            Path reportFile,
            String reportSha256,
            Map<String, Object> report) {
        public VerificationIndexPackageAuditReportInspection {
            reportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (reportSha256 == null || reportSha256.isBlank()) {
                throw new IllegalArgumentException("reportSha256 must not be blank");
            }
            report = immutableMap(Objects.requireNonNull(report, "report must not be null"));
        }

        public String format() {
            return stringValue(report, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(report.get("passed"));
        }

        public Path indexFile() {
            return pathValue(report, "indexFile").orElse(null);
        }

        public String indexSha256() {
            return stringValue(report, "indexSha256").orElse(null);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("reportFile", reportFile.toString());
            map.put("reportSha256", reportSha256);
            map.put("format", format());
            map.put("passed", passed());
            if (indexFile() != null) {
                map.put("indexFile", indexFile().toString());
            }
            if (indexSha256() != null) {
                map.put("indexSha256", indexSha256());
            }
            map.put("report", report);
            return Map.copyOf(map);
        }
    }

    public record VerificationIndexPackageAuditReportVerification(
            VerificationIndexPackageAuditReportInspection inspection,
            String expectedReportSha256,
            boolean reportSha256Matches,
            boolean schemaValid,
            boolean auditRevalidated,
            VerificationIndexPackageAudit audit,
            List<String> failures) {
        public VerificationIndexPackageAuditReportVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReportSha256 = expectedReportSha256 == null || expectedReportSha256.isBlank()
                    ? null
                    : expectedReportSha256.trim();
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
                return "Promotion gate verification index package audit report verified for "
                        + inspection.reportFile() + ".";
            }
            return "Promotion gate verification index package audit report failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("reportSha256Matches", reportSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("auditRevalidated", auditRevalidated);
            map.put("actualReportSha256", inspection.reportSha256());
            if (expectedReportSha256 != null) {
                map.put("expectedReportSha256", expectedReportSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (audit != null) {
                map.put("audit", audit.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceManifest(
            Path evidenceFile,
            String evidenceSha256) {
        public VerificationEvidenceManifest {
            evidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (evidenceSha256 == null || evidenceSha256.isBlank()) {
                throw new IllegalArgumentException("evidenceSha256 must not be blank");
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("evidenceFile", evidenceFile.toString());
            map.put("evidenceSha256", evidenceSha256);
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceInspection(
            Path evidenceFile,
            String evidenceSha256,
            Map<String, Object> evidence) {
        public VerificationEvidenceInspection {
            evidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (evidenceSha256 == null || evidenceSha256.isBlank()) {
                throw new IllegalArgumentException("evidenceSha256 must not be blank");
            }
            evidence = immutableMap(Objects.requireNonNull(evidence, "evidence must not be null"));
        }

        public String format() {
            return stringValue(evidence, "format").orElse("UNKNOWN");
        }

        public Path packageDirectory() {
            return pathValue(evidence, "packageDirectory").orElse(null);
        }

        public Path reportDirectory() {
            return pathValue(evidence, "reportDirectory").orElse(null);
        }

        public boolean passed() {
            return Boolean.TRUE.equals(evidence.get("passed"));
        }

        public boolean promotable() {
            return Boolean.TRUE.equals(evidence.get("promotable"));
        }

        public String decisionStatus() {
            return stringValue(evidence, "decisionStatus").orElse("UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            return stringValue(evidence, "decisionCandidate");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("evidenceFile", evidenceFile.toString());
            map.put("evidenceSha256", evidenceSha256);
            map.put("format", format());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("decisionStatus", decisionStatus());
            decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
            if (packageDirectory() != null) {
                map.put("packageDirectory", packageDirectory().toString());
            }
            if (reportDirectory() != null) {
                map.put("reportDirectory", reportDirectory().toString());
            }
            map.put("evidence", evidence);
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceVerification(
            VerificationEvidenceInspection inspection,
            String expectedEvidenceSha256,
            boolean evidenceSha256Matches,
            boolean schemaValid,
            boolean evidenceFilesSha256Match,
            boolean packageArtifactsSha256Match,
            List<String> failures) {
        public VerificationEvidenceVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedEvidenceSha256 = expectedEvidenceSha256 == null || expectedEvidenceSha256.isBlank()
                    ? null
                    : expectedEvidenceSha256.trim();
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
                return "Promotion gate package verification evidence verified for "
                        + inspection.evidenceFile() + ".";
            }
            return "Promotion gate package verification evidence failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("evidenceSha256Matches", evidenceSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("evidenceFilesSha256Match", evidenceFilesSha256Match);
            map.put("packageArtifactsSha256Match", packageArtifactsSha256Match);
            map.put("actualEvidenceSha256", inspection.evidenceSha256());
            if (expectedEvidenceSha256 != null) {
                map.put("expectedEvidenceSha256", expectedEvidenceSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceReceipt(
            Path receiptFile,
            String receiptSha256,
            VerificationEvidenceVerification verification) {
        public VerificationEvidenceReceipt {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("passed", passed());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceReceiptInspection(
            Path receiptFile,
            String receiptSha256,
            Map<String, Object> receipt) {
        public VerificationEvidenceReceiptInspection {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (receiptSha256 == null || receiptSha256.isBlank()) {
                throw new IllegalArgumentException("receiptSha256 must not be blank");
            }
            receipt = immutableMap(Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path evidenceFile() {
            return pathValue(receipt, "evidenceFile").orElse(null);
        }

        public String evidenceSha256() {
            return stringValue(receipt, "evidenceSha256").orElse(null);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("format", format());
            map.put("passed", passed());
            if (evidenceFile() != null) {
                map.put("evidenceFile", evidenceFile().toString());
            }
            if (evidenceSha256() != null) {
                map.put("evidenceSha256", evidenceSha256());
            }
            map.put("receipt", receipt);
            return Map.copyOf(map);
        }
    }

    public record VerificationEvidenceReceiptVerification(
            VerificationEvidenceReceiptInspection inspection,
            String expectedReceiptSha256,
            boolean receiptSha256Matches,
            boolean schemaValid,
            boolean evidenceRevalidated,
            VerificationEvidenceVerification evidenceVerification,
            List<String> failures) {
        public VerificationEvidenceReceiptVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReceiptSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                    ? null
                    : expectedReceiptSha256.trim();
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
                return "Promotion gate package verification evidence receipt verified for "
                        + inspection.receiptFile() + ".";
            }
            return "Promotion gate package verification evidence receipt failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("receiptSha256Matches", receiptSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("evidenceRevalidated", evidenceRevalidated);
            map.put("actualReceiptSha256", inspection.receiptSha256());
            if (expectedReceiptSha256 != null) {
                map.put("expectedReceiptSha256", expectedReceiptSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (evidenceVerification != null) {
                map.put("evidenceVerification", evidenceVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public record VerifiedPackageBundle(
            PackageBundle packageBundle,
            VerificationReportBundle reports,
            VerificationReportBundleReceipt reportBundleReceipt,
            VerificationIndex index,
            VerificationIndexReceipt receipt,
            VerificationIndexPackageAuditReport packageAuditReport,
            VerificationEvidenceManifest evidence,
            VerificationEvidenceReceipt evidenceReceipt) {
        public VerifiedPackageBundle(PackageBundle packageBundle, VerificationReportBundle reports) {
            this(packageBundle, reports, null, null, null, null, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationReportBundleReceipt reportBundleReceipt) {
            this(packageBundle, reports, reportBundleReceipt, null, null, null, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationIndex index) {
            this(packageBundle, reports, null, index, null, null, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationIndex index,
                VerificationIndexReceipt receipt) {
            this(packageBundle, reports, null, index, receipt, null, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationIndex index,
                VerificationIndexReceipt receipt,
                VerificationIndexPackageAuditReport packageAuditReport) {
            this(packageBundle, reports, null, index, receipt, packageAuditReport, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationReportBundleReceipt reportBundleReceipt,
                VerificationIndex index,
                VerificationIndexReceipt receipt,
                VerificationIndexPackageAuditReport packageAuditReport) {
            this(packageBundle, reports, reportBundleReceipt, index, receipt, packageAuditReport, null, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationIndex index,
                VerificationIndexReceipt receipt,
                VerificationIndexPackageAuditReport packageAuditReport,
                VerificationEvidenceManifest evidence) {
            this(packageBundle, reports, null, index, receipt, packageAuditReport, evidence, null);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationIndex index,
                VerificationIndexReceipt receipt,
                VerificationIndexPackageAuditReport packageAuditReport,
                VerificationEvidenceManifest evidence,
                VerificationEvidenceReceipt evidenceReceipt) {
            this(packageBundle, reports, null, index, receipt, packageAuditReport, evidence, evidenceReceipt);
        }

        public VerifiedPackageBundle(
                PackageBundle packageBundle,
                VerificationReportBundle reports,
                VerificationReportBundleReceipt reportBundleReceipt,
                VerificationIndex index,
                VerificationIndexReceipt receipt,
                VerificationIndexPackageAuditReport packageAuditReport,
                VerificationEvidenceManifest evidence) {
            this(packageBundle, reports, reportBundleReceipt, index, receipt, packageAuditReport, evidence, null);
        }

        public VerifiedPackageBundle {
            packageBundle = Objects.requireNonNull(packageBundle, "packageBundle must not be null");
            reports = Objects.requireNonNull(reports, "reports must not be null");
            if (!packageBundle.directory().equals(reports.verification().inspection().directory())) {
                throw new IllegalArgumentException("reports must verify the package bundle directory");
            }
            if (reportBundleReceipt != null
                    && !reportBundleReceipt.verification().inspection().directory().equals(reports.directory())) {
                throw new IllegalArgumentException("reportBundleReceipt must verify the report bundle directory");
            }
            if (evidenceReceipt != null && evidence == null) {
                throw new IllegalArgumentException("evidenceReceipt requires evidence");
            }
        }

        public Path directory() {
            return packageBundle.directory();
        }

        public PackageVerification verification() {
            return reports.verification();
        }

        public boolean passed() {
            return packageBundle.passed()
                    && reports.passed()
                    && (reportBundleReceipt == null || reportBundleReceipt.passed())
                    && (receipt == null || receipt.passed())
                    && (packageAuditReport == null || packageAuditReport.passed())
                    && (evidenceReceipt == null || evidenceReceipt.passed());
        }

        public boolean promotable() {
            return packageBundle.promotable();
        }

        public void requirePassed() {
            packageBundle.requirePassed();
            reports.requirePassed();
            if (reportBundleReceipt != null) {
                reportBundleReceipt.requirePassed();
            }
            if (receipt != null) {
                receipt.requirePassed();
            }
            if (packageAuditReport != null) {
                packageAuditReport.requirePassed();
            }
            if (evidenceReceipt != null) {
                evidenceReceipt.requirePassed();
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory().toString());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("package", packageBundle.toMap());
            map.put("reports", reports.toMap());
            if (reportBundleReceipt != null) {
                map.put("reportBundleReceipt", reportBundleReceipt.toMap());
            }
            if (index != null) {
                map.put("index", index.toMap());
            }
            if (receipt != null) {
                map.put("receipt", receipt.toMap());
            }
            if (packageAuditReport != null) {
                map.put("packageAuditReport", packageAuditReport.toMap());
            }
            if (evidence != null) {
                map.put("evidence", evidence.toMap());
            }
            if (evidenceReceipt != null) {
                map.put("evidenceReceipt", evidenceReceipt.toMap());
            }
            map.put("verification", verification().toMap());
            return Map.copyOf(map);
        }
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result) throws IOException {
        return write(directory, result, Options.defaults(), Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Instant generatedAt) throws IOException {
        return write(directory, result, Options.defaults(), generatedAt);
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options) throws IOException {
        return write(directory, result, options, Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPromotionGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");

        TrainingReportPromotionArtifacts.ArtifactBundle review = TrainingReportPromotionArtifacts.write(
                resolvedDirectory,
                resolvedResult.review(),
                resolvedOptions.review());
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts =
                TrainingReportPromotionGateArtifacts.write(
                        resolvedDirectory,
                        resolvedResult,
                        resolvedOptions.artifacts());
        Map<String, Path> packageArtifacts = packageArtifactPaths(
                review,
                snapshotSourceReports(resolvedDirectory, review));
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                TrainingReportPromotionGateArtifactManifest.write(
                        artifacts,
                        packageArtifacts,
                        resolvedOptions.manifest(),
                        resolvedGeneratedAt);
        return new PackageBundle(resolvedDirectory, review, artifacts, manifest);
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result) throws IOException {
        return writeAndVerify(directory, result, Options.defaults(), Instant.now(), directory);
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Instant generatedAt) throws IOException {
        return writeAndVerify(directory, result, Options.defaults(), generatedAt, directory);
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options) throws IOException {
        return writeAndVerify(directory, result, options, Instant.now(), directory);
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options,
            Instant generatedAt) throws IOException {
        return writeAndVerify(directory, result, options, generatedAt, directory);
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = write(directory, result, options, generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    public static PackageBundle run(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory) throws IOException {
        return run(reportFiles, baselineName, policy, outputDirectory, Options.defaults(), Instant.now());
    }

    public static PackageBundle run(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedOutputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        TrainingReportPromotionGate.Request request = new TrainingReportPromotionGate.Request(
                reportFiles,
                baselineName,
                policy,
                resolvedOutputDirectory,
                resolvedOptions.review());
        TrainingReportPromotionGate.Result result = TrainingReportPromotionGate.evaluate(request);
        return write(resolvedOutputDirectory, result, resolvedOptions, resolvedGeneratedAt);
    }

    public static VerifiedPackageBundle runAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory) throws IOException {
        return runAndVerify(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                Options.defaults(),
                Instant.now(),
                outputDirectory);
    }

    public static VerifiedPackageBundle runAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        return runAndVerify(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                options,
                generatedAt,
                outputDirectory);
    }

    public static VerifiedPackageBundle runAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = run(reportFiles, baselineName, policy, outputDirectory, options, generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    public static PackageBundle runWithSeverity(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory) throws IOException {
        return runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                Options.defaults(),
                Instant.now());
    }

    public static PackageBundle runWithSeverity(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedOutputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportDiagnostics.Severity resolvedSeverity = Objects.requireNonNull(
                maxAllowedDiagnosticSeverity,
                "maxAllowedDiagnosticSeverity must not be null");
        TrainingReportPortfolio.PromotionPolicy policy = TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                .withMaxCandidateDiagnosticSeverity(resolvedSeverity);
        return run(
                reportFiles,
                baselineName,
                policy,
                resolvedOutputDirectory,
                options,
                generatedAt);
    }

    public static VerifiedPackageBundle runWithSeverityAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory) throws IOException {
        return runWithSeverityAndVerify(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                Options.defaults(),
                Instant.now(),
                outputDirectory);
    }

    public static VerifiedPackageBundle runWithSeverityAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        return runWithSeverityAndVerify(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt,
                outputDirectory);
    }

    public static VerifiedPackageBundle runWithSeverityAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    public static PackageInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static PackageInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest =
                TrainingReportPromotionGateArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        return readFromManifest(manifest, resolvedOptions);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(
            PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return TrainingReportPromotionGateArtifactManifest.verify(bundle.manifest());
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(Path directory)
            throws IOException {
        return verify(directory, null, Options.defaults());
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verify(directory, expectedManifestSha256, Options.defaults());
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return TrainingReportPromotionGateArtifactManifest.verify(
                resolvedDirectory,
                expectedManifestSha256,
                resolvedOptions.manifest());
    }

    public static PackageVerification verifyComplete(PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification = verify(bundle);
        PackageInspection inspection = readFromManifest(manifestVerification.inspection(), Options.defaults());
        SourceSnapshotVerification sourceSnapshotVerification = verifySourceReportSnapshots(inspection);
        return packageVerification(inspection, manifestVerification, sourceSnapshotVerification);
    }

    public static VerifiedPackageBundle verifyPackage(PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verifyPackage(bundle, bundle.directory());
    }

    public static VerifiedPackageBundle verifyPackage(
            PackageBundle bundle,
            Path reportDirectory) throws IOException {
        PackageBundle resolvedBundle = Objects.requireNonNull(bundle, "bundle must not be null");
        PackageVerification verification = verifyComplete(resolvedBundle);
        VerificationReportBundle reports = writeVerificationReports(reportDirectory, verification);
        VerificationReportBundleReceipt reportBundleReceipt = verifyVerificationReportBundleAndWriteReceipt(
                reports.directory(),
                reports.json().reportSha256(),
                defaultVerificationReportBundleReceiptFile(reports.directory()));
        VerifiedPackageBundle verifiedPackage = new VerifiedPackageBundle(
                resolvedBundle,
                reports,
                reportBundleReceipt);
        VerificationIndex index = writeVerificationIndex(
                defaultVerificationIndexFile(reports.directory()),
                verifiedPackage);
        VerificationIndexReceipt receipt = verifyVerificationIndexAndWriteReceipt(
                index.indexFile(),
                index.indexSha256(),
                defaultVerificationIndexReceiptFile(reports.directory()));
        VerificationIndexPackageAuditReport packageAuditReport = auditVerificationIndexPackageAndWriteReport(
                index.indexFile(),
                index.indexSha256(),
                defaultVerificationIndexPackageAuditFile(reports.directory()));
        VerifiedPackageBundle verifiedPackageWithAudit =
                new VerifiedPackageBundle(resolvedBundle, reports, reportBundleReceipt, index, receipt,
                        packageAuditReport);
        VerificationEvidenceManifest evidence = writeVerificationEvidenceManifest(
                defaultVerificationEvidenceFile(reports.directory()),
                verifiedPackageWithAudit);
        VerificationEvidenceReceipt evidenceReceipt = verifyVerificationEvidenceManifestAndWriteReceipt(
                evidence.evidenceFile(),
                evidence.evidenceSha256(),
                defaultVerificationEvidenceReceiptFile(reports.directory()));
        return new VerifiedPackageBundle(
                resolvedBundle,
                reports,
                reportBundleReceipt,
                index,
                receipt,
                packageAuditReport,
                evidence,
                evidenceReceipt);
    }

    public static PackageRefresh refreshComplete(Path directory) throws IOException {
        return refreshComplete(directory, Options.defaults(), Instant.now(), directory);
    }

    public static PackageRefresh refreshComplete(
            Path directory,
            Options options) throws IOException {
        return refreshComplete(directory, options, Instant.now(), directory);
    }

    public static PackageRefresh refreshComplete(
            Path directory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Path resolvedReportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory must not be null")
                .toAbsolutePath()
                .normalize();

        TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest =
                TrainingReportPromotionGateArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        verifyNonRefreshableManifestArtifacts(
                manifest,
                Set.of(GATE_MARKDOWN_ARTIFACT, GATE_JUNIT_XML_ARTIFACT));
        TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts =
                TrainingReportPromotionGateArtifacts.refreshDerivedFiles(
                        requiredManifestArtifact(manifest, GATE_JSON_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, GATE_MARKDOWN_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, GATE_JUNIT_XML_ARTIFACT).file());
        TrainingReportPromotionGateArtifactManifest.ManifestInspection refreshedManifest =
                TrainingReportPromotionGateArtifactManifest.refresh(manifest, resolvedGeneratedAt);
        PackageVerification verification = verifyComplete(resolvedDirectory, null, resolvedOptions);
        VerificationReportBundle reports = writeVerificationReports(resolvedReportDirectory, verification);
        VerificationReportBundleVerification reportVerification =
                verifyVerificationReportBundle(resolvedReportDirectory, reports.json().reportSha256());
        return new PackageRefresh(
                resolvedDirectory,
                refreshedManifest,
                artifacts,
                verification,
                reports,
                reportVerification);
    }

    public static PackageVerification verifyComplete(Path directory) throws IOException {
        return verifyComplete(directory, null, Options.defaults());
    }

    public static PackageVerification verifyComplete(Path directory, String expectedManifestSha256)
            throws IOException {
        return verifyComplete(directory, expectedManifestSha256, Options.defaults());
    }

    public static PackageVerification verifyComplete(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification =
                TrainingReportPromotionGateArtifactManifest.verify(
                        resolvedDirectory,
                        expectedManifestSha256,
                        resolvedOptions.manifest());
        PackageInspection inspection = readFromManifest(manifestVerification.inspection(), resolvedOptions);
        SourceSnapshotVerification sourceSnapshotVerification = verifySourceReportSnapshots(inspection);
        return packageVerification(inspection, manifestVerification, sourceSnapshotVerification);
    }

    public static VerificationReport verifyCompleteAndWriteReport(Path directory) throws IOException {
        return verifyCompleteAndWriteReport(
                directory,
                null,
                Options.defaults(),
                defaultVerificationReportFile(directory));
    }

    public static VerificationReport verifyCompleteAndWriteReport(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verifyCompleteAndWriteReport(
                directory,
                expectedManifestSha256,
                Options.defaults(),
                defaultVerificationReportFile(directory));
    }

    public static VerificationReport verifyCompleteAndWriteReport(
            Path directory,
            Path reportFile) throws IOException {
        return verifyCompleteAndWriteReport(directory, null, Options.defaults(), reportFile);
    }

    public static VerificationReport verifyCompleteAndWriteReport(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        return verifyCompleteAndWriteReport(
                directory,
                expectedManifestSha256,
                options,
                defaultVerificationReportFile(directory));
    }

    public static VerificationReport verifyCompleteAndWriteReport(
            Path directory,
            String expectedManifestSha256,
            Options options,
            Path reportFile) throws IOException {
        PackageVerification verification = verifyComplete(directory, expectedManifestSha256, options);
        return writeVerificationReport(reportFile, verification);
    }

    public static VerificationMarkdownReport verifyCompleteAndWriteMarkdownReport(Path directory) throws IOException {
        return verifyCompleteAndWriteMarkdownReport(
                directory,
                null,
                Options.defaults(),
                defaultVerificationMarkdownFile(directory));
    }

    public static VerificationMarkdownReport verifyCompleteAndWriteMarkdownReport(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verifyCompleteAndWriteMarkdownReport(
                directory,
                expectedManifestSha256,
                Options.defaults(),
                defaultVerificationMarkdownFile(directory));
    }

    public static VerificationMarkdownReport verifyCompleteAndWriteMarkdownReport(
            Path directory,
            Path markdownFile) throws IOException {
        return verifyCompleteAndWriteMarkdownReport(directory, null, Options.defaults(), markdownFile);
    }

    public static VerificationMarkdownReport verifyCompleteAndWriteMarkdownReport(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        return verifyCompleteAndWriteMarkdownReport(
                directory,
                expectedManifestSha256,
                options,
                defaultVerificationMarkdownFile(directory));
    }

    public static VerificationMarkdownReport verifyCompleteAndWriteMarkdownReport(
            Path directory,
            String expectedManifestSha256,
            Options options,
            Path markdownFile) throws IOException {
        PackageVerification verification = verifyComplete(directory, expectedManifestSha256, options);
        return writeVerificationMarkdownReport(markdownFile, verification);
    }

    public static VerificationJUnitXmlReport verifyCompleteAndWriteJUnitXmlReport(Path directory) throws IOException {
        return verifyCompleteAndWriteJUnitXmlReport(
                directory,
                null,
                Options.defaults(),
                defaultVerificationJunitXmlFile(directory));
    }

    public static VerificationJUnitXmlReport verifyCompleteAndWriteJUnitXmlReport(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verifyCompleteAndWriteJUnitXmlReport(
                directory,
                expectedManifestSha256,
                Options.defaults(),
                defaultVerificationJunitXmlFile(directory));
    }

    public static VerificationJUnitXmlReport verifyCompleteAndWriteJUnitXmlReport(
            Path directory,
            Path junitXmlFile) throws IOException {
        return verifyCompleteAndWriteJUnitXmlReport(directory, null, Options.defaults(), junitXmlFile);
    }

    public static VerificationJUnitXmlReport verifyCompleteAndWriteJUnitXmlReport(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        return verifyCompleteAndWriteJUnitXmlReport(
                directory,
                expectedManifestSha256,
                options,
                defaultVerificationJunitXmlFile(directory));
    }

    public static VerificationJUnitXmlReport verifyCompleteAndWriteJUnitXmlReport(
            Path directory,
            String expectedManifestSha256,
            Options options,
            Path junitXmlFile) throws IOException {
        PackageVerification verification = verifyComplete(directory, expectedManifestSha256, options);
        return writeVerificationJUnitXmlReport(junitXmlFile, verification);
    }

    public static VerificationReportBundle verifyCompleteAndWriteReports(Path directory) throws IOException {
        return verifyCompleteAndWriteReports(directory, null, Options.defaults(), directory);
    }

    public static VerificationReportBundle verifyCompleteAndWriteReports(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verifyCompleteAndWriteReports(directory, expectedManifestSha256, Options.defaults(), directory);
    }

    public static VerificationReportBundle verifyCompleteAndWriteReports(
            Path directory,
            Path reportDirectory) throws IOException {
        return verifyCompleteAndWriteReports(directory, null, Options.defaults(), reportDirectory);
    }

    public static VerificationReportBundle verifyCompleteAndWriteReports(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        return verifyCompleteAndWriteReports(directory, expectedManifestSha256, options, directory);
    }

    public static VerificationReportBundle verifyCompleteAndWriteReports(
            Path directory,
            String expectedManifestSha256,
            Options options,
            Path reportDirectory) throws IOException {
        PackageVerification verification = verifyComplete(directory, expectedManifestSha256, options);
        return writeVerificationReports(reportDirectory, verification);
    }

    public static VerificationReport writeVerificationReport(
            Path reportFile,
            PackageVerification verification) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReportFile,
                TrainerJson.toJson(verificationReportMap(resolvedVerification, Instant.now())) + "\n");
        return new VerificationReport(
                resolvedReportFile,
                TrainerCheckpointIO.sha256Hex(resolvedReportFile),
                resolvedVerification);
    }

    public static VerificationReportInspection readVerificationReport(Path reportFile) throws IOException {
        return TrainingReportPromotionGateVerificationReportVerifier.read(reportFile);
    }

    public static VerificationReportVerification verifyVerificationReport(Path reportFile) throws IOException {
        return verifyVerificationReport(reportFile, null);
    }

    public static VerificationReportVerification verifyVerificationReport(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        return TrainingReportPromotionGateVerificationReportVerifier.verify(reportFile, expectedReportSha256);
    }

    public static VerificationMarkdownReport writeVerificationMarkdownReport(
            Path markdownFile,
            PackageVerification verification) throws IOException {
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedMarkdownFile,
                renderVerificationMarkdown(resolvedVerification));
        return new VerificationMarkdownReport(
                resolvedMarkdownFile,
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                resolvedVerification);
    }

    public static VerificationJUnitXmlReport writeVerificationJUnitXmlReport(
            Path junitXmlFile,
            PackageVerification verification) throws IOException {
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedJunitXmlFile,
                renderVerificationJUnitXml(resolvedVerification));
        return new VerificationJUnitXmlReport(
                resolvedJunitXmlFile,
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile),
                resolvedVerification);
    }

    public static VerificationReportBundle writeVerificationReports(
            Path reportDirectory,
            PackageVerification verification) throws IOException {
        Path resolvedReportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        return new VerificationReportBundle(
                resolvedReportDirectory,
                writeVerificationReport(defaultVerificationReportFile(resolvedReportDirectory), resolvedVerification),
                writeVerificationMarkdownReport(
                        defaultVerificationMarkdownFile(resolvedReportDirectory),
                        resolvedVerification),
                writeVerificationJUnitXmlReport(
                        defaultVerificationJunitXmlFile(resolvedReportDirectory),
                        resolvedVerification));
    }

    public static VerificationReportBundleInspection readVerificationReportBundle(Path reportDirectory)
            throws IOException {
        return TrainingReportPromotionGateVerificationReportBundleVerifier.read(reportDirectory);
    }

    public static VerificationReportBundleVerification verifyVerificationReportBundle(Path reportDirectory)
            throws IOException {
        return verifyVerificationReportBundle(reportDirectory, null);
    }

    public static VerificationReportBundleVerification verifyVerificationReportBundle(
            Path reportDirectory,
            String expectedJsonReportSha256) throws IOException {
        return TrainingReportPromotionGateVerificationReportBundleVerifier.verify(
                reportDirectory,
                expectedJsonReportSha256);
    }

    public static VerificationReportBundleReceipt writeVerificationReportBundleReceipt(
            Path receiptFile,
            VerificationReportBundleVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationReportBundleVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(verificationReportBundleReceiptMap(resolvedVerification, Instant.now())) + "\n");
        return new VerificationReportBundleReceipt(
                resolvedReceiptFile,
                TrainerCheckpointIO.sha256Hex(resolvedReceiptFile),
                resolvedVerification);
    }

    public static VerificationReportBundleReceipt verifyVerificationReportBundleAndWriteReceipt(
            Path reportDirectory,
            Path receiptFile) throws IOException {
        return verifyVerificationReportBundleAndWriteReceipt(reportDirectory, null, receiptFile);
    }

    public static VerificationReportBundleReceipt verifyVerificationReportBundleAndWriteReceipt(
            Path reportDirectory,
            String expectedJsonReportSha256,
            Path receiptFile) throws IOException {
        VerificationReportBundleVerification verification =
                verifyVerificationReportBundle(reportDirectory, expectedJsonReportSha256);
        return writeVerificationReportBundleReceipt(receiptFile, verification);
    }

    public static VerificationReportBundleReceiptInspection readVerificationReportBundleReceipt(Path receiptFile)
            throws IOException {
        return TrainingReportPromotionGateVerificationReportBundleReceiptVerifier.read(receiptFile);
    }

    public static VerificationReportBundleReceiptVerification verifyVerificationReportBundleReceipt(Path receiptFile)
            throws IOException {
        return verifyVerificationReportBundleReceipt(receiptFile, null);
    }

    public static VerificationReportBundleReceiptVerification verifyVerificationReportBundleReceipt(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateVerificationReportBundleReceiptVerifier.verify(
                receiptFile,
                expectedReceiptSha256);
    }

    public static VerificationIndex writeVerificationIndex(
            Path indexFile,
            VerifiedPackageBundle verifiedPackage) throws IOException {
        Path resolvedIndexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedIndexFile,
                TrainerJson.toJson(verificationIndexMap(resolvedVerifiedPackage, Instant.now())) + "\n");
        return new VerificationIndex(
                resolvedIndexFile,
                TrainerCheckpointIO.sha256Hex(resolvedIndexFile));
    }

    public static VerificationIndexInspection readVerificationIndex(Path indexFile) throws IOException {
        return TrainingReportPromotionGateVerificationIndexVerifier.read(indexFile);
    }

    public static VerificationIndexVerification verifyVerificationIndex(Path indexFile) throws IOException {
        return verifyVerificationIndex(indexFile, null);
    }

    public static VerificationIndexVerification verifyVerificationIndex(
            Path indexFile,
            String expectedIndexSha256) throws IOException {
        return TrainingReportPromotionGateVerificationIndexVerifier.verify(indexFile, expectedIndexSha256);
    }

    public static VerificationIndexPackageAudit auditVerificationIndexPackage(Path indexFile)
            throws IOException {
        return auditVerificationIndexPackage(indexFile, null);
    }

    public static VerificationIndexPackageAudit auditVerificationIndexPackage(
            Path indexFile,
            String expectedIndexSha256) throws IOException {
        return auditVerificationIndexPackage(verifyVerificationIndex(indexFile, expectedIndexSha256));
    }

    public static VerificationIndexPackageAudit auditVerificationIndexPackage(
            VerificationIndexVerification indexVerification) throws IOException {
        VerificationIndexVerification resolvedIndexVerification = Objects.requireNonNull(
                indexVerification,
                "indexVerification must not be null");
        List<String> failures = new ArrayList<>(resolvedIndexVerification.failures());
        PackageVerification packageVerification = null;
        Path packageDirectory = resolvedIndexVerification.inspection().packageDirectory();
        if (packageDirectory == null) {
            failures.add("Verification index package audit cannot resolve packageDirectory from "
                    + resolvedIndexVerification.inspection().indexFile());
        } else {
            try {
                packageVerification = verifyComplete(
                        packageDirectory,
                        indexedManifestSha256(resolvedIndexVerification.inspection()).orElse(null),
                        Options.defaults());
                failures.addAll(packageVerification.failures());
            } catch (IOException error) {
                failures.add("Complete package verification failed for "
                        + packageDirectory + ": " + error.getMessage());
            }
        }
        return new VerificationIndexPackageAudit(
                resolvedIndexVerification,
                packageVerification,
                failures);
    }

    public static VerificationIndexPackageAuditReport writeVerificationIndexPackageAuditReport(
            Path reportFile,
            VerificationIndexPackageAudit audit) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationIndexPackageAudit resolvedAudit = Objects.requireNonNull(audit, "audit must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReportFile,
                TrainerJson.toJson(verificationIndexPackageAuditReportMap(resolvedAudit, Instant.now())) + "\n");
        return new VerificationIndexPackageAuditReport(
                resolvedReportFile,
                TrainerCheckpointIO.sha256Hex(resolvedReportFile),
                resolvedAudit);
    }

    public static VerificationIndexPackageAuditReportInspection readVerificationIndexPackageAuditReport(
            Path reportFile) throws IOException {
        return TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier.read(reportFile);
    }

    public static VerificationIndexPackageAuditReportVerification verifyVerificationIndexPackageAuditReport(
            Path reportFile) throws IOException {
        return verifyVerificationIndexPackageAuditReport(reportFile, null);
    }

    public static VerificationIndexPackageAuditReportVerification verifyVerificationIndexPackageAuditReport(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        return TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier.verify(
                reportFile,
                expectedReportSha256);
    }

    public static VerificationIndexPackageAuditReport auditVerificationIndexPackageAndWriteReport(
            Path indexFile,
            Path reportFile) throws IOException {
        return auditVerificationIndexPackageAndWriteReport(indexFile, null, reportFile);
    }

    public static VerificationIndexPackageAuditReport auditVerificationIndexPackageAndWriteReport(
            Path indexFile,
            String expectedIndexSha256,
            Path reportFile) throws IOException {
        VerificationIndexPackageAudit audit = auditVerificationIndexPackage(indexFile, expectedIndexSha256);
        return writeVerificationIndexPackageAuditReport(reportFile, audit);
    }

    public static VerificationEvidenceManifest writeVerificationEvidenceManifest(
            Path evidenceFile,
            VerifiedPackageBundle verifiedPackage) throws IOException {
        Path resolvedEvidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedEvidenceFile,
                TrainerJson.toJson(verificationEvidenceManifestMap(resolvedVerifiedPackage, Instant.now())) + "\n");
        return new VerificationEvidenceManifest(
                resolvedEvidenceFile,
                TrainerCheckpointIO.sha256Hex(resolvedEvidenceFile));
    }

    public static VerificationEvidenceInspection readVerificationEvidenceManifest(Path evidenceFile)
            throws IOException {
        return TrainingReportPromotionGateEvidenceManifestVerifier.read(evidenceFile);
    }

    public static VerificationEvidenceVerification verifyVerificationEvidenceManifest(Path evidenceFile)
            throws IOException {
        return verifyVerificationEvidenceManifest(evidenceFile, null);
    }

    public static VerificationEvidenceVerification verifyVerificationEvidenceManifest(
            Path evidenceFile,
            String expectedEvidenceSha256) throws IOException {
        return TrainingReportPromotionGateEvidenceManifestVerifier.verify(evidenceFile, expectedEvidenceSha256);
    }

    public static VerificationEvidenceReceipt writeVerificationEvidenceReceipt(
            Path receiptFile,
            VerificationEvidenceVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationEvidenceVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(verificationEvidenceReceiptMap(resolvedVerification, Instant.now())) + "\n");
        return new VerificationEvidenceReceipt(
                resolvedReceiptFile,
                TrainerCheckpointIO.sha256Hex(resolvedReceiptFile),
                resolvedVerification);
    }

    public static VerificationEvidenceReceipt verifyVerificationEvidenceManifestAndWriteReceipt(
            Path evidenceFile,
            Path receiptFile) throws IOException {
        return verifyVerificationEvidenceManifestAndWriteReceipt(evidenceFile, null, receiptFile);
    }

    public static VerificationEvidenceReceipt verifyVerificationEvidenceManifestAndWriteReceipt(
            Path evidenceFile,
            String expectedEvidenceSha256,
            Path receiptFile) throws IOException {
        VerificationEvidenceVerification verification =
                verifyVerificationEvidenceManifest(evidenceFile, expectedEvidenceSha256);
        return writeVerificationEvidenceReceipt(receiptFile, verification);
    }

    public static VerificationEvidenceReceiptInspection readVerificationEvidenceReceipt(Path receiptFile)
            throws IOException {
        return TrainingReportPromotionGateEvidenceReceiptVerifier.read(receiptFile);
    }

    public static VerificationEvidenceReceiptVerification verifyVerificationEvidenceReceipt(Path receiptFile)
            throws IOException {
        return verifyVerificationEvidenceReceipt(receiptFile, null);
    }

    public static VerificationEvidenceReceiptVerification verifyVerificationEvidenceReceipt(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateEvidenceReceiptVerifier.verify(receiptFile, expectedReceiptSha256);
    }

    public static VerificationIndexReceipt writeVerificationIndexReceipt(
            Path receiptFile,
            VerificationIndexVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationIndexVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(verificationIndexReceiptMap(resolvedVerification, Instant.now())) + "\n");
        return new VerificationIndexReceipt(
                resolvedReceiptFile,
                TrainerCheckpointIO.sha256Hex(resolvedReceiptFile),
                resolvedVerification);
    }

    public static VerificationIndexReceiptInspection readVerificationIndexReceipt(Path receiptFile)
            throws IOException {
        return TrainingReportPromotionGateVerificationIndexReceiptVerifier.read(receiptFile);
    }

    public static VerificationIndexReceiptVerification verifyVerificationIndexReceipt(Path receiptFile)
            throws IOException {
        return verifyVerificationIndexReceipt(receiptFile, null);
    }

    public static VerificationIndexReceiptVerification verifyVerificationIndexReceipt(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateVerificationIndexReceiptVerifier.verify(receiptFile, expectedReceiptSha256);
    }

    public static VerificationIndexReceipt verifyVerificationIndexAndWriteReceipt(
            Path indexFile,
            Path receiptFile) throws IOException {
        return verifyVerificationIndexAndWriteReceipt(indexFile, null, receiptFile);
    }

    public static VerificationIndexReceipt verifyVerificationIndexAndWriteReceipt(
            Path indexFile,
            String expectedIndexSha256,
            Path receiptFile) throws IOException {
        VerificationIndexVerification verification = verifyVerificationIndex(indexFile, expectedIndexSha256);
        return writeVerificationIndexReceipt(receiptFile, verification);
    }

    public static String renderVerificationJUnitXml(PackageVerification verification) {
        return TrainingReportPromotionGatePackageJUnitXml.render(verification);
    }

    public static String renderVerificationMarkdown(PackageVerification verification) {
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        PackageInspection inspection = resolvedVerification.inspection();
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifest =
                resolvedVerification.manifestVerification();
        SourceSnapshotVerification sourceSnapshots = resolvedVerification.sourceSnapshotVerification();

        StringBuilder markdown = new StringBuilder();
        markdown.append("# Gollek Promotion Gate Package Verification\n\n");
        markdown.append("| Field | Value |\n");
        markdown.append("| --- | --- |\n");
        markdownRow(markdown, "Status", status(resolvedVerification.passed()));
        markdownRow(markdown, "Package", inspection.directory().toString());
        markdownRow(markdown, "Decision", inspection.manifest().decisionStatus());
        markdownRow(markdown, "Promotable", Boolean.toString(inspection.promotable()));
        inspection.manifest().decisionCandidate()
                .ifPresent(candidate -> markdownRow(markdown, "Candidate", candidate));
        markdownRow(markdown, "Manifest", status(manifest.passed()));
        markdownRow(markdown, "Manifest checksum", matchStatus(manifest.manifestSha256Matches()));
        markdownRow(markdown, "Artifact bytes", matchStatus(manifest.artifactBytesMatch()));
        markdownRow(markdown, "Artifact checksums", matchStatus(manifest.artifactSha256Match()));
        manifest.artifactVerificationOptional()
                .ifPresent(artifactVerification -> markdownRow(
                        markdown,
                        "JUnit XML",
                        artifactVerification.junitXmlWellFormed() ? "well formed" : "invalid"));
        markdownRow(markdown, "Source report snapshots", status(sourceSnapshots.passed()));
        markdownRow(markdown, "Expected source snapshots",
                Integer.toString(sourceSnapshots.expectedSourceReportArtifactNames().size()));
        markdownRow(markdown, "Present source snapshots",
                Integer.toString(sourceSnapshots.presentSourceReportArtifactNames().size()));
        markdown.append('\n');

        appendArtifactTable(markdown, inspection);
        appendSourceSnapshotSection(markdown, sourceSnapshots);
        appendFailures(markdown, resolvedVerification.failures());
        return markdown.toString();
    }

    public static Path defaultVerificationReportFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_REPORT_FILE_NAME);
    }

    public static Path defaultVerificationMarkdownFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_MARKDOWN_FILE_NAME);
    }

    public static Path defaultVerificationJunitXmlFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_JUNIT_XML_FILE_NAME);
    }

    public static Path defaultVerificationIndexFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_INDEX_FILE_NAME);
    }

    public static Path defaultVerificationIndexReceiptFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_INDEX_RECEIPT_FILE_NAME);
    }

    public static Path defaultVerificationIndexPackageAuditFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_INDEX_PACKAGE_AUDIT_FILE_NAME);
    }

    public static Path defaultVerificationReportBundleReceiptFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_REPORT_BUNDLE_RECEIPT_FILE_NAME);
    }

    public static Path defaultVerificationEvidenceFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_EVIDENCE_FILE_NAME);
    }

    public static Path defaultVerificationEvidenceReceiptFile(Path directory) {
        return Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize()
                .resolve(DEFAULT_VERIFICATION_EVIDENCE_RECEIPT_FILE_NAME);
    }

    public static List<SourceReportSnapshot> sourceReportSnapshots(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<SourceReportSnapshot> snapshots = new ArrayList<>();
        Set<String> artifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            String artifactName = sourceReportArtifactName(report, artifactNames);
            inspection.manifest().artifact(artifactName)
                    .ifPresent(artifact -> snapshots.add(new SourceReportSnapshot(report, artifact)));
            artifactNames.add(artifactName);
        }
        return List.copyOf(snapshots);
    }

    public static List<String> expectedSourceReportArtifactNames(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        Set<String> artifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            artifactNames.add(sourceReportArtifactName(report, artifactNames));
        }
        return List.copyOf(artifactNames);
    }

    public static List<String> presentSourceReportArtifactNames(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<String> artifactNames = new ArrayList<>();
        for (String artifactName : inspection.manifest().artifacts().keySet()) {
            if (artifactName.startsWith(SOURCE_REPORT_ARTIFACT_PREFIX)) {
                artifactNames.add(artifactName);
            }
        }
        return List.copyOf(artifactNames);
    }

    public static List<String> missingSourceReportArtifactNames(PackageInspection inspection) {
        Set<String> present = new LinkedHashSet<>(presentSourceReportArtifactNames(inspection));
        List<String> missing = new ArrayList<>();
        for (String artifactName : expectedSourceReportArtifactNames(inspection)) {
            if (!present.contains(artifactName)) {
                missing.add(artifactName);
            }
        }
        return List.copyOf(missing);
    }

    public static List<String> unexpectedSourceReportArtifactNames(PackageInspection inspection) {
        Set<String> expected = new LinkedHashSet<>(expectedSourceReportArtifactNames(inspection));
        List<String> unexpected = new ArrayList<>();
        for (String artifactName : presentSourceReportArtifactNames(inspection)) {
            if (!expected.contains(artifactName)) {
                unexpected.add(artifactName);
            }
        }
        return List.copyOf(unexpected);
    }

    public static SourceSnapshotVerification verifySourceReportSnapshots(Path directory) throws IOException {
        return verifySourceReportSnapshots(read(directory));
    }

    public static SourceSnapshotVerification verifySourceReportSnapshots(Path directory, Options options)
            throws IOException {
        return verifySourceReportSnapshots(read(directory, options));
    }

    public static SourceSnapshotVerification verifySourceReportSnapshots(PackageInspection inspection)
            throws IOException {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<SourceReportSnapshot> snapshots = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        Set<String> artifactNames = new LinkedHashSet<>();
        Set<String> expectedArtifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            String artifactName = sourceReportArtifactName(report, artifactNames);
            artifactNames.add(artifactName);
            expectedArtifactNames.add(artifactName);
            TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact =
                    inspection.manifest().artifact(artifactName).orElse(null);
            if (artifact == null) {
                failures.add("Missing packaged source report snapshot for "
                        + report.role() + " " + report.name() + " (" + artifactName + ")");
                continue;
            }
            SourceReportSnapshot snapshot = new SourceReportSnapshot(report, artifact);
            snapshots.add(snapshot);
            verifySourceReportSnapshot(snapshot, failures);
        }
        verifyNoUnexpectedSourceReportArtifacts(inspection, expectedArtifactNames, failures);
        return new SourceSnapshotVerification(inspection, snapshots, failures);
    }

    private static Map<String, Object> verificationIndexMap(
            VerifiedPackageBundle verifiedPackage,
            Instant generatedAt) {
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        PackageVerification verification = resolvedVerifiedPackage.verification();
        PackageInspection inspection = verification.inspection();
        SourceSnapshotVerification sourceSnapshots = verification.sourceSnapshotVerification();

        Map<String, Object> reports = new LinkedHashMap<>();
        reports.put("json", verificationIndexFile(
                resolvedVerifiedPackage.reports().json().reportFile(),
                resolvedVerifiedPackage.reports().json().reportSha256()));
        reports.put("markdown", verificationIndexFile(
                resolvedVerifiedPackage.reports().markdown().markdownFile(),
                resolvedVerifiedPackage.reports().markdown().markdownSha256()));
        reports.put("junitXml", verificationIndexFile(
                resolvedVerifiedPackage.reports().junitXml().junitXmlFile(),
                resolvedVerifiedPackage.reports().junitXml().junitXmlSha256()));
        if (resolvedVerifiedPackage.reportBundleReceipt() != null) {
            reports.put("receipt", verificationIndexFile(
                    resolvedVerifiedPackage.reportBundleReceipt().receiptFile(),
                    resolvedVerifiedPackage.reportBundleReceipt().receiptSha256()));
        }

        Map<String, Object> sourceSnapshotIndex = new LinkedHashMap<>();
        sourceSnapshotIndex.put("expected", sourceSnapshots.expectedSourceReportArtifactNames().size());
        sourceSnapshotIndex.put("present", sourceSnapshots.presentSourceReportArtifactNames().size());
        sourceSnapshotIndex.put("missing", sourceSnapshots.missingSourceReportArtifactNames());
        sourceSnapshotIndex.put("unexpected", sourceSnapshots.unexpectedSourceReportArtifactNames());
        sourceSnapshotIndex.put("snapshots", sourceSnapshots.snapshots().stream()
                .map(TrainingReportPromotionGateArtifactPackage::sourceSnapshotIndex)
                .toList());

        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_INDEX_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("packageDirectory", resolvedVerifiedPackage.directory().toString());
        map.put("reportDirectory", resolvedVerifiedPackage.reports().directory().toString());
        map.put("passed", resolvedVerifiedPackage.passed());
        map.put("promotable", resolvedVerifiedPackage.promotable());
        map.put("decisionStatus", inspection.manifest().decisionStatus());
        inspection.manifest().decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
        map.put("manifest", verificationIndexFile(
                inspection.manifest().manifestFile(),
                inspection.manifest().manifestSha256()));
        map.put("reports", reports);
        map.put("sourceReportSnapshots", sourceSnapshotIndex);
        map.put("failures", verification.failures());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationReportMap(
            PackageVerification verification,
            Instant generatedAt) {
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_REPORT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.putAll(resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationIndexReceiptMap(
            VerificationIndexVerification verification,
            Instant generatedAt) {
        VerificationIndexVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_INDEX_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("indexFile", resolvedVerification.inspection().indexFile().toString());
        map.put("indexSha256", resolvedVerification.inspection().indexSha256());
        map.put("indexSha256Matches", resolvedVerification.indexSha256Matches());
        map.put("schemaValid", resolvedVerification.schemaValid());
        map.put("referencedSha256Match", resolvedVerification.referencedSha256Match());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationReportBundleReceiptMap(
            VerificationReportBundleVerification verification,
            Instant generatedAt) {
        VerificationReportBundleVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        VerificationReportBundleInspection inspection = resolvedVerification.inspection();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("reportDirectory", inspection.directory().toString());
        map.put("jsonReportFile", inspection.json().reportFile().toString());
        map.put("jsonReportSha256", inspection.json().reportSha256());
        map.put("markdownFile", inspection.markdownFile().toString());
        map.put("markdownSha256", inspection.markdownSha256());
        map.put("junitXmlFile", inspection.junitXmlFile().toString());
        map.put("junitXmlSha256", inspection.junitXmlSha256());
        map.put("jsonReportVerified", resolvedVerification.jsonReportVerified());
        map.put("markdownMatchesRendered", resolvedVerification.markdownMatchesRendered());
        map.put("junitXmlMatchesRendered", resolvedVerification.junitXmlMatchesRendered());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationEvidenceReceiptMap(
            VerificationEvidenceVerification verification,
            Instant generatedAt) {
        VerificationEvidenceVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_EVIDENCE_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("evidenceFile", resolvedVerification.inspection().evidenceFile().toString());
        map.put("evidenceSha256", resolvedVerification.inspection().evidenceSha256());
        map.put("evidenceSha256Matches", resolvedVerification.evidenceSha256Matches());
        map.put("schemaValid", resolvedVerification.schemaValid());
        map.put("evidenceFilesSha256Match", resolvedVerification.evidenceFilesSha256Match());
        map.put("packageArtifactsSha256Match", resolvedVerification.packageArtifactsSha256Match());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationIndexPackageAuditReportMap(
            VerificationIndexPackageAudit audit,
            Instant generatedAt) {
        VerificationIndexPackageAudit resolvedAudit = Objects.requireNonNull(audit, "audit must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedAudit.passed());
        map.put("indexFile", resolvedAudit.indexVerification().inspection().indexFile().toString());
        map.put("indexSha256", resolvedAudit.indexVerification().inspection().indexSha256());
        map.put("indexPassed", resolvedAudit.indexVerification().passed());
        map.put("packagePassed", resolvedAudit.packageVerification() != null
                && resolvedAudit.packageVerification().passed());
        map.put("failures", resolvedAudit.failures());
        map.put("audit", resolvedAudit.toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> verificationEvidenceManifestMap(
            VerifiedPackageBundle verifiedPackage,
            Instant generatedAt) {
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        PackageInspection inspection = resolvedVerifiedPackage.verification().inspection();

        Map<String, Object> evidenceFiles = new LinkedHashMap<>();
        evidenceFiles.put("manifest", verificationIndexFile(
                inspection.manifest().manifestFile(),
                inspection.manifest().manifestSha256()));
        evidenceFiles.put("verificationJson", verificationIndexFile(
                resolvedVerifiedPackage.reports().json().reportFile(),
                resolvedVerifiedPackage.reports().json().reportSha256()));
        evidenceFiles.put("verificationMarkdown", verificationIndexFile(
                resolvedVerifiedPackage.reports().markdown().markdownFile(),
                resolvedVerifiedPackage.reports().markdown().markdownSha256()));
        evidenceFiles.put("verificationJunitXml", verificationIndexFile(
                resolvedVerifiedPackage.reports().junitXml().junitXmlFile(),
                resolvedVerifiedPackage.reports().junitXml().junitXmlSha256()));
        if (resolvedVerifiedPackage.reportBundleReceipt() != null) {
            evidenceFiles.put("verificationReportBundleReceipt", verificationIndexFile(
                    resolvedVerifiedPackage.reportBundleReceipt().receiptFile(),
                    resolvedVerifiedPackage.reportBundleReceipt().receiptSha256()));
        }
        if (resolvedVerifiedPackage.index() != null) {
            evidenceFiles.put("verificationIndex", verificationIndexFile(
                    resolvedVerifiedPackage.index().indexFile(),
                    resolvedVerifiedPackage.index().indexSha256()));
        }
        if (resolvedVerifiedPackage.receipt() != null) {
            evidenceFiles.put("verificationIndexReceipt", verificationIndexFile(
                    resolvedVerifiedPackage.receipt().receiptFile(),
                    resolvedVerifiedPackage.receipt().receiptSha256()));
        }
        if (resolvedVerifiedPackage.packageAuditReport() != null) {
            evidenceFiles.put("verificationIndexPackageAudit", verificationIndexFile(
                    resolvedVerifiedPackage.packageAuditReport().reportFile(),
                    resolvedVerifiedPackage.packageAuditReport().reportSha256()));
        }

        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", VERIFICATION_EVIDENCE_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("packageDirectory", resolvedVerifiedPackage.directory().toString());
        map.put("reportDirectory", resolvedVerifiedPackage.reports().directory().toString());
        map.put("passed", resolvedVerifiedPackage.passed());
        map.put("promotable", resolvedVerifiedPackage.promotable());
        map.put("decisionStatus", inspection.manifest().decisionStatus());
        inspection.manifest().decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
        map.put("evidenceFiles", evidenceFiles);
        map.put("packageArtifacts", packageArtifactsToMap(inspection.manifest().artifacts()));
        map.put("sourceReportSnapshots", resolvedVerifiedPackage.verification()
                .sourceSnapshotVerification()
                .toMap());
        return Map.copyOf(map);
    }

    private static Optional<String> indexedManifestSha256(VerificationIndexInspection inspection) {
        return objectValue(inspection.index(), "manifest")
                .flatMap(manifest -> stringValue(manifest, "sha256"));
    }

    private static Map<String, Object> packageArtifactsToMap(
            Map<String, TrainingReportPromotionGateArtifactManifest.ArtifactEntry> artifacts) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<String, TrainingReportPromotionGateArtifactManifest.ArtifactEntry> entry : artifacts.entrySet()) {
            values.put(entry.getKey(), entry.getValue().toMap());
        }
        return Map.copyOf(values);
    }

    private static Map<String, Object> verificationIndexFile(Path file, String sha256) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("file", Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize()
                .toString());
        map.put("sha256", Objects.requireNonNull(sha256, "sha256 must not be null"));
        return Map.copyOf(map);
    }

    private static Map<String, Object> sourceSnapshotIndex(SourceReportSnapshot snapshot) {
        SourceReportSnapshot resolvedSnapshot = Objects.requireNonNull(snapshot, "snapshot must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", resolvedSnapshot.role());
        map.put("name", resolvedSnapshot.name());
        map.put("artifact", resolvedSnapshot.artifact().name());
        map.put("file", resolvedSnapshot.snapshotFile().toString());
        map.put("bytes", resolvedSnapshot.artifact().bytes());
        map.put("sha256", resolvedSnapshot.artifact().sha256());
        map.put("manifestBytesMatchSource", resolvedSnapshot.manifestBytesMatchSource());
        map.put("manifestSha256MatchesSource", resolvedSnapshot.manifestSha256MatchesSource());
        return Map.copyOf(map);
    }

    private static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    private static Optional<Path> pathValue(Map<String, ?> map, String key) {
        return stringValue(map, key).flatMap(value -> {
            try {
                return Optional.of(Path.of(value).toAbsolutePath().normalize());
            } catch (InvalidPathException ignored) {
                return Optional.empty();
            }
        });
    }

    private static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Map<?, ?> object) {
            return Optional.of(immutableMap(object));
        }
        return Optional.empty();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static Map<String, Path> packageArtifactPaths(
            TrainingReportPromotionArtifacts.ArtifactBundle review,
            Map<String, Path> sourceSnapshots) {
        Map<String, Path> artifacts = new LinkedHashMap<>();
        artifacts.put(REVIEW_JSON_ARTIFACT, review.jsonFile());
        artifacts.put(REVIEW_MARKDOWN_ARTIFACT, review.markdownFile());
        artifacts.putAll(sourceSnapshots);
        return Map.copyOf(artifacts);
    }

    private static Map<String, Path> snapshotSourceReports(
            Path directory,
            TrainingReportPromotionArtifacts.ArtifactBundle review) throws IOException {
        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                TrainingReportPromotionArtifacts.readFiles(review.jsonFile(), review.markdownFile());
        Map<String, Path> snapshots = new LinkedHashMap<>();
        Set<String> fileNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.sourceReports()) {
            if (report.source() == null || !Files.isRegularFile(report.source())) {
                continue;
            }
            String artifactName = sourceReportArtifactName(report, snapshots.keySet());
            String fileName = sourceReportFileName(report, fileNames);
            Path target = directory.resolve(fileName).toAbsolutePath().normalize();
            if (!Files.exists(target) || !Files.isSameFile(report.source(), target)) {
                Files.copy(report.source(), target, StandardCopyOption.REPLACE_EXISTING);
            }
            snapshots.put(artifactName, target);
            fileNames.add(fileName);
        }
        return Map.copyOf(snapshots);
    }

    private static String sourceReportArtifactName(
            TrainingReportPromotionArtifacts.SourceReport report,
            Set<String> existing) {
        String base = SOURCE_REPORT_ARTIFACT_PREFIX + safeToken(report.role()) + "." + safeToken(report.name());
        String name = base;
        int suffix = 2;
        while (existing.contains(name)) {
            name = base + "." + suffix++;
        }
        return name;
    }

    private static String sourceReportFileName(
            TrainingReportPromotionArtifacts.SourceReport report,
            Set<String> existing) {
        String base = "source-report-" + safeToken(report.role()) + "-" + safeToken(report.name());
        String suffix = sourceFileSuffix(report);
        String fileName = base + suffix;
        int duplicate = 2;
        while (existing.contains(fileName)) {
            fileName = base + "-" + duplicate++ + suffix;
        }
        return fileName;
    }

    private static String sourceFileSuffix(TrainingReportPromotionArtifacts.SourceReport report) {
        String sha = report.sha256();
        String fingerprint = sha == null || sha.isBlank() ? "" : "-" + sha.substring(0, Math.min(12, sha.length()));
        String extension = fileExtension(report.source());
        return fingerprint + extension;
    }

    private static String fileExtension(Path source) {
        if (source == null || source.getFileName() == null) {
            return ".json";
        }
        String fileName = source.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        if (dot <= 0 || dot == fileName.length() - 1) {
            return ".json";
        }
        String extension = fileName.substring(dot);
        return extension.length() > 16 ? ".json" : extension;
    }

    private static String safeToken(String value) {
        String text = value == null ? "report" : value.trim().toLowerCase(java.util.Locale.ROOT);
        StringBuilder token = new StringBuilder();
        for (int index = 0; index < text.length(); index++) {
            char ch = text.charAt(index);
            if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
                token.append(ch);
            } else if (token.length() > 0 && token.charAt(token.length() - 1) != '-') {
                token.append('-');
            }
        }
        while (token.length() > 0 && token.charAt(token.length() - 1) == '-') {
            token.setLength(token.length() - 1);
        }
        return token.length() == 0 ? "report" : token.toString();
    }

    private static void verifySourceReportSnapshot(SourceReportSnapshot snapshot, List<String> failures)
            throws IOException {
        TrainingReportPromotionArtifacts.SourceReport sourceReport = snapshot.sourceReport();
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact = snapshot.artifact();
        if (!snapshot.manifestBytesMatchSource()) {
            failures.add("Packaged source report snapshot byte size does not match review provenance for "
                    + sourceReport.role() + " " + sourceReport.name());
        }
        if (!snapshot.manifestSha256MatchesSource()) {
            failures.add("Packaged source report snapshot SHA-256 does not match review provenance for "
                    + sourceReport.role() + " " + sourceReport.name());
        }
        if (!Files.isRegularFile(artifact.file())) {
            failures.add("Packaged source report snapshot is missing for "
                    + sourceReport.role() + " " + sourceReport.name() + ": " + artifact.file());
            return;
        }
        long actualBytes = Files.size(artifact.file());
        if (actualBytes != artifact.bytes()) {
            failures.add("Packaged source report snapshot byte size mismatch for " + artifact.file()
                    + " (expected " + artifact.bytes() + " bytes, got " + actualBytes + " bytes)");
        }
        String actualSha256 = TrainerCheckpointIO.sha256Hex(artifact.file());
        if (!artifact.sha256().equalsIgnoreCase(actualSha256)) {
            failures.add("Packaged source report snapshot SHA-256 mismatch for " + artifact.file()
                    + " (expected " + artifact.sha256() + ", got " + actualSha256 + ")");
        }
    }

    private static void verifyNonRefreshableManifestArtifacts(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            Set<String> refreshableArtifactNames) throws IOException {
        for (TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact : manifest.artifacts().values()) {
            if (refreshableArtifactNames.contains(artifact.name())) {
                continue;
            }
            if (!Files.isRegularFile(artifact.file())) {
                throw new IOException("Cannot refresh promotion package because non-refreshable artifact is missing: "
                        + artifact.name() + " (" + artifact.file() + ")");
            }
            long actualBytes = Files.size(artifact.file());
            if (actualBytes != artifact.bytes()) {
                throw new IOException("Cannot refresh promotion package because non-refreshable artifact byte size "
                        + "changed: " + artifact.name() + " (" + artifact.file() + ")");
            }
            String actualSha256 = TrainerCheckpointIO.sha256Hex(artifact.file());
            if (!artifact.sha256().equalsIgnoreCase(actualSha256)) {
                throw new IOException("Cannot refresh promotion package because non-refreshable artifact SHA-256 "
                        + "changed: " + artifact.name() + " (" + artifact.file() + ")");
            }
        }
    }

    private static TrainingReportPromotionGateArtifactManifest.ArtifactEntry requiredManifestArtifact(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            String artifactName) throws IOException {
        return manifest.artifact(artifactName)
                .orElseThrow(() -> new IOException(
                        "Cannot refresh promotion package because manifest is missing required artifact: "
                                + artifactName));
    }

    private static void verifyNoUnexpectedSourceReportArtifacts(
            PackageInspection inspection,
            Set<String> expectedArtifactNames,
            List<String> failures) {
        for (String artifactName : inspection.manifest().artifacts().keySet()) {
            if (!artifactName.startsWith(SOURCE_REPORT_ARTIFACT_PREFIX)
                    || expectedArtifactNames.contains(artifactName)) {
                continue;
            }
            failures.add("Unexpected packaged source report snapshot in manifest: " + artifactName);
        }
    }

    private static PackageVerification packageVerification(
            PackageInspection inspection,
            TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification,
            SourceSnapshotVerification sourceSnapshotVerification) {
        List<String> failures = new ArrayList<>();
        failures.addAll(manifestVerification.failures());
        failures.addAll(sourceSnapshotVerification.failures());
        return new PackageVerification(
                inspection,
                manifestVerification,
                sourceSnapshotVerification,
                failures);
    }

    private static PackageInspection readFromManifest(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            Options options) throws IOException {
        TrainingReportPromotionGateArtifactManifest.ManifestInspection resolvedManifest =
                Objects.requireNonNull(manifest, "manifest must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path directory = resolvedManifest.directory();
        TrainingReportPromotionArtifacts.ArtifactInspection review =
                readReviewArtifacts(directory, resolvedManifest, resolvedOptions.review());
        TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts =
                TrainingReportPromotionGateArtifactManifest.readArtifacts(resolvedManifest);
        return new PackageInspection(directory, resolvedManifest, review, artifacts);
    }

    private static void appendArtifactTable(StringBuilder markdown, PackageInspection inspection) {
        markdown.append("## Package Artifacts\n\n");
        markdown.append("| Artifact | Bytes | SHA-256 |\n");
        markdown.append("| --- | ---: | --- |\n");
        for (TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact
                : inspection.manifest().artifacts().values()) {
            markdown.append("| ")
                    .append(markdownCell(artifact.name()))
                    .append(" | ")
                    .append(artifact.bytes())
                    .append(" | `")
                    .append(artifact.sha256())
                    .append("` |\n");
        }
        markdown.append('\n');
    }

    private static void appendSourceSnapshotSection(
            StringBuilder markdown,
            SourceSnapshotVerification sourceSnapshots) {
        markdown.append("## Source Report Snapshots\n\n");
        if (sourceSnapshots.snapshots().isEmpty()) {
            markdown.append("No source report snapshots were packaged.\n\n");
        } else {
            markdown.append("| Role | Name | Artifact | SHA-256 |\n");
            markdown.append("| --- | --- | --- | --- |\n");
            for (SourceReportSnapshot snapshot : sourceSnapshots.snapshots()) {
                markdown.append("| ")
                        .append(markdownCell(snapshot.role()))
                        .append(" | ")
                        .append(markdownCell(snapshot.name()))
                        .append(" | ")
                        .append(markdownCell(snapshot.artifact().name()))
                        .append(" | `")
                        .append(snapshot.artifact().sha256())
                        .append("` |\n");
            }
            markdown.append('\n');
        }
        appendNamedList(markdown, "Missing snapshots", sourceSnapshots.missingSourceReportArtifactNames());
        appendNamedList(markdown, "Unexpected snapshots", sourceSnapshots.unexpectedSourceReportArtifactNames());
    }

    private static void appendFailures(StringBuilder markdown, List<String> failures) {
        markdown.append("## Failures\n\n");
        if (failures.isEmpty()) {
            markdown.append("None.\n");
            return;
        }
        for (String failure : failures) {
            markdown.append("- ").append(markdownLine(failure)).append('\n');
        }
    }

    private static void appendNamedList(StringBuilder markdown, String title, List<String> values) {
        markdown.append("### ").append(title).append("\n\n");
        if (values.isEmpty()) {
            markdown.append("None.\n\n");
            return;
        }
        for (String value : values) {
            markdown.append("- `").append(value).append("`\n");
        }
        markdown.append('\n');
    }

    private static void markdownRow(StringBuilder markdown, String field, String value) {
        markdown.append("| ")
                .append(markdownCell(field))
                .append(" | ")
                .append(markdownCell(value))
                .append(" |\n");
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }

    private static String matchStatus(boolean matched) {
        return matched ? "match" : "mismatch";
    }

    private static String markdownCell(String value) {
        return markdownLine(value).replace("|", "\\|");
    }

    private static String markdownLine(String value) {
        return value == null ? "" : value.replace("\r", " ").replace("\n", " ").trim();
    }

    private static TrainingReportPromotionArtifacts.ArtifactInspection readReviewArtifacts(
            Path directory,
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry json =
                manifest.artifact(REVIEW_JSON_ARTIFACT).orElse(null);
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry markdown =
                manifest.artifact(REVIEW_MARKDOWN_ARTIFACT).orElse(null);
        if (json != null && markdown != null) {
            return TrainingReportPromotionArtifacts.readFiles(json.file(), markdown.file());
        }
        return TrainingReportPromotionArtifacts.read(directory, options);
    }
}
