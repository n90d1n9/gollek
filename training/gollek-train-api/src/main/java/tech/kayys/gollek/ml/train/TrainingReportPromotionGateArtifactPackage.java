package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

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
            report = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(report, "report must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(report, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(report.get("passed"));
        }

        public Path packageDirectory() {
            return TrainingReportPromotionGateMapValues.objectValue(report, "inspection")
                    .flatMap(inspection -> TrainingReportPromotionGateMapValues.pathValue(inspection, "directory"))
                    .orElse(null);
        }

        public Path manifestFile() {
            return TrainingReportPromotionGateMapValues.objectValue(report, "manifestVerification")
                    .flatMap(manifest -> TrainingReportPromotionGateMapValues.objectValue(manifest, "inspection"))
                    .flatMap(inspection -> TrainingReportPromotionGateMapValues.pathValue(
                            inspection,
                            "manifestFile"))
                    .orElse(null);
        }

        public String manifestSha256() {
            return TrainingReportPromotionGateMapValues.objectValue(report, "manifestVerification")
                    .flatMap(manifest -> TrainingReportPromotionGateMapValues.stringValue(
                            manifest,
                            "actualManifestSha256"))
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
            receipt = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path reportDirectory() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "reportDirectory").orElse(null);
        }

        public Path jsonReportFile() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "jsonReportFile").orElse(null);
        }

        public String jsonReportSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "jsonReportSha256").orElse(null);
        }

        public Path markdownFile() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "markdownFile").orElse(null);
        }

        public String markdownSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "markdownSha256").orElse(null);
        }

        public Path junitXmlFile() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "junitXmlFile").orElse(null);
        }

        public String junitXmlSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "junitXmlSha256").orElse(null);
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
            index = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(index, "index must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(index, "format").orElse("UNKNOWN");
        }

        public Path packageDirectory() {
            return TrainingReportPromotionGateMapValues.pathValue(index, "packageDirectory").orElse(null);
        }

        public Path reportDirectory() {
            return TrainingReportPromotionGateMapValues.pathValue(index, "reportDirectory").orElse(null);
        }

        public boolean passed() {
            return Boolean.TRUE.equals(index.get("passed"));
        }

        public boolean promotable() {
            return Boolean.TRUE.equals(index.get("promotable"));
        }

        public String decisionStatus() {
            return TrainingReportPromotionGateMapValues.stringValue(index, "decisionStatus").orElse("UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            return TrainingReportPromotionGateMapValues.stringValue(index, "decisionCandidate");
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
            receipt = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path indexFile() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "indexFile").orElse(null);
        }

        public String indexSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "indexSha256").orElse(null);
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
            report = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(report, "report must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(report, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(report.get("passed"));
        }

        public Path indexFile() {
            return TrainingReportPromotionGateMapValues.pathValue(report, "indexFile").orElse(null);
        }

        public String indexSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(report, "indexSha256").orElse(null);
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
            evidence = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(evidence, "evidence must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(evidence, "format").orElse("UNKNOWN");
        }

        public Path packageDirectory() {
            return TrainingReportPromotionGateMapValues.pathValue(evidence, "packageDirectory").orElse(null);
        }

        public Path reportDirectory() {
            return TrainingReportPromotionGateMapValues.pathValue(evidence, "reportDirectory").orElse(null);
        }

        public boolean passed() {
            return Boolean.TRUE.equals(evidence.get("passed"));
        }

        public boolean promotable() {
            return Boolean.TRUE.equals(evidence.get("promotable"));
        }

        public String decisionStatus() {
            return TrainingReportPromotionGateMapValues.stringValue(evidence, "decisionStatus").orElse("UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            return TrainingReportPromotionGateMapValues.stringValue(evidence, "decisionCandidate");
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
            receipt = TrainingReportPromotionGateMapValues.immutableMap(
                    Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "format").orElse("UNKNOWN");
        }

        public boolean passed() {
            return Boolean.TRUE.equals(receipt.get("passed"));
        }

        public Path evidenceFile() {
            return TrainingReportPromotionGateMapValues.pathValue(receipt, "evidenceFile").orElse(null);
        }

        public String evidenceSha256() {
            return TrainingReportPromotionGateMapValues.stringValue(receipt, "evidenceSha256").orElse(null);
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
        return TrainingReportPromotionGatePackageWorkflow.write(directory, result, options, generatedAt);
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
        return TrainingReportPromotionGatePackageWorkflow.writeAndVerify(
                directory,
                result,
                options,
                generatedAt,
                reportDirectory);
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
        return TrainingReportPromotionGatePackageWorkflow.run(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                options,
                generatedAt);
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
        return TrainingReportPromotionGatePackageWorkflow.runAndVerify(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                options,
                generatedAt,
                reportDirectory);
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
        return TrainingReportPromotionGatePackageWorkflow.runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
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
        return TrainingReportPromotionGatePackageWorkflow.runWithSeverityAndVerify(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt,
                reportDirectory);
    }

    public static PackageInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static PackageInspection read(Path directory, Options options) throws IOException {
        return TrainingReportPromotionGatePackageWorkflow.read(directory, options);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(
            PackageBundle bundle) throws IOException {
        return TrainingReportPromotionGatePackageWorkflow.verify(bundle);
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
        return TrainingReportPromotionGatePackageWorkflow.verify(directory, expectedManifestSha256, options);
    }

    public static PackageVerification verifyComplete(PackageBundle bundle) throws IOException {
        return TrainingReportPromotionGatePackageWorkflow.verifyComplete(bundle);
    }

    public static VerifiedPackageBundle verifyPackage(PackageBundle bundle) throws IOException {
        return TrainingReportPromotionGatePackageWorkflow.verifyPackage(bundle);
    }

    public static VerifiedPackageBundle verifyPackage(
            PackageBundle bundle,
            Path reportDirectory) throws IOException {
        return TrainingReportPromotionGatePackageWorkflow.verifyPackage(bundle, reportDirectory);
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
        return TrainingReportPromotionGatePackageWorkflow.refreshComplete(
                directory,
                options,
                generatedAt,
                reportDirectory);
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
        return TrainingReportPromotionGatePackageWorkflow.verifyComplete(directory, expectedManifestSha256, options);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationReport(
                reportFile,
                verification);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationMarkdownReport(
                markdownFile,
                verification);
    }

    public static VerificationJUnitXmlReport writeVerificationJUnitXmlReport(
            Path junitXmlFile,
            PackageVerification verification) throws IOException {
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationJUnitXmlReport(
                junitXmlFile,
                verification);
    }

    public static VerificationReportBundle writeVerificationReports(
            Path reportDirectory,
            PackageVerification verification) throws IOException {
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationReports(
                reportDirectory,
                verification);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationReportBundleReceipt(
                receiptFile,
                verification);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationReportBundleAndWriteReceipt(
                reportDirectory,
                expectedJsonReportSha256,
                receiptFile);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationIndex(
                indexFile,
                verifiedPackage);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.auditVerificationIndexPackage(
                indexVerification);
    }

    public static VerificationIndexPackageAuditReport writeVerificationIndexPackageAuditReport(
            Path reportFile,
            VerificationIndexPackageAudit audit) throws IOException {
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationIndexPackageAuditReport(
                reportFile,
                audit);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.auditVerificationIndexPackageAndWriteReport(
                indexFile,
                expectedIndexSha256,
                reportFile);
    }

    public static VerificationEvidenceManifest writeVerificationEvidenceManifest(
            Path evidenceFile,
            VerifiedPackageBundle verifiedPackage) throws IOException {
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationEvidenceManifest(
                evidenceFile,
                verifiedPackage);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationEvidenceReceipt(
                receiptFile,
                verification);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationEvidenceManifestAndWriteReceipt(
                evidenceFile,
                expectedEvidenceSha256,
                receiptFile);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationIndexReceipt(
                receiptFile,
                verification);
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
        return TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationIndexAndWriteReceipt(
                indexFile,
                expectedIndexSha256,
                receiptFile);
    }

    public static String renderVerificationJUnitXml(PackageVerification verification) {
        return TrainingReportPromotionGatePackageJUnitXml.render(verification);
    }

    public static String renderVerificationMarkdown(PackageVerification verification) {
        return TrainingReportPromotionGatePackageMarkdown.render(verification);
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
        return TrainingReportPromotionGateSourceSnapshots.snapshots(inspection);
    }

    public static List<String> expectedSourceReportArtifactNames(PackageInspection inspection) {
        return TrainingReportPromotionGateSourceSnapshots.expectedArtifactNames(inspection);
    }

    public static List<String> presentSourceReportArtifactNames(PackageInspection inspection) {
        return TrainingReportPromotionGateSourceSnapshots.presentArtifactNames(inspection);
    }

    public static List<String> missingSourceReportArtifactNames(PackageInspection inspection) {
        return TrainingReportPromotionGateSourceSnapshots.missingArtifactNames(inspection);
    }

    public static List<String> unexpectedSourceReportArtifactNames(PackageInspection inspection) {
        return TrainingReportPromotionGateSourceSnapshots.unexpectedArtifactNames(inspection);
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
        return TrainingReportPromotionGateSourceSnapshots.verify(inspection);
    }

}
