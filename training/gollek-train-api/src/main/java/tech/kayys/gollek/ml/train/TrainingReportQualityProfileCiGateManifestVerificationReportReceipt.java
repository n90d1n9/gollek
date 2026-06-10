package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Terminal receipt for a quality-profile CI gate manifest verification report bundle.
 */
public final class TrainingReportQualityProfileCiGateManifestVerificationReportReceipt {
    public static final String FORMAT =
            "gollek.training.quality-profile.ci-gate.manifest.verification.receipt.v1";
    public static final String DEFAULT_FILE_NAME =
            "quality-profile-ci-gate-manifest-verification.receipt.json";

    private TrainingReportQualityProfileCiGateManifestVerificationReportReceipt() {
    }

    public record Receipt(
            Path receiptFile,
            String receiptSha256,
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification) {
        public Receipt {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            receiptSha256 = TrainingReportMapValues.requireChecksum(receiptSha256, "receiptSha256");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("passed", passed());
            map.put(
                    "verificationSummary",
                    TrainingReportQualityProfileCiGateManifestVerificationReportReceiptPayload.summary(verification));
            return Map.copyOf(map);
        }
    }

    public record ReceiptInspection(
            Path receiptFile,
            String receiptSha256,
            Map<String, Object> receipt) {
        public ReceiptInspection {
            receiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            receiptSha256 = TrainingReportMapValues.requireChecksum(receiptSha256, "receiptSha256");
            receipt = TrainingReportMapValues.immutableMap(
                    Objects.requireNonNull(receipt, "receipt must not be null"));
        }

        public String format() {
            return TrainingReportMapValues.stringValue(receipt, "format", "");
        }

        public Path reportDirectory() {
            return TrainingReportMapValues.pathValue(receipt, "reportDirectory").orElse(null);
        }

        public Path jsonFile() {
            return TrainingReportMapValues.pathValue(receipt, "jsonFile").orElse(null);
        }

        public Path markdownFile() {
            return TrainingReportMapValues.pathValue(receipt, "markdownFile").orElse(null);
        }

        public Path junitXmlFile() {
            return TrainingReportMapValues.pathValue(receipt, "junitXmlFile").orElse(null);
        }

        public String jsonSha256() {
            return TrainingReportMapValues.stringValue(receipt, "jsonSha256", "");
        }

        public long jsonBytes() {
            return TrainingReportMapValues.longValue(receipt, "jsonBytes", -1L);
        }

        public String markdownSha256() {
            return TrainingReportMapValues.stringValue(receipt, "markdownSha256", "");
        }

        public long markdownBytes() {
            return TrainingReportMapValues.longValue(receipt, "markdownBytes", -1L);
        }

        public String junitXmlSha256() {
            return TrainingReportMapValues.stringValue(receipt, "junitXmlSha256", "");
        }

        public long junitXmlBytes() {
            return TrainingReportMapValues.longValue(receipt, "junitXmlBytes", -1L);
        }

        public TrainingReportArtifactFingerprint jsonFingerprint() {
            return artifactFingerprints().getOrDefault(
                    "json",
                    new TrainingReportArtifactFingerprint(jsonFile(), jsonBytes(), jsonSha256()));
        }

        public TrainingReportArtifactFingerprint markdownFingerprint() {
            return artifactFingerprints().getOrDefault(
                    "markdown",
                    new TrainingReportArtifactFingerprint(markdownFile(), markdownBytes(), markdownSha256()));
        }

        public TrainingReportArtifactFingerprint junitXmlFingerprint() {
            return artifactFingerprints().getOrDefault(
                    "junitXml",
                    new TrainingReportArtifactFingerprint(junitXmlFile(), junitXmlBytes(), junitXmlSha256()));
        }

        public Map<String, TrainingReportArtifactFingerprint> artifactFingerprints() {
            return TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.read(receipt);
        }

        public boolean passed() {
            return TrainingReportMapValues.booleanValue(receipt, "passed", false);
        }

        public boolean reportVerified() {
            return TrainingReportMapValues.booleanValue(receipt, "reportVerified", false);
        }

        public boolean junitXmlContractValid() {
            return TrainingReportMapValues.booleanValue(receipt, "junitXmlContractValid", false);
        }

        public String manifestStatus() {
            return TrainingReportMapValues.stringValue(receipt, "manifestStatus", "");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("receiptFile", receiptFile.toString());
            map.put("receiptSha256", receiptSha256);
            map.put("format", format());
            map.put("reportDirectory", reportDirectory() == null ? "" : reportDirectory().toString());
            map.put("jsonFile", jsonFile() == null ? "" : jsonFile().toString());
            map.put("markdownFile", markdownFile() == null ? "" : markdownFile().toString());
            map.put("junitXmlFile", junitXmlFile() == null ? "" : junitXmlFile().toString());
            map.put("jsonBytes", jsonBytes());
            map.put("markdownBytes", markdownBytes());
            map.put("junitXmlBytes", junitXmlBytes());
            map.put(
                    "artifactFingerprints",
                    TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.toMap(
                            artifactFingerprints()));
            map.put("passed", passed());
            map.put("reportVerified", reportVerified());
            map.put("junitXmlContractValid", junitXmlContractValid());
            map.put("manifestStatus", manifestStatus());
            map.put("receipt", receipt);
            return Map.copyOf(map);
        }
    }

    public record ReceiptVerification(
            ReceiptInspection inspection,
            String expectedReceiptSha256,
            boolean receiptSha256Matches,
            boolean schemaValid,
            boolean reportRevalidated,
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification reportVerification,
            List<String> failures) {
        public ReceiptVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedReceiptSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                    ? ""
                    : expectedReceiptSha256.trim().toLowerCase();
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
                return "Quality-profile CI gate manifest verification report receipt verified for "
                        + inspection.receiptFile() + ".";
            }
            return "Quality-profile CI gate manifest verification report receipt failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("expectedReceiptSha256", expectedReceiptSha256);
            map.put("receiptSha256Matches", receiptSha256Matches);
            map.put("schemaValid", schemaValid);
            map.put("reportRevalidated", reportRevalidated);
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            if (reportVerification != null) {
                map.put("reportVerification", reportVerification.toMap());
            }
            return Map.copyOf(map);
        }
    }

    public static Receipt write(
            Path receiptFile,
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification)
            throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(
                        TrainingReportQualityProfileCiGateManifestVerificationReportReceiptPayload.receipt(
                                resolvedVerification,
                                Instant.now()))
                        + "\n");
        TrainingReportArtifactFingerprint receiptFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedReceiptFile);
        return new Receipt(
                resolvedReceiptFile,
                receiptFingerprint.sha256(),
                resolvedVerification);
    }

    public static Receipt write(
            Path outputDirectory,
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification,
            String receiptFileName) throws IOException {
        String resolvedFileName = receiptFileName == null || receiptFileName.isBlank()
                ? DEFAULT_FILE_NAME
                : receiptFileName.trim();
        if (resolvedFileName.contains("/") || resolvedFileName.contains("\\")) {
            throw new IllegalArgumentException("receiptFileName must be a file name, not a path");
        }
        return write(
                Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                        .toAbsolutePath()
                        .normalize()
                        .resolve(resolvedFileName),
                verification);
    }

    public static ReceiptInspection read(Path receiptFile) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid quality-profile CI manifest verification report receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object quality-profile CI manifest verification report receipt at "
                    + resolvedReceiptFile);
        }
        TrainingReportArtifactFingerprint receiptFingerprint =
                TrainingReportArtifactFingerprint.of(resolvedReceiptFile);
        return new ReceiptInspection(
                resolvedReceiptFile,
                receiptFingerprint.sha256(),
                TrainingReportMapValues.immutableMap(map));
    }

    public static ReceiptVerification verify(
            Path receiptFile,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification manifestVerification)
            throws IOException {
        return verify(receiptFile, null, manifestVerification);
    }

    public static ReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification manifestVerification)
            throws IOException {
        ReceiptInspection inspection = read(receiptFile);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolvedManifestVerification =
                Objects.requireNonNull(manifestVerification, "manifestVerification must not be null");
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                ? null
                : expectedReceiptSha256.trim().toLowerCase();
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Manifest verification report receipt checksum mismatch for "
                    + inspection.receiptFile() + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid =
                TrainingReportQualityProfileCiGateManifestVerificationReportReceiptSchema.verify(inspection, failures);
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification reportVerification = null;
        if (schemaValid) {
            try {
                reportVerification = TrainingReportQualityProfileCiGateManifestVerificationReport.verify(
                        inspection.reportDirectory(),
                        resolvedManifestVerification);
                failures.addAll(reportVerification.failures());
                TrainingReportQualityProfileCiGateManifestVerificationReportReceiptRevalidation.verifyMatches(
                        inspection,
                        reportVerification,
                        failures);
            } catch (IOException error) {
                failures.add("Manifest verification report receipt could not revalidate report bundle "
                        + inspection.reportDirectory() + ": " + error.getMessage());
            }
        }
        boolean reportRevalidated = reportVerification != null && reportVerification.passed();
        return new ReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                reportRevalidated,
                reportVerification,
                failures);
    }

}
