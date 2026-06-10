package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.immutableMap;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader/verifier for the terminal promotion verification-report bundle receipt.
 */
final class TrainingReportPromotionGateVerificationReportBundleReceiptVerifier {
    private TrainingReportPromotionGateVerificationReportBundleReceiptVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection read(
            Path receiptFile) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification report bundle receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification report bundle receipt at "
                    + resolvedReceiptFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection(
                resolvedReceiptFile,
                TrainingReportArtifactFingerprint.of(resolvedReceiptFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection =
                read(receiptFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                ? null
                : expectedReceiptSha256.trim();
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Verification report bundle receipt checksum mismatch for " + inspection.receiptFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid =
                TrainingReportPromotionGateVerificationReportBundleReceiptSchema.verify(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification bundleVerification = null;
        if (schemaValid && inspection.reportDirectory() != null) {
            try {
                bundleVerification = TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundle(
                        inspection.reportDirectory(),
                        inspection.jsonReportSha256());
                failures.addAll(bundleVerification.failures());
                TrainingReportPromotionGateVerificationReportBundleReceiptRevalidation.verifyMatches(
                        inspection,
                        bundleVerification,
                        failures);
            } catch (IOException error) {
                failures.add("Verification report bundle receipt could not revalidate report bundle "
                        + inspection.reportDirectory() + ": " + error.getMessage());
            }
        }
        boolean reportBundleRevalidated = bundleVerification != null && bundleVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                reportBundleRevalidated,
                bundleVerification,
                failures);
    }
}
