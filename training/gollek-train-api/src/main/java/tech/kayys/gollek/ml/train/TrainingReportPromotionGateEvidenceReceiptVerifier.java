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
 * Reader/verifier for the terminal promotion evidence receipt.
 */
final class TrainingReportPromotionGateEvidenceReceiptVerifier {
    private TrainingReportPromotionGateEvidenceReceiptVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection read(Path receiptFile)
            throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification evidence receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification evidence receipt at "
                    + resolvedReceiptFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection(
                resolvedReceiptFile,
                TrainingReportArtifactFingerprint.of(resolvedReceiptFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection =
                read(receiptFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 =
                TrainingReportSha256.normalizeOptional(expectedReceiptSha256, "expectedReceiptSha256");
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Verification evidence receipt checksum mismatch for " + inspection.receiptFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid = TrainingReportPromotionGateEvidenceReceiptSchema.verify(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification evidenceVerification = null;
        if (schemaValid && inspection.evidenceFile() != null) {
            try {
                evidenceVerification = TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifest(
                        inspection.evidenceFile(),
                        inspection.evidenceSha256());
                failures.addAll(evidenceVerification.failures());
                TrainingReportPromotionGateEvidenceReceiptRevalidation.verifyMatches(
                        inspection,
                        evidenceVerification,
                        failures);
            } catch (IOException error) {
                failures.add("Verification evidence receipt could not revalidate evidence manifest "
                        + inspection.evidenceFile() + ": " + error.getMessage());
            }
        }
        boolean evidenceRevalidated = evidenceVerification != null && evidenceVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                evidenceRevalidated,
                evidenceVerification,
                failures);
    }
}
