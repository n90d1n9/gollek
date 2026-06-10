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
 * Reader/verifier for the terminal promotion verification-index receipt.
 */
final class TrainingReportPromotionGateVerificationIndexReceiptVerifier {
    private TrainingReportPromotionGateVerificationIndexReceiptVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection read(Path receiptFile)
            throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReceiptFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification index receipt JSON at "
                    + resolvedReceiptFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification index receipt at "
                    + resolvedReceiptFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection(
                resolvedReceiptFile,
                TrainingReportArtifactFingerprint.of(resolvedReceiptFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification verify(
            Path receiptFile,
            String expectedReceiptSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection inspection = read(receiptFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReceiptSha256 == null || expectedReceiptSha256.isBlank()
                ? null
                : expectedReceiptSha256.trim();
        boolean receiptSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.receiptSha256());
        if (!receiptSha256Matches) {
            failures.add("Verification index receipt checksum mismatch for " + inspection.receiptFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.receiptSha256());
        }
        boolean schemaValid = TrainingReportPromotionGateVerificationIndexReceiptSchema.verify(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification indexVerification = null;
        if (schemaValid && inspection.indexFile() != null) {
            try {
                indexVerification = TrainingReportPromotionGateArtifactPackage.verifyVerificationIndex(
                        inspection.indexFile(),
                        inspection.indexSha256());
                failures.addAll(indexVerification.failures());
                TrainingReportPromotionGateVerificationIndexReceiptRevalidation.verifyMatches(
                        inspection,
                        indexVerification,
                        failures);
            } catch (IOException error) {
                failures.add("Verification index receipt could not revalidate index "
                        + inspection.indexFile() + ": " + error.getMessage());
            }
        }
        boolean indexRevalidated = indexVerification != null && indexVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification(
                inspection,
                normalizedExpectedSha256,
                receiptSha256Matches,
                schemaValid,
                indexRevalidated,
                indexVerification,
                failures);
    }
}
