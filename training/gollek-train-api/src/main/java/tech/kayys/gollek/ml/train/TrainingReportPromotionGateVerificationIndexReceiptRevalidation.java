package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyFailures;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyPassStatus;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyString;

import java.util.List;
import java.util.Map;

/**
 * Compares persisted verification-index receipts with freshly verified index output.
 */
final class TrainingReportPromotionGateVerificationIndexReceiptRevalidation {
    private TrainingReportPromotionGateVerificationIndexReceiptRevalidation() {
    }

    static void verifyMatches(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification revalidated,
            List<String> failures) {
        verifyPassStatus(inspection.receiptFile(), "Verification index receipt",
                inspection.passed(), revalidated.passed(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), "Verification index receipt",
                "indexSha256", revalidated.inspection().indexSha256(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification index receipt",
                "indexSha256Matches", revalidated.indexSha256Matches(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification index receipt",
                "schemaValid", revalidated.schemaValid(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification index receipt",
                "referencedSha256Match", revalidated.referencedSha256Match(), failures);
        verifyFailures(
                inspection.receiptFile(),
                inspection.receipt(),
                "Verification index receipt",
                revalidated.failures(),
                failures);
        objectValue(inspection.receipt(), "verification").ifPresent(verification ->
                verifyEmbeddedVerificationMatchesRevalidation(
                        inspection,
                        verification,
                        revalidated,
                        failures));
    }

    private static void verifyEmbeddedVerificationMatchesRevalidation(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection inspection,
            Map<String, ?> verification,
            TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification revalidated,
            List<String> failures) {
        String owner = "Verification index receipt embedded verification";
        verifyBoolean(inspection.receiptFile(), verification, owner, "passed", revalidated.passed(), failures);
        verifyString(
                inspection.receiptFile(),
                verification,
                owner,
                "actualIndexSha256",
                revalidated.inspection().indexSha256(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                verification,
                owner,
                "indexSha256Matches",
                revalidated.indexSha256Matches(),
                failures);
        verifyBoolean(inspection.receiptFile(), verification, owner, "schemaValid", revalidated.schemaValid(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                verification,
                owner,
                "referencedSha256Match",
                revalidated.referencedSha256Match(),
                failures);
        verifyFailures(inspection.receiptFile(), verification, owner, revalidated.failures(), failures);
    }
}
