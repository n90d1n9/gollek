package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.objectValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyFailures;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyPassStatus;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyString;

import java.util.List;
import java.util.Map;

/**
 * Compares persisted verification-evidence receipts with freshly verified evidence output.
 */
final class TrainingReportPromotionGateEvidenceReceiptRevalidation {
    private TrainingReportPromotionGateEvidenceReceiptRevalidation() {
    }

    static void verifyMatches(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification revalidated,
            List<String> failures) {
        verifyPassStatus(inspection.receiptFile(), "Verification evidence receipt",
                inspection.passed(), revalidated.passed(), failures);
        verifyString(
                inspection.receiptFile(),
                inspection.receipt(),
                "Verification evidence receipt",
                "evidenceSha256",
                revalidated.inspection().evidenceSha256(),
                failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification evidence receipt",
                "evidenceSha256Matches", revalidated.evidenceSha256Matches(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification evidence receipt",
                "schemaValid", revalidated.schemaValid(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), "Verification evidence receipt",
                "evidenceFilesSha256Match", revalidated.evidenceFilesSha256Match(), failures);
        verifyBoolean(
                inspection.receiptFile(),
                inspection.receipt(),
                "Verification evidence receipt",
                "packageArtifactsSha256Match",
                revalidated.packageArtifactsSha256Match(),
                failures);
        verifyFailures(
                inspection.receiptFile(),
                inspection.receipt(),
                "Verification evidence receipt",
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
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            Map<String, ?> verification,
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification revalidated,
            List<String> failures) {
        String owner = "Verification evidence receipt embedded verification";
        verifyBoolean(inspection.receiptFile(), verification, owner, "passed", revalidated.passed(), failures);
        verifyString(
                inspection.receiptFile(),
                verification,
                owner,
                "actualEvidenceSha256",
                revalidated.inspection().evidenceSha256(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                verification,
                owner,
                "evidenceSha256Matches",
                revalidated.evidenceSha256Matches(),
                failures);
        verifyBoolean(inspection.receiptFile(), verification, owner, "schemaValid", revalidated.schemaValid(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                verification,
                owner,
                "evidenceFilesSha256Match",
                revalidated.evidenceFilesSha256Match(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                verification,
                owner,
                "packageArtifactsSha256Match",
                revalidated.packageArtifactsSha256Match(),
                failures);
        verifyFailures(inspection.receiptFile(), verification, owner, revalidated.failures(), failures);
    }
}
