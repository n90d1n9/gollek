package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireInstant;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireSha256;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireStringList;

import java.util.List;
import java.util.Map;

/**
 * Schema contract for terminal verification-evidence receipts.
 */
final class TrainingReportPromotionGateEvidenceReceiptSchema {
    private TrainingReportPromotionGateEvidenceReceiptSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification evidence receipt format mismatch for " + inspection.receiptFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT
                    + " but found " + inspection.format());
        }
        String owner = "verification evidence receipt";
        requireInstant(receipt, "generatedAt", owner, failures);
        requireString(receipt, "evidenceFile", owner, failures);
        requireSha256(receipt, "evidenceSha256", owner, failures);
        requireBoolean(receipt, "passed", owner, failures);
        requireBoolean(receipt, "evidenceSha256Matches", owner, failures);
        requireBoolean(receipt, "schemaValid", owner, failures);
        requireBoolean(receipt, "evidenceFilesSha256Match", owner, failures);
        requireBoolean(receipt, "packageArtifactsSha256Match", owner, failures);
        requireStringList(receipt, "failures", owner, failures);
        Map<String, Object> verification = requireObject(receipt, "verification", owner, failures);
        if (verification != null) {
            validateEmbeddedVerificationSchema(verification, failures);
        }
        return failures.size() == before;
    }

    private static void validateEmbeddedVerificationSchema(
            Map<String, ?> verification,
            List<String> failures) {
        String owner = "verification evidence receipt verification";
        requireBoolean(verification, "passed", owner, failures);
        requireBoolean(verification, "evidenceSha256Matches", owner, failures);
        requireBoolean(verification, "schemaValid", owner, failures);
        requireBoolean(verification, "evidenceFilesSha256Match", owner, failures);
        requireBoolean(verification, "packageArtifactsSha256Match", owner, failures);
        requireSha256(verification, "actualEvidenceSha256", owner, failures);
        requireStringList(verification, "failures", owner, failures);
        requireObject(verification, "inspection", owner, failures);
    }
}
