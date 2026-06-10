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
 * Schema contract for terminal verification-index receipts.
 */
final class TrainingReportPromotionGateVerificationIndexReceiptSchema {
    private TrainingReportPromotionGateVerificationIndexReceiptSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_RECEIPT_FORMAT.equals(inspection.format())) {
            failures.add("Verification index receipt format mismatch for " + inspection.receiptFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_RECEIPT_FORMAT
                    + " but found " + inspection.format());
        }
        String owner = "verification index receipt";
        requireInstant(receipt, "generatedAt", owner, failures);
        requireString(receipt, "indexFile", owner, failures);
        requireSha256(receipt, "indexSha256", owner, failures);
        requireBoolean(receipt, "passed", owner, failures);
        requireBoolean(receipt, "indexSha256Matches", owner, failures);
        requireBoolean(receipt, "schemaValid", owner, failures);
        requireBoolean(receipt, "referencedSha256Match", owner, failures);
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
        String owner = "verification index receipt verification";
        requireBoolean(verification, "passed", owner, failures);
        requireBoolean(verification, "indexSha256Matches", owner, failures);
        requireBoolean(verification, "schemaValid", owner, failures);
        requireBoolean(verification, "referencedSha256Match", owner, failures);
        requireSha256(verification, "actualIndexSha256", owner, failures);
        requireStringList(verification, "failures", owner, failures);
        requireObject(verification, "inspection", owner, failures);
    }
}
