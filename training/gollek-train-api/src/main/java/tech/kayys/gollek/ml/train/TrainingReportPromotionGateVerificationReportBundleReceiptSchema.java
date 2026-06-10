package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireIterable;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptRequirements.requireString;

import java.util.List;
import java.util.Map;

/**
 * Schema contract for terminal verification-report bundle receipts.
 */
final class TrainingReportPromotionGateVerificationReportBundleReceiptSchema {
    private TrainingReportPromotionGateVerificationReportBundleReceiptSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification report bundle receipt format mismatch for " + inspection.receiptFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT
                    + " but found " + inspection.format());
        }
        String owner = "verification report bundle receipt";
        requireString(receipt, "reportDirectory", owner, failures);
        requireString(receipt, "jsonReportFile", owner, failures);
        requireString(receipt, "jsonReportSha256", owner, failures);
        requireString(receipt, "markdownFile", owner, failures);
        requireString(receipt, "markdownSha256", owner, failures);
        requireString(receipt, "junitXmlFile", owner, failures);
        requireString(receipt, "junitXmlSha256", owner, failures);
        requireBoolean(receipt, "passed", owner, failures);
        requireBoolean(receipt, "jsonReportVerified", owner, failures);
        requireBoolean(receipt, "markdownMatchesRendered", owner, failures);
        requireBoolean(receipt, "junitXmlMatchesRendered", owner, failures);
        requireIterable(receipt, "failures", owner, failures);
        requireObject(receipt, "verification", owner, failures);
        return failures.size() == before;
    }
}
