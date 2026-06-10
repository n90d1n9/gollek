package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyPassStatus;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyPath;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyString;

import java.util.List;

/**
 * Compares persisted verification-report bundle receipts with fresh bundle verification output.
 */
final class TrainingReportPromotionGateVerificationReportBundleReceiptRevalidation {
    private TrainingReportPromotionGateVerificationReportBundleReceiptRevalidation() {
    }

    static void verifyMatches(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification revalidated,
            List<String> failures) {
        String owner = "Verification report bundle receipt";
        verifyPassStatus(inspection.receiptFile(), owner, inspection.passed(), revalidated.passed(), failures);
        verifyPath(inspection.receiptFile(), inspection.receipt(), owner,
                "jsonReportFile", revalidated.inspection().json().reportFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), owner,
                "jsonReportSha256", revalidated.inspection().json().reportSha256(), failures);
        verifyPath(inspection.receiptFile(), inspection.receipt(), owner,
                "markdownFile", revalidated.inspection().markdownFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), owner,
                "markdownSha256", revalidated.inspection().markdownSha256(), failures);
        verifyPath(inspection.receiptFile(), inspection.receipt(), owner,
                "junitXmlFile", revalidated.inspection().junitXmlFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), owner,
                "junitXmlSha256", revalidated.inspection().junitXmlSha256(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), owner,
                "jsonReportVerified", revalidated.jsonReportVerified(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), owner,
                "markdownMatchesRendered", revalidated.markdownMatchesRendered(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), owner,
                "junitXmlMatchesRendered", revalidated.junitXmlMatchesRendered(), failures);
    }
}
