package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyNumber;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyPath;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptStaleness.verifyString;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Compares quality-profile CI gate verification receipts with freshly verified report output.
 */
final class TrainingReportQualityProfileCiGateManifestVerificationReportReceiptRevalidation {
    private static final String OWNER = "Manifest verification report receipt";

    private TrainingReportQualityProfileCiGateManifestVerificationReportReceiptRevalidation() {
    }

    static void verifyMatches(
            TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection inspection,
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification revalidated,
            List<String> failures) throws IOException {
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "passed", revalidated.passed(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "reportVerified", revalidated.passed(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "formatValid", revalidated.formatValid(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "jsonMatchesVerification", revalidated.jsonMatchesVerification(), failures);
        verifyBoolean(
                inspection.receiptFile(),
                inspection.receipt(),
                OWNER,
                "markdownMatchesVerification",
                revalidated.markdownMatchesVerification(),
                failures);
        verifyBoolean(
                inspection.receiptFile(),
                inspection.receipt(),
                OWNER,
                "junitXmlMatchesVerification",
                revalidated.junitXmlMatchesVerification(),
                failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "junitXmlWellFormed", revalidated.junitXmlWellFormed(), failures);
        verifyBoolean(inspection.receiptFile(), inspection.receipt(), OWNER,
                "junitXmlContractValid", revalidated.junitXmlContractValid(), failures);
        Map<String, TrainingReportArtifactFingerprint> revalidatedFingerprints =
                TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.reportFingerprints(
                        revalidated.inspection());
        TrainingReportArtifactFingerprint jsonFingerprint = revalidatedFingerprints.get("json");
        TrainingReportArtifactFingerprint markdownFingerprint = revalidatedFingerprints.get("markdown");
        TrainingReportArtifactFingerprint junitXmlFingerprint = revalidatedFingerprints.get("junitXml");
        verifyPath(inspection.receiptFile(), inspection.receipt(), OWNER,
                "jsonFile", revalidated.inspection().jsonFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), OWNER,
                "jsonSha256", revalidated.inspection().jsonSha256(), failures);
        verifyNumber(inspection.receiptFile(), inspection.receipt(), OWNER,
                "jsonBytes", jsonFingerprint.bytes(), failures);
        verifyPath(inspection.receiptFile(), inspection.receipt(), OWNER,
                "markdownFile", revalidated.inspection().markdownFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), OWNER,
                "markdownSha256", revalidated.inspection().markdownSha256(), failures);
        verifyNumber(
                inspection.receiptFile(),
                inspection.receipt(),
                OWNER,
                "markdownBytes",
                markdownFingerprint.bytes(),
                failures);
        verifyPath(inspection.receiptFile(), inspection.receipt(), OWNER,
                "junitXmlFile", revalidated.inspection().junitXmlFile(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), OWNER,
                "junitXmlSha256", revalidated.inspection().junitXmlSha256(), failures);
        verifyNumber(
                inspection.receiptFile(),
                inspection.receipt(),
                OWNER,
                "junitXmlBytes",
                junitXmlFingerprint.bytes(),
                failures);
        TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.verifyMatches(
                inspection,
                revalidatedFingerprints,
                failures);
        verifyNumber(inspection.receiptFile(), inspection.receipt(), OWNER,
                "testcaseCount", revalidated.junitXmlContract().testcaseCount(), failures);
        verifyNumber(inspection.receiptFile(), inspection.receipt(), OWNER,
                "propertyCount", revalidated.junitXmlContract().propertyCount(), failures);
        verifyString(inspection.receiptFile(), inspection.receipt(), OWNER,
                "manifestStatus", revalidated.junitXmlContract().manifestStatus(), failures);
        verifyString(
                inspection.receiptFile(),
                inspection.receipt(),
                OWNER,
                "manifestReadyForRelease",
                revalidated.junitXmlContract().manifestReadyForRelease(),
                failures);
    }
}
