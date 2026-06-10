package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Payload builders for quality-profile CI gate manifest verification receipts.
 */
final class TrainingReportQualityProfileCiGateManifestVerificationReportReceiptPayload {
    private TrainingReportQualityProfileCiGateManifestVerificationReportReceiptPayload() {
    }

    static Map<String, Object> receipt(
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification,
            Instant createdAt) throws IOException {
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportInspection inspection =
                verification.inspection();
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection contract =
                verification.junitXmlContract();
        Map<String, TrainingReportArtifactFingerprint> fingerprints =
                TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.reportFingerprints(
                        inspection);
        TrainingReportArtifactFingerprint jsonFingerprint = fingerprints.get("json");
        TrainingReportArtifactFingerprint markdownFingerprint = fingerprints.get("markdown");
        TrainingReportArtifactFingerprint junitXmlFingerprint = fingerprints.get("junitXml");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.FORMAT);
        map.put("createdAt", createdAt.toString());
        map.put("reportDirectory", inspection.directory().toString());
        map.put("jsonFile", jsonFingerprint.file().toString());
        map.put("jsonSha256", jsonFingerprint.sha256());
        map.put("jsonBytes", jsonFingerprint.bytes());
        map.put("markdownFile", markdownFingerprint.file().toString());
        map.put("markdownSha256", markdownFingerprint.sha256());
        map.put("markdownBytes", markdownFingerprint.bytes());
        map.put("junitXmlFile", junitXmlFingerprint.file().toString());
        map.put("junitXmlSha256", junitXmlFingerprint.sha256());
        map.put("junitXmlBytes", junitXmlFingerprint.bytes());
        map.put(
                "artifactFingerprints",
                TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts.toMap(fingerprints));
        map.put("passed", verification.passed());
        map.put("reportVerified", verification.passed());
        map.put("formatValid", verification.formatValid());
        map.put("jsonMatchesVerification", verification.jsonMatchesVerification());
        map.put("markdownMatchesVerification", verification.markdownMatchesVerification());
        map.put("junitXmlMatchesVerification", verification.junitXmlMatchesVerification());
        map.put("junitXmlWellFormed", verification.junitXmlWellFormed());
        map.put("junitXmlContractValid", verification.junitXmlContractValid());
        map.put("junitXmlContract", contract.toMap());
        map.put("testcaseCount", contract.testcaseCount());
        map.put("propertyCount", contract.propertyCount());
        map.put("manifestStatus", contract.manifestStatus());
        map.put("manifestReadyForRelease", contract.manifestReadyForRelease());
        map.put("failures", verification.failures());
        map.put("verificationSummary", summary(verification));
        return Map.copyOf(map);
    }

    static Map<String, Object> summary(
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification) {
        TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection contract =
                verification.junitXmlContract();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("passed", verification.passed());
        map.put("formatValid", verification.formatValid());
        map.put("jsonMatchesVerification", verification.jsonMatchesVerification());
        map.put("markdownMatchesVerification", verification.markdownMatchesVerification());
        map.put("junitXmlMatchesVerification", verification.junitXmlMatchesVerification());
        map.put("junitXmlWellFormed", verification.junitXmlWellFormed());
        map.put("junitXmlContractValid", verification.junitXmlContractValid());
        map.put("testcaseCount", contract.testcaseCount());
        map.put("propertyCount", contract.propertyCount());
        map.put("manifestStatus", contract.manifestStatus());
        map.put("manifestReadyForRelease", contract.manifestReadyForRelease());
        map.put("failureCount", verification.failures().size());
        return Map.copyOf(map);
    }
}
