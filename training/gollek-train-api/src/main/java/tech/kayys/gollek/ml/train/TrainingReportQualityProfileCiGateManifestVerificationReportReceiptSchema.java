package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Schema checks for quality-profile CI gate manifest verification receipts.
 */
final class TrainingReportQualityProfileCiGateManifestVerificationReportReceiptSchema {
    private static final String OWNER = "Manifest verification report receipt";

    private TrainingReportQualityProfileCiGateManifestVerificationReportReceiptSchema() {
    }

    static boolean verify(
            TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> receipt = inspection.receipt();
        if (!TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.FORMAT
                .equals(inspection.format())) {
            failures.add(OWNER + " format mismatch for "
                    + inspection.receiptFile() + ": expected "
                    + TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.FORMAT
                    + " but found " + inspection.format());
        }
        requireString(receipt, "createdAt", failures);
        requireString(receipt, "reportDirectory", failures);
        requireString(receipt, "jsonFile", failures);
        requireString(receipt, "jsonSha256", failures);
        requirePositiveNumber(receipt, "jsonBytes", failures);
        requireString(receipt, "markdownFile", failures);
        requireString(receipt, "markdownSha256", failures);
        requirePositiveNumber(receipt, "markdownBytes", failures);
        requireString(receipt, "junitXmlFile", failures);
        requireString(receipt, "junitXmlSha256", failures);
        requirePositiveNumber(receipt, "junitXmlBytes", failures);
        requireObject(receipt, "artifactFingerprints", failures);
        verifyArtifactFingerprintSchema(inspection, failures);
        requireBoolean(receipt, "passed", failures);
        requireBoolean(receipt, "reportVerified", failures);
        requireBoolean(receipt, "formatValid", failures);
        requireBoolean(receipt, "jsonMatchesVerification", failures);
        requireBoolean(receipt, "markdownMatchesVerification", failures);
        requireBoolean(receipt, "junitXmlMatchesVerification", failures);
        requireBoolean(receipt, "junitXmlWellFormed", failures);
        requireBoolean(receipt, "junitXmlContractValid", failures);
        requireString(receipt, "manifestStatus", failures);
        requireString(receipt, "manifestReadyForRelease", failures);
        requireNumber(receipt, "testcaseCount", failures);
        requireNumber(receipt, "propertyCount", failures);
        requireIterable(receipt, "failures", failures);
        requireObject(receipt, "junitXmlContract", failures);
        requireObject(receipt, "verificationSummary", failures);
        return failures.size() == before;
    }

    private static void verifyArtifactFingerprintSchema(
            TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection inspection,
            List<String> failures) {
        Object raw = inspection.receipt().get("artifactFingerprints");
        if (!(raw instanceof Map<?, ?> artifactMap)) {
            return;
        }
        verifyArtifactFingerprintSchema(
                artifactMap,
                "json",
                inspection.jsonFile(),
                "jsonFile",
                inspection.jsonBytes(),
                "jsonBytes",
                inspection.jsonSha256(),
                "jsonSha256",
                failures);
        verifyArtifactFingerprintSchema(
                artifactMap,
                "markdown",
                inspection.markdownFile(),
                "markdownFile",
                inspection.markdownBytes(),
                "markdownBytes",
                inspection.markdownSha256(),
                "markdownSha256",
                failures);
        verifyArtifactFingerprintSchema(
                artifactMap,
                "junitXml",
                inspection.junitXmlFile(),
                "junitXmlFile",
                inspection.junitXmlBytes(),
                "junitXmlBytes",
                inspection.junitXmlSha256(),
                "junitXmlSha256",
                failures);
    }

    private static void verifyArtifactFingerprintSchema(
            Map<?, ?> artifactMap,
            String name,
            Path flatFile,
            String flatFileKey,
            long flatBytes,
            String flatBytesKey,
            String flatSha256,
            String flatSha256Key,
            List<String> failures) {
        Object raw = artifactMap.get(name);
        if (!(raw instanceof Map<?, ?> fingerprintMap)) {
            failures.add(OWNER + " artifactFingerprints." + name + " must be an object.");
            return;
        }
        TrainingReportArtifactFingerprint fingerprint;
        try {
            fingerprint = TrainingReportArtifactFingerprint.fromMap(
                    fingerprintMap,
                    "artifactFingerprints." + name);
        } catch (IllegalArgumentException error) {
            failures.add(OWNER + " artifactFingerprints." + name
                    + " is invalid: " + error.getMessage());
            return;
        }
        TrainingReportArtifactFingerprint flatFingerprint;
        try {
            flatFingerprint = new TrainingReportArtifactFingerprint(flatFile, flatBytes, flatSha256);
        } catch (IllegalArgumentException error) {
            failures.add(OWNER + " flat fingerprint fields for " + name
                    + " are invalid: " + error.getMessage());
            return;
        }
        verifyFlatFingerprintConsistency(name, fingerprint, flatFingerprint, flatFileKey, flatBytesKey,
                flatSha256Key, failures);
    }

    private static void verifyFlatFingerprintConsistency(
            String name,
            TrainingReportArtifactFingerprint fingerprint,
            TrainingReportArtifactFingerprint flatFingerprint,
            String flatFileKey,
            String flatBytesKey,
            String flatSha256Key,
            List<String> failures) {
        String prefix = OWNER + " artifactFingerprints." + name;
        if (!fingerprint.file().equals(flatFingerprint.file())) {
            failures.add(prefix + " does not match flat field `" + flatFileKey + "`: fingerprint says "
                    + fingerprint.file() + " but flat field says " + flatFingerprint.file());
        }
        if (fingerprint.bytes() != flatFingerprint.bytes()) {
            failures.add(prefix + " does not match flat field `" + flatBytesKey + "`: fingerprint says "
                    + fingerprint.bytes() + " but flat field says " + flatFingerprint.bytes());
        }
        if (!fingerprint.sha256().equalsIgnoreCase(flatFingerprint.sha256())) {
            failures.add(prefix + " does not match flat field `" + flatSha256Key + "`.");
        }
    }

    private static void requireString(Map<String, Object> values, String key, List<String> failures) {
        Object value = values.get(key);
        if (!(value instanceof String text) || text.isBlank()) {
            failures.add(OWNER + " field `" + key + "` must be a non-blank string.");
        }
    }

    private static void requireBoolean(Map<String, Object> values, String key, List<String> failures) {
        if (!(values.get(key) instanceof Boolean)) {
            failures.add(OWNER + " field `" + key + "` must be a boolean.");
        }
    }

    private static void requireNumber(Map<String, Object> values, String key, List<String> failures) {
        if (!(values.get(key) instanceof Number)) {
            failures.add(OWNER + " field `" + key + "` must be a number.");
        }
    }

    private static void requirePositiveNumber(Map<String, Object> values, String key, List<String> failures) {
        Object value = values.get(key);
        if (!(value instanceof Number number) || number.longValue() <= 0L) {
            failures.add(OWNER + " field `" + key + "` must be a positive number.");
        }
    }

    private static void requireIterable(Map<String, Object> values, String key, List<String> failures) {
        if (!(values.get(key) instanceof Iterable<?>)) {
            failures.add(OWNER + " field `" + key + "` must be an array.");
        }
    }

    private static void requireObject(Map<String, Object> values, String key, List<String> failures) {
        if (!(values.get(key) instanceof Map<?, ?>)) {
            failures.add(OWNER + " field `" + key + "` must be an object.");
        }
    }
}
