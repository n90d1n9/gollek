package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Artifact fingerprint helpers for quality-profile CI gate manifest verification receipts.
 */
final class TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts {
    private static final String OWNER = "Manifest verification report receipt artifactFingerprints";

    private TrainingReportQualityProfileCiGateManifestVerificationReportReceiptArtifacts() {
    }

    static Map<String, TrainingReportArtifactFingerprint> reportFingerprints(
            TrainingReportQualityProfileCiGateManifestVerificationReport.ReportInspection inspection)
            throws IOException {
        Map<String, TrainingReportArtifactFingerprint> fingerprints = new LinkedHashMap<>();
        fingerprints.put("json", TrainingReportArtifactFingerprint.of(inspection.jsonFile()));
        fingerprints.put("markdown", TrainingReportArtifactFingerprint.of(inspection.markdownFile()));
        fingerprints.put("junitXml", TrainingReportArtifactFingerprint.of(inspection.junitXmlFile()));
        return Map.copyOf(fingerprints);
    }

    static Map<String, Object> toMap(Map<String, TrainingReportArtifactFingerprint> fingerprints) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (String name : List.of("json", "markdown", "junitXml")) {
            TrainingReportArtifactFingerprint fingerprint = fingerprints.get(name);
            if (fingerprint != null) {
                map.put(name, fingerprint.toMap());
            }
        }
        return Map.copyOf(map);
    }

    static Map<String, TrainingReportArtifactFingerprint> read(Map<String, Object> receipt) {
        Object raw = receipt.get("artifactFingerprints");
        if (!(raw instanceof Map<?, ?> artifactMap)) {
            return Map.of();
        }
        Map<String, TrainingReportArtifactFingerprint> fingerprints = new LinkedHashMap<>();
        for (String name : List.of("json", "markdown", "junitXml")) {
            Object value = artifactMap.get(name);
            if (value instanceof Map<?, ?> fingerprintMap) {
                try {
                    fingerprints.put(
                            name,
                            TrainingReportArtifactFingerprint.fromMap(
                                    fingerprintMap,
                                    "artifactFingerprints." + name));
                } catch (IllegalArgumentException ignored) {
                    // Schema verification reports malformed fingerprint fields explicitly.
                }
            }
        }
        return Map.copyOf(fingerprints);
    }

    static void verifyMatches(
            TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection inspection,
            Map<String, TrainingReportArtifactFingerprint> actualFingerprints,
            List<String> failures) {
        Map<String, TrainingReportArtifactFingerprint> recordedFingerprints = inspection.artifactFingerprints();
        for (String name : List.of("json", "markdown", "junitXml")) {
            TrainingReportArtifactFingerprint recorded = recordedFingerprints.get(name);
            TrainingReportArtifactFingerprint actual = actualFingerprints.get(name);
            if (recorded == null) {
                failures.add(OWNER + "." + name + " is missing for " + inspection.receiptFile());
            } else if (actual != null) {
                recorded.verifyMatches(OWNER + "." + name, actual, failures);
            }
        }
    }
}
