package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireFileReference;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireFileReferenceFields;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.objectValue;

import java.util.List;
import java.util.Map;

/**
 * Evidence-file reference validation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestEvidenceFiles {
    private static final String OWNER = "verification evidence evidenceFiles";

    private TrainingReportPromotionGateEvidenceManifestEvidenceFiles() {
    }

    static Map<String, Object> validate(
            Map<String, ?> evidence,
            List<String> failures) {
        Map<String, Object> evidenceFiles = requireObject(evidence, "evidenceFiles", "verification evidence", failures);
        if (evidenceFiles == null) {
            return null;
        }
        for (String required : List.of(
                "manifest",
                "verificationJson",
                "verificationMarkdown",
                "verificationJunitXml")) {
            requireFileReference(evidenceFiles, required, OWNER, failures);
        }
        for (String optional : List.of(
                "verificationReportBundleReceipt",
                "verificationIndex",
                "verificationIndexReceipt",
                "verificationIndexPackageAudit")) {
            objectValue(evidenceFiles, optional).ifPresent(reference ->
                    requireFileReferenceFields(reference, OWNER + "." + optional, failures));
        }
        TrainingReportPromotionGateEvidenceManifestReferenceRoots.validateEvidenceFiles(
                evidence,
                evidenceFiles,
                failures);
        return evidenceFiles;
    }
}
