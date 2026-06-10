package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireDirectoryPath;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireInstant;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireOptionalString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireString;

import java.util.List;
import java.util.Map;

/**
 * Schema and semantic validation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestSchema {
    private TrainingReportPromotionGateEvidenceManifestSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> evidence = inspection.evidence();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT.equals(inspection.format())) {
            failures.add("Verification evidence format mismatch for " + inspection.evidenceFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT
                    + " but found " + inspection.format());
        }
        requireInstant(evidence, "generatedAt", "verification evidence", failures);
        requireDirectoryPath(evidence, "packageDirectory", "verification evidence", failures);
        requireDirectoryPath(evidence, "reportDirectory", "verification evidence", failures);
        requireBoolean(evidence, "passed", "verification evidence", failures);
        requireBoolean(evidence, "promotable", "verification evidence", failures);
        requireString(evidence, "decisionStatus", "verification evidence", failures);
        requireOptionalString(evidence, "decisionCandidate", "verification evidence", failures);
        TrainingReportPromotionGateEvidenceManifestEvidenceFiles.validate(evidence, failures);
        Map<String, Object> packageArtifacts =
                TrainingReportPromotionGateEvidenceManifestPackageArtifacts.validate(evidence, failures);
        Map<String, Object> sourceReportSnapshots =
                requireObject(evidence, "sourceReportSnapshots", "verification evidence", failures);
        if (sourceReportSnapshots != null) {
            TrainingReportPromotionGateEvidenceManifestSourceSnapshots.validate(
                    sourceReportSnapshots,
                    packageArtifacts,
                    failures);
        }
        return failures.size() == before;
    }
}
