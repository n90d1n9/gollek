package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireNonNegativeNumber;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireSha256;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.immutableMap;

import java.util.List;
import java.util.Map;

/**
 * Package artifact reference validation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestPackageArtifacts {
    private static final String OWNER = "verification evidence packageArtifacts";

    private TrainingReportPromotionGateEvidenceManifestPackageArtifacts() {
    }

    static Map<String, Object> validate(
            Map<String, ?> evidence,
            List<String> failures) {
        Map<String, Object> packageArtifacts =
                requireObject(evidence, "packageArtifacts", "verification evidence", failures);
        if (packageArtifacts == null) {
            return null;
        }
        for (Map.Entry<String, Object> artifact : packageArtifacts.entrySet()) {
            validateArtifactReference(artifact, failures);
        }
        TrainingReportPromotionGateEvidenceManifestReferenceRoots.validatePackageArtifacts(
                evidence,
                packageArtifacts,
                failures);
        return packageArtifacts;
    }

    private static void validateArtifactReference(
            Map.Entry<String, Object> artifact,
            List<String> failures) {
        if (!(artifact.getValue() instanceof Map<?, ?> artifactMap)) {
            failures.add(OWNER + "." + artifact.getKey() + " must be an object");
            return;
        }
        Map<String, Object> artifactReference = immutableMap(artifactMap);
        String owner = OWNER + "." + artifact.getKey();
        requireString(artifactReference, "file", owner, failures);
        requireSha256(artifactReference, "sha256", owner, failures);
        requireNonNegativeNumber(artifactReference, "bytes", owner, failures);
    }
}
