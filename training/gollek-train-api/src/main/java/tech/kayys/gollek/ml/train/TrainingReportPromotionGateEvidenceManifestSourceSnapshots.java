package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireIterable;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireObject;

import java.util.List;
import java.util.Map;

/**
 * Source report snapshot validation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestSourceSnapshots {
    static final String OWNER = "verification evidence sourceReportSnapshots";

    private TrainingReportPromotionGateEvidenceManifestSourceSnapshots() {
    }

    static void validate(
            Map<String, ?> sourceReportSnapshots,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        requireBoolean(sourceReportSnapshots, "passed", OWNER, failures);
        requireIterable(sourceReportSnapshots, "snapshots", OWNER, failures);
        requireIterable(sourceReportSnapshots, "expectedSourceReportArtifacts", OWNER, failures);
        requireIterable(sourceReportSnapshots, "presentSourceReportArtifacts", OWNER, failures);
        requireIterable(sourceReportSnapshots, "missingSourceReportArtifacts", OWNER, failures);
        requireIterable(sourceReportSnapshots, "unexpectedSourceReportArtifacts", OWNER, failures);
        requireIterable(sourceReportSnapshots, "failures", OWNER, failures);
        requireObject(sourceReportSnapshots, "inspection", OWNER, failures);
        TrainingReportPromotionGateEvidenceManifestSourceSnapshotEntries.validate(
                sourceReportSnapshots,
                packageArtifacts,
                failures);
        TrainingReportPromotionGateEvidenceManifestSourceSnapshotInventory.validate(
                sourceReportSnapshots,
                failures);
    }
}
