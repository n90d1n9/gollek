package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.difference;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.rejectDuplicateStrings;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.stringListValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Source snapshot inventory reconciliation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestSourceSnapshotInventory {
    private TrainingReportPromotionGateEvidenceManifestSourceSnapshotInventory() {
    }

    static void validate(
            Map<String, ?> sourceReportSnapshots,
            List<String> failures) {
        String owner = TrainingReportPromotionGateEvidenceManifestSourceSnapshots.OWNER;
        List<String> snapshotArtifacts = snapshotArtifactNames(sourceReportSnapshots, failures);
        Optional<List<String>> expected =
                stringListValue(sourceReportSnapshots, "expectedSourceReportArtifacts", owner, failures);
        Optional<List<String>> present =
                stringListValue(sourceReportSnapshots, "presentSourceReportArtifacts", owner, failures);
        Optional<List<String>> missing =
                stringListValue(sourceReportSnapshots, "missingSourceReportArtifacts", owner, failures);
        Optional<List<String>> unexpected =
                stringListValue(sourceReportSnapshots, "unexpectedSourceReportArtifacts", owner, failures);

        expected.ifPresent(values -> rejectDuplicateStrings(values, owner + ".expectedSourceReportArtifacts", failures));
        present.ifPresent(values -> rejectDuplicateStrings(values, owner + ".presentSourceReportArtifacts", failures));
        missing.ifPresent(values -> rejectDuplicateStrings(values, owner + ".missingSourceReportArtifacts", failures));
        unexpected.ifPresent(values -> rejectDuplicateStrings(
                values,
                owner + ".unexpectedSourceReportArtifacts",
                failures));
        rejectDuplicateStrings(snapshotArtifacts, owner + ".snapshots[].snapshotArtifact", failures);

        if (present.isPresent() && !sameStringMembers(present.orElseThrow(), snapshotArtifacts)) {
            failures.add(owner + ".presentSourceReportArtifacts does not match snapshots[].snapshotArtifact");
        }

        if (expected.isPresent() && present.isPresent() && missing.isPresent()) {
            List<String> expectedMissing = difference(expected.orElseThrow(), present.orElseThrow());
            if (!missing.orElseThrow().equals(expectedMissing)) {
                failures.add(owner + ".missingSourceReportArtifacts does not match expected-minus-present artifacts");
            }
        }
        if (expected.isPresent() && present.isPresent() && unexpected.isPresent()) {
            List<String> expectedUnexpected = difference(present.orElseThrow(), expected.orElseThrow());
            if (!unexpected.orElseThrow().equals(expectedUnexpected)) {
                failures.add(owner + ".unexpectedSourceReportArtifacts does not match present-minus-expected artifacts");
            }
        }
    }

    private static List<String> snapshotArtifactNames(
            Map<String, ?> sourceReportSnapshots,
            List<String> failures) {
        List<?> snapshots = iterableValue(sourceReportSnapshots, "snapshots").orElse(List.of());
        List<String> artifactNames = new ArrayList<>();
        int index = 0;
        for (Object item : snapshots) {
            if (item instanceof Map<?, ?> snapshotMap) {
                Optional<String> artifactName = stringValue(immutableMap(snapshotMap), "snapshotArtifact");
                artifactName.ifPresent(artifactNames::add);
            } else {
                failures.add(TrainingReportPromotionGateEvidenceManifestSourceSnapshots.OWNER
                        + ".snapshots[" + index + "] cannot be included in snapshot artifact inventory");
            }
            index++;
        }
        return List.copyOf(artifactNames);
    }

    private static boolean sameStringMembers(List<String> left, List<String> right) {
        return new LinkedHashSet<>(left).equals(new LinkedHashSet<>(right));
    }
}
