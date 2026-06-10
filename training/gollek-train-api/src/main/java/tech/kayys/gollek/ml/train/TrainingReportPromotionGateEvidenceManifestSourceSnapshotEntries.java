package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireNonNegativeNumber;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireOptionalNumber;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireOptionalSha256;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireOptionalString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireSha256;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestRequirements.requireString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.pathValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateEvidenceManifestValues.stringValue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Per-source-snapshot entry validation for promotion-gate evidence manifests.
 */
final class TrainingReportPromotionGateEvidenceManifestSourceSnapshotEntries {
    private TrainingReportPromotionGateEvidenceManifestSourceSnapshotEntries() {
    }

    static void validate(
            Map<String, ?> sourceReportSnapshots,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        List<?> snapshots = iterableValue(sourceReportSnapshots, "snapshots").orElse(List.of());
        int index = 0;
        for (Object item : snapshots) {
            if (!(item instanceof Map<?, ?> snapshotMap)) {
                failures.add(TrainingReportPromotionGateEvidenceManifestSourceSnapshots.OWNER
                        + ".snapshots[" + index + "] must be an object");
                index++;
                continue;
            }
            Map<String, Object> snapshot = immutableMap(snapshotMap);
            String owner = TrainingReportPromotionGateEvidenceManifestSourceSnapshots.OWNER
                    + ".snapshots[" + index + "]";
            requireString(snapshot, "role", owner, failures);
            requireString(snapshot, "name", owner, failures);
            requireString(snapshot, "snapshotArtifact", owner, failures);
            requireString(snapshot, "snapshotFile", owner, failures);
            requireNonNegativeNumber(snapshot, "snapshotBytes", owner, failures);
            requireSha256(snapshot, "snapshotSha256", owner, failures);
            requireBoolean(snapshot, "manifestBytesMatchSource", owner, failures);
            requireBoolean(snapshot, "manifestSha256MatchesSource", owner, failures);
            Map<String, Object> sourceReport = requireObject(snapshot, "sourceReport", owner, failures);
            if (sourceReport != null) {
                validateEmbeddedSourceReport(sourceReport, owner + ".sourceReport", failures);
            }
            validatePackageArtifactReference(snapshot, owner, packageArtifacts, failures);
            index++;
        }
    }

    private static void validatePackageArtifactReference(
            Map<String, ?> snapshot,
            String owner,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        Optional<String> artifactName = stringValue(snapshot, "snapshotArtifact");
        if (artifactName.isEmpty()) {
            return;
        }
        Object artifact = packageArtifacts == null ? null : packageArtifacts.get(artifactName.orElseThrow());
        if (artifact == null) {
            failures.add(owner + " references missing package artifact " + artifactName.orElseThrow());
            return;
        }
        if (!(artifact instanceof Map<?, ?> artifactMap)) {
            return;
        }
        Map<String, Object> artifactReference = immutableMap(artifactMap);
        comparePathField(
                snapshot,
                "snapshotFile",
                artifactReference,
                "file",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".file",
                failures);
        compareLongField(
                snapshot,
                "snapshotBytes",
                artifactReference,
                "bytes",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".bytes",
                failures);
        compareSha256Field(
                snapshot,
                "snapshotSha256",
                artifactReference,
                "sha256",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".sha256",
                failures);
    }

    private static void validateEmbeddedSourceReport(
            Map<String, ?> sourceReport,
            String owner,
            List<String> failures) {
        requireString(sourceReport, "role", owner, failures);
        requireString(sourceReport, "name", owner, failures);
        requireOptionalString(sourceReport, "source", owner, failures);
        requireOptionalNumber(sourceReport, "bytes", owner, failures);
        requireOptionalSha256(sourceReport, "sha256", owner, failures);
    }

    private static void comparePathField(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<Path> leftPath = pathValue(left, leftKey);
        Optional<Path> rightPath = pathValue(right, rightKey);
        if (leftPath.isPresent() && rightPath.isPresent() && !leftPath.orElseThrow().equals(rightPath.orElseThrow())) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }

    private static void compareLongField(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<Long> leftValue = longValue(left, leftKey);
        Optional<Long> rightValue = longValue(right, rightKey);
        if (leftValue.isPresent()
                && rightValue.isPresent()
                && leftValue.orElseThrow().longValue() != rightValue.orElseThrow().longValue()) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }

    private static void compareSha256Field(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<String> leftValue = stringValue(left, leftKey);
        Optional<String> rightValue = stringValue(right, rightKey);
        if (leftValue.isPresent()
                && rightValue.isPresent()
                && !leftValue.orElseThrow().equalsIgnoreCase(rightValue.orElseThrow())) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }
}
