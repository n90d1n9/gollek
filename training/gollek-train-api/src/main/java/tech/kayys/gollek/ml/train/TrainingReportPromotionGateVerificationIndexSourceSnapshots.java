package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireIterable;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireNumber;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.iterableValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.stringValue;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Schema and file-reference checks for source-report snapshots in a verification index.
 */
final class TrainingReportPromotionGateVerificationIndexSourceSnapshots {
    private static final String OWNER = "verification index sourceReportSnapshots";

    private TrainingReportPromotionGateVerificationIndexSourceSnapshots() {
    }

    static void requireSchema(Map<String, ?> sourceSnapshots, List<String> failures) {
        requireNumber(sourceSnapshots, "expected", OWNER, failures);
        requireNumber(sourceSnapshots, "present", OWNER, failures);
        requireIterable(sourceSnapshots, "snapshots", OWNER, failures);
    }

    static boolean verifyReferences(Map<String, ?> sourceSnapshots, List<String> failures)
            throws IOException {
        int before = failures.size();
        List<?> snapshots = iterableValue(sourceSnapshots, "snapshots").orElse(List.of());
        int snapshotIndex = 0;
        for (Object item : snapshots) {
            if (!(item instanceof Map<?, ?> snapshotMap)) {
                failures.add("Verification index sourceReportSnapshots.snapshots["
                        + snapshotIndex + "] must be an object");
                snapshotIndex++;
                continue;
            }
            Map<String, Object> snapshot = immutableMap(snapshotMap);
            String artifact = stringValue(snapshot, "artifact").orElse(Integer.toString(snapshotIndex));
            TrainingReportPromotionGateVerificationIndexFileReferences.verify(
                    "sourceReportSnapshots." + artifact,
                    snapshot,
                    longValue(snapshot, "bytes").orElse(null),
                    failures);
            snapshotIndex++;
        }
        return failures.size() == before;
    }
}
