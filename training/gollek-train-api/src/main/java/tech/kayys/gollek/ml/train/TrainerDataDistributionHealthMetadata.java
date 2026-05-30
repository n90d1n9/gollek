package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;

import java.util.List;
import java.util.Map;

/**
 * Turns optional distribution diagnostics into compact, reportable data-health signals.
 */
final class TrainerDataDistributionHealthMetadata {
    private TrainerDataDistributionHealthMetadata() {
    }

    static void put(Map<String, Object> metadata) {
        if (!booleanValue(metadata.get("dataDistributionDiagnosticsEnabled"))) {
            TrainerHealthMetadataPublisher.putUnavailable(
                    metadata,
                    "dataDistributionHealth",
                    "data-distribution-diagnostics-disabled");
            return;
        }

        List<TrainerHealthIssue> issues = TrainerDataDistributionHealthRules.evaluate(metadata);
        TrainerHealthMetadataPublisher.put(metadata, "dataDistributionHealth", issues);
    }
}
