package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;

import java.util.List;
import java.util.Map;

/**
 * Converts loader plans into compact data health metadata for reports and CI.
 */
final class TrainerDataLoaderPlanHealthMetadata {
    private TrainerDataLoaderPlanHealthMetadata() {
    }

    static void put(Map<String, Object> metadata) {
        if (!booleanValue(metadata.get("dataLoaderPlanMetadataCaptured"))) {
            TrainerHealthMetadataPublisher.putUnavailable(
                    metadata,
                    "dataLoaderPlanHealth",
                    "loader-plan-metadata-not-captured");
            return;
        }

        List<TrainerHealthIssue> issues = TrainerDataLoaderPlanHealthRules.evaluate(metadata);
        TrainerHealthMetadataPublisher.put(metadata, "dataLoaderPlanHealth", issues);
    }
}
