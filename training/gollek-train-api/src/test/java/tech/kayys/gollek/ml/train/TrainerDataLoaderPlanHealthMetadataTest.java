package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerDataLoaderPlanHealthMetadataTest {

    @Test
    void publishesUnknownHealthWhenLoaderPlanMetadataWasNotCaptured() {
        Map<String, Object> metadata = new LinkedHashMap<>();

        TrainerDataLoaderPlanHealthMetadata.put(metadata);

        assertEquals(Boolean.FALSE, metadata.get("dataLoaderPlanHealth.available"));
        assertEquals("loader-plan-metadata-not-captured", metadata.get("dataLoaderPlanHealth.skipReason"));
        assertEquals("unknown", metadata.get("dataLoaderPlanHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealthGatePassed"));
    }

    @Test
    void publishesHealthyStatusForCompleteTrainPlanWithoutValidationLoader() {
        Map<String, Object> metadata = healthyPlan();

        TrainerDataLoaderPlanHealthMetadata.put(metadata);

        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealth.available"));
        assertEquals("healthy", metadata.get("dataLoaderPlanHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealthHealthy"));
        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealthGatePassed"));
        assertEquals(0, metadata.get("dataLoaderPlanHealthIssueCount"));
        assertEquals(0, metadata.get("dataLoaderPlanHealthWarningCount"));
        assertEquals(0, metadata.get("dataLoaderPlanHealthErrorCount"));
        assertEquals(0, metadata.get("trainLoaderPlan.droppedSampleCount"));
        assertEquals(List.of(), metadata.get("dataLoaderPlanHealthIssues"));
        assertThrows(UnsupportedOperationException.class, () -> healthIssues(metadata).clear());
    }

    @Test
    void flagsDropLastAndLowSamplerCoverageAsWarnings() {
        Map<String, Object> metadata = healthyPlan();
        metadata.put("trainLoaderPlan.sampleCount", 5);
        metadata.put("trainLoaderPlan.batchCount", 2);
        metadata.put("trainLoaderPlan.dropLast", Boolean.TRUE);
        metadata.put("trainLoaderPlan.sampled", Boolean.TRUE);
        metadata.put("trainLoaderPlan.sampleCoverageRatio", 0.8);

        TrainerDataLoaderPlanHealthMetadata.put(metadata);

        assertEquals("warning", metadata.get("dataLoaderPlanHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealthGatePassed"));
        assertEquals(2, metadata.get("dataLoaderPlanHealthIssueCount"));
        assertEquals(2, metadata.get("dataLoaderPlanHealthWarningCount"));
        assertEquals(0, metadata.get("dataLoaderPlanHealthErrorCount"));
        assertEquals(1, metadata.get("trainLoaderPlan.droppedSampleCount"));
        assertEquals(
                List.of(
                        "data-loader-train-drop-last-discarded-samples",
                        "data-loader-train-low-sampler-coverage"),
                metadata.get("dataLoaderPlanHealthIssueCodes"));
        assertEquals(List.of("warning"), metadata.get("dataLoaderPlanHealthIssueSeverities"));
        assertFalse((boolean) healthIssues(metadata).get(0).get("blocking"));
    }

    @Test
    void blocksTrainingWhenRequiredTrainPlanIsUnavailable() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataLoaderPlanMetadataCaptured", Boolean.TRUE);
        metadata.put("trainLoaderPlan.available", Boolean.FALSE);
        metadata.put("trainLoaderPlan.skipReason", "unsupported-loader");
        metadata.put("validationLoaderPlan.available", Boolean.FALSE);
        metadata.put("validationLoaderPlan.skipReason", "no-loader");

        TrainerDataLoaderPlanHealthMetadata.put(metadata);

        assertEquals("error", metadata.get("dataLoaderPlanHealthStatus"));
        assertEquals(Boolean.FALSE, metadata.get("dataLoaderPlanHealthGatePassed"));
        assertEquals(1, metadata.get("dataLoaderPlanHealthIssueCount"));
        assertEquals(0, metadata.get("dataLoaderPlanHealthWarningCount"));
        assertEquals(1, metadata.get("dataLoaderPlanHealthErrorCount"));
        assertEquals(List.of("data-loader-train-plan-unavailable"), metadata.get("dataLoaderPlanHealthIssueCodes"));
        assertTrue((boolean) healthIssues(metadata).get(0).get("blocking"));
        assertEquals("train", healthIssues(metadata).get(0).get("artifact"));
    }

    private static Map<String, Object> healthyPlan() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataLoaderPlanMetadataCaptured", Boolean.TRUE);
        metadata.put("trainLoaderPlan.available", Boolean.TRUE);
        metadata.put("trainLoaderPlan.kind", "tensor");
        metadata.put("trainLoaderPlan.sampleCount", 4);
        metadata.put("trainLoaderPlan.batchSize", 2);
        metadata.put("trainLoaderPlan.batchCount", 2);
        metadata.put("trainLoaderPlan.dropLast", Boolean.FALSE);
        metadata.put("trainLoaderPlan.sampled", Boolean.FALSE);
        metadata.put("trainLoaderPlan.shuffle", Boolean.TRUE);
        metadata.put("trainLoaderPlan.reshuffleEachEpoch", Boolean.TRUE);
        metadata.put("trainLoaderPlan.hasShuffleSeed", Boolean.TRUE);
        metadata.put("trainLoaderPlan.sampleCoverageRatio", 1.0);
        metadata.put("validationLoaderPlan.available", Boolean.FALSE);
        metadata.put("validationLoaderPlan.skipReason", "no-loader");
        return metadata;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> healthIssues(Map<String, Object> metadata) {
        return (List<Map<String, Object>>) metadata.get("dataLoaderPlanHealthIssues");
    }
}
