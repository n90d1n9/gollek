package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerDataDistributionHealthMetadataTest {

    @Test
    void publishesUnknownHealthWhenDistributionDiagnosticsAreDisabled() {
        Map<String, Object> metadata = new LinkedHashMap<>();

        TrainerDataDistributionHealthMetadata.put(metadata);

        assertEquals(Boolean.FALSE, metadata.get("dataDistributionHealth.available"));
        assertEquals("data-distribution-diagnostics-disabled", metadata.get("dataDistributionHealth.skipReason"));
        assertEquals("unknown", metadata.get("dataDistributionHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealthGatePassed"));
    }

    @Test
    void publishesHealthyStatusForBalancedTrainDistributionWithoutValidationLoader() {
        Map<String, Object> metadata = balancedClassification();

        TrainerDataDistributionHealthMetadata.put(metadata);

        assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealth.available"));
        assertEquals("healthy", metadata.get("dataDistributionHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealthHealthy"));
        assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealthGatePassed"));
        assertEquals(0, metadata.get("dataDistributionHealthIssueCount"));
        assertEquals(List.of(), metadata.get("dataDistributionHealthIssues"));
        assertThrows(UnsupportedOperationException.class, () -> healthIssues(metadata).clear());
    }

    @Test
    void flagsClassificationImbalanceAndValidationDriftAsWarnings() {
        Map<String, Object> metadata = balancedClassification();
        metadata.put("trainDataDistribution.missingClassCount", 1);
        metadata.put("trainDataDistribution.imbalanceRatio", 12.0);
        metadata.put("validationDataDistribution.available", Boolean.TRUE);
        metadata.put("validationDataDistribution.kind", "classification");
        metadata.put("validationDataDistribution.sampleCount", 10);
        metadata.put("validationDataDistribution.missingClassCount", 0);
        metadata.put("validationDataDistribution.imbalanceRatio", 1.0);
        metadata.put("dataDistributionDrift.available", Boolean.TRUE);
        metadata.put("dataDistributionDrift.kind", "classification");
        metadata.put("dataDistributionDrift.totalVariationDistance", 0.35);
        metadata.put("dataDistributionDrift.maxAbsoluteFractionDelta", 0.40);
        metadata.put("dataDistributionDrift.candidateMissingClasses", List.of(2));

        TrainerDataDistributionHealthMetadata.put(metadata);

        assertEquals("warning", metadata.get("dataDistributionHealthStatus"));
        assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealthGatePassed"));
        assertEquals(4, metadata.get("dataDistributionHealthIssueCount"));
        assertEquals(4, metadata.get("dataDistributionHealthWarningCount"));
        assertEquals(
                List.of(
                        "data-distribution-train-missing-classes",
                        "data-distribution-train-class-imbalance",
                        "data-distribution-classification-drift",
                        "data-distribution-validation-missing-classes"),
                metadata.get("dataDistributionHealthIssueCodes"));
        assertEquals(List.of("warning"), metadata.get("dataDistributionHealthIssueSeverities"));
    }

    @Test
    void flagsMultiLabelSparsityAndDriftAsWarnings() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataDistributionDiagnosticsEnabled", Boolean.TRUE);
        metadata.put("trainDataDistribution.available", Boolean.TRUE);
        metadata.put("trainDataDistribution.kind", "multi-label");
        metadata.put("trainDataDistribution.sampleCount", 12);
        metadata.put("trainDataDistribution.labelCount", 3);
        metadata.put("trainDataDistribution.zeroPositiveLabelCount", 1);
        metadata.put("trainDataDistribution.allPositiveLabelCount", 1);
        metadata.put("trainDataDistribution.positiveImbalanceRatio", 20.0);
        metadata.put("validationDataDistribution.available", Boolean.FALSE);
        metadata.put("validationDataDistribution.skipReason", "no-loader");
        metadata.put("dataDistributionDrift.available", Boolean.TRUE);
        metadata.put("dataDistributionDrift.kind", "multi-label");
        metadata.put("dataDistributionDrift.meanAbsolutePositiveFractionDelta", 0.12);
        metadata.put("dataDistributionDrift.maxAbsolutePositiveFractionDelta", 0.32);
        metadata.put("dataDistributionDrift.candidateZeroPositiveLabels", List.of(1));

        TrainerDataDistributionHealthMetadata.put(metadata);

        assertEquals("warning", metadata.get("dataDistributionHealthStatus"));
        assertEquals(5, metadata.get("dataDistributionHealthIssueCount"));
        assertEquals(
                List.of(
                        "data-distribution-train-zero-positive-labels",
                        "data-distribution-train-all-positive-labels",
                        "data-distribution-train-positive-label-imbalance",
                        "data-distribution-multilabel-drift",
                        "data-distribution-validation-zero-positive-labels"),
                metadata.get("dataDistributionHealthIssueCodes"));
    }

    private static Map<String, Object> balancedClassification() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataDistributionDiagnosticsEnabled", Boolean.TRUE);
        metadata.put("trainDataDistribution.available", Boolean.TRUE);
        metadata.put("trainDataDistribution.kind", "classification");
        metadata.put("trainDataDistribution.sampleCount", 4);
        metadata.put("trainDataDistribution.batchCount", 2);
        metadata.put("trainDataDistribution.numClasses", 2);
        metadata.put("trainDataDistribution.missingClassCount", 0);
        metadata.put("trainDataDistribution.imbalanceRatio", 1.0);
        metadata.put("trainDataDistribution.normalizedEntropy", 1.0);
        metadata.put("validationDataDistribution.available", Boolean.FALSE);
        metadata.put("validationDataDistribution.skipReason", "no-loader");
        metadata.put("dataDistributionDrift.available", Boolean.FALSE);
        metadata.put("dataDistributionDrift.skipped", Boolean.TRUE);
        metadata.put("dataDistributionDrift.skipReason", "missing-compatible-distributions");
        return metadata;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> healthIssues(Map<String, Object> metadata) {
        return (List<Map<String, Object>>) metadata.get("dataDistributionHealthIssues");
    }
}
