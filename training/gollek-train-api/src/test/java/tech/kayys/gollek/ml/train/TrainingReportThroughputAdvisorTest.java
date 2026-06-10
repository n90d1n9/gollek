package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportThroughputAdvisorTest {
    @Test
    void recommendsInvestigatingZeroTrainThroughput() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("trainBatchCount", 2L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainComputeMillis", 2_000.0),
                Map.entry("trainSamplesPerSecond", 0.0),
                Map.entry("trainBatchesPerSecond", 1.0),
                Map.entry("trainAverageBatchMillis", 1_000.0)));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().stream()
                .filter(item -> item.diagnosticCode().equals("throughput.train.zero_samples_per_second"))
                .findFirst()
                .orElseThrow();

        assertEquals(TrainingReportRecommendation.Priority.HIGH, recommendation.priority());
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, recommendation.category());
        assertEquals(16L, recommendation.evidence().get("trainSampleCount"));
        assertEquals(0.0, (double) recommendation.evidence().get("trainSamplesPerSecond"), 1e-12);
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("backend placement")));
        assertTrue(report.actionPlanMarkdown().contains("`throughput.train.zero_samples_per_second`"));
    }

    @Test
    void recommendsValidationThroughputReviewWhenValidationBatchesAreMuchSlower() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("trainBatchCount", 4L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainComputeMillis", 1_000.0),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainBatchesPerSecond", 4.0),
                Map.entry("trainAverageBatchMillis", 250.0),
                Map.entry("validationBatchCount", 2L),
                Map.entry("validationSampleCount", 8L),
                Map.entry("validationComputeMillis", 1_500.0),
                Map.entry("validationSamplesPerSecond", 8.0 / 1.5),
                Map.entry("validationBatchesPerSecond", 2.0 / 1.5),
                Map.entry("validationAverageBatchMillis", 750.0)));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().stream()
                .filter(item -> item.diagnosticCode().equals("throughput.validation.slower_than_train"))
                .findFirst()
                .orElseThrow();

        assertEquals(TrainingReportRecommendation.Priority.MEDIUM, recommendation.priority());
        assertEquals(TrainingReportRecommendation.Category.VALIDATION, recommendation.category());
        assertEquals(3.0,
                (double) recommendation.evidence().get("validationToTrainAverageBatchMillisRatio"), 1e-12);
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("validation-only metrics")));
    }

    @Test
    void skipsRecommendationWhenThroughputLooksBalanced() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("trainBatchCount", 4L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainComputeMillis", 1_000.0),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainBatchesPerSecond", 4.0),
                Map.entry("trainAverageBatchMillis", 250.0),
                Map.entry("validationBatchCount", 2L),
                Map.entry("validationSampleCount", 8L),
                Map.entry("validationComputeMillis", 600.0),
                Map.entry("validationSamplesPerSecond", 8.0 / 0.6),
                Map.entry("validationBatchesPerSecond", 2.0 / 0.6),
                Map.entry("validationAverageBatchMillis", 300.0)));

        assertFalse(report.actionPlan().recommendations().stream()
                .anyMatch(item -> item.diagnosticCode().startsWith("throughput.")));
    }

    private static TrainingReport report(Map<String, Object> throughputMetadata) {
        java.util.LinkedHashMap<String, Object> metadata = new java.util.LinkedHashMap<>(throughputMetadata);
        metadata.put("epochHistory", List.of(Map.of(
                "epoch", 0,
                "trainLoss", 0.7,
                "validationLoss", 0.8,
                "learningRate", 0.01)));
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                metadata);
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-06-10T02:03:04Z")));
    }
}
