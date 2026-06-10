package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.GollekDlTrainingFacade;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportAccelerationAdvisorTest {
    @Test
    void exposesTypedAccelerationViewFromReportAndFacade() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", false),
                Map.entry("executionFallback", true),
                Map.entry("acceleratedMatmulCalls", 0L)));

        TrainingReportAcceleration acceleration = report.acceleration();

        assertTrue(acceleration.available());
        assertEquals("metal", acceleration.requestedDevice());
        assertEquals("cpu", acceleration.executionBackend());
        assertTrue(acceleration.requestedAcceleratorUnavailable());
        assertTrue(acceleration.requestedAcceleratorFellBack());
        assertEquals("cpu", GollekDlTrainingFacade.trainingReportAcceleration(report).executionBackend());
        assertEquals("metal", report.accelerationMap().get("requestedDevice"));
        assertEquals(Boolean.TRUE, GollekDlTrainingFacade.trainingReportAccelerationMap(report).get("executionFallback"));

        String markdown = report.accelerationMarkdown();
        assertEquals(markdown, GollekDlTrainingFacade.trainingReportAccelerationMarkdown(report));
        assertTrue(markdown.startsWith("## Acceleration\n"));
        assertTrue(markdown.contains("| `metal` | `cpu` | CPU | `no` | `yes` | 0 |  |"));
        assertTrue(report.actionPlanMarkdown().contains("## Acceleration"));
    }

    @Test
    void recommendsFixingExplicitAcceleratorFallback() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", false),
                Map.entry("executionFallback", true),
                Map.entry("acceleratedMatmulCalls", 0L)));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().stream()
                .filter(item -> item.diagnosticCode().equals("acceleration.requested_backend_fallback"))
                .findFirst()
                .orElseThrow();

        assertEquals(TrainingReportRecommendation.Priority.HIGH, recommendation.priority());
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, recommendation.category());
        assertEquals("metal", recommendation.evidence().get("requestedDevice"));
        assertEquals("cpu", recommendation.evidence().get("executionBackend"));
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("trainer smoke job")));
        assertTrue(report.actionPlanMarkdown().contains("`acceleration.requested_backend_fallback`"));
    }

    @Test
    void recommendsVerifyingAcceleratedBackendWhenNoMatmulDeltaWasRecorded() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "auto"),
                Map.entry("executionBackend", "metal"),
                Map.entry("executionDeviceName", "Apple GPU"),
                Map.entry("executionAccelerated", true),
                Map.entry("requestedDeviceAvailable", true),
                Map.entry("executionFallback", false),
                Map.entry("acceleratedMatmulCalls", 10L),
                Map.entry("executionBackendAtStart", "metal"),
                Map.entry("executionAcceleratedAtStart", true),
                Map.entry("requestedDeviceAvailableAtStart", true),
                Map.entry("acceleratedMatmulCallsAtStart", 10L),
                Map.entry("acceleratedMatmulCallsDelta", 0L),
                Map.entry("acceleratedMatmulUsed", false),
                Map.entry("executionBackendChanged", false)));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().stream()
                .filter(item -> item.diagnosticCode().equals("acceleration.no_accelerated_matmul_delta"))
                .findFirst()
                .orElseThrow();

        assertEquals(TrainingReportRecommendation.Priority.MEDIUM, recommendation.priority());
        assertEquals(0L, recommendation.evidence().get("acceleratedMatmulCallsDelta"));
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("matrix-heavy batches")));
    }

    @Test
    void skipsRecommendationWhenCpuIsExplicitAndHealthy() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "cpu"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", true),
                Map.entry("executionFallback", false),
                Map.entry("acceleratedMatmulCalls", 0L)));

        assertFalse(report.actionPlan().recommendations().stream()
                .anyMatch(item -> item.diagnosticCode().startsWith("acceleration.")));
    }

    private static TrainingReport report(Map<String, Object> accelerationMetadata) {
        java.util.LinkedHashMap<String, Object> metadata = new java.util.LinkedHashMap<>(accelerationMetadata);
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
