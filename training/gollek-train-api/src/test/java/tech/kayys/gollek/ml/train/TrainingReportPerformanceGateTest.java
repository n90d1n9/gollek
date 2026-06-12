package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.GollekDlTrainingFacade;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportPerformanceGateTest {
    @TempDir
    Path tempDir;

    @Test
    void reportsUnavailableWhenPerformanceMetadataIsMissing() {
        TrainingReportPerformanceGate.Result result =
                TrainingReport.of(Map.of("metadata", Map.of())).performanceGate();

        assertFalse(result.available());
        assertTrue(result.passed());
        assertEquals("Trainer performance metadata is not available.", result.message());
        assertTrue(result.markdown().contains("**Available:** `no`"));
    }

    @Test
    void failsWhenExplicitAcceleratorFallsBack() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", false),
                Map.entry("executionFallback", true),
                Map.entry("acceleratedMatmulCalls", 0L),
                Map.entry("trainBatchCount", 2L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainAverageBatchMillis", 500.0)));

        TrainingReportPerformanceGate.Result result = report.performanceGate();

        assertTrue(result.available());
        assertFalse(result.passed());
        assertEquals("performance.accelerator_fallback", result.findings().get(0).code());
        assertEquals("failure", result.findings().get(0).severity());
        assertEquals(result.toMap(), GollekDlTrainingFacade.trainingReportPerformanceGate(report).toMap());
        assertTrue(GollekDlTrainingFacade.trainingReportPerformanceGateMarkdown(report)
                .contains("performance.accelerator_fallback"));
        assertTrue(GollekDlTrainingFacade.trainingReportPerformanceGateJUnitXml(report)
                .contains("performance-failure"));
        assertThrows(IllegalStateException.class, result::requirePassed);
    }

    @Test
    void flagsTrainThroughputBelowPolicyMinimum() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "cpu"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", true),
                Map.entry("executionFallback", false),
                Map.entry("trainBatchCount", 2L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 0.5),
                Map.entry("trainAverageBatchMillis", 500.0)));
        TrainingReportPerformanceGate.Policy policy =
                TrainingReportPerformanceGate.Policy.defaults().withMinTrainSamplesPerSecond(1.0);

        TrainingReportPerformanceGate.Result result = report.performanceGate(policy);

        assertFalse(result.passed());
        assertEquals("performance.train_throughput_below_minimum", result.findings().get(0).code());
        assertEquals(0.5, (double) result.findings().get(0).evidence().get("trainSamplesPerSecond"), 1e-12);
    }

    @Test
    void warnsWhenValidationBatchTimeIsTooHighComparedWithTrain() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "cpu"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", true),
                Map.entry("executionFallback", false),
                Map.entry("trainBatchCount", 4L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainAverageBatchMillis", 250.0),
                Map.entry("validationBatchCount", 2L),
                Map.entry("validationSampleCount", 8L),
                Map.entry("validationSamplesPerSecond", 8.0),
                Map.entry("validationAverageBatchMillis", 750.0)));

        TrainingReportPerformanceGate.Result result = report.performanceGate();

        assertFalse(result.passed());
        assertEquals("performance.validation_batch_time_skew", result.findings().get(0).code());
        assertEquals("warning", result.findings().get(0).severity());
        assertEquals(3.0,
                (double) result.findings().get(0).evidence()
                        .get("validationToTrainAverageBatchMillisRatio"), 1e-12);
    }

    @Test
    void passesWhenAccelerationAndThroughputAreHealthy() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "metal"),
                Map.entry("executionDeviceName", "Apple GPU"),
                Map.entry("executionAccelerated", true),
                Map.entry("requestedDeviceAvailable", true),
                Map.entry("executionFallback", false),
                Map.entry("acceleratedMatmulCalls", 10L),
                Map.entry("acceleratedMatmulCallsDelta", 5L),
                Map.entry("acceleratedMatmulUsed", true),
                Map.entry("trainBatchCount", 4L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 16.0),
                Map.entry("trainAverageBatchMillis", 250.0),
                Map.entry("validationBatchCount", 2L),
                Map.entry("validationSampleCount", 8L),
                Map.entry("validationSamplesPerSecond", 8.0),
                Map.entry("validationAverageBatchMillis", 300.0)));

        TrainingReportPerformanceGate.Result result = report.performanceGate();

        assertTrue(result.available());
        assertTrue(result.passed());
        assertTrue(result.markdown().contains("No performance gate findings."));
        assertTrue(result.junitXml().contains("failures=\"0\""));
    }

    @Test
    void writesVerifiesAndRefreshesPerformanceGateArtifacts() throws IOException {
        TrainingReportPerformanceGate.Result result = report(Map.ofEntries(
                Map.entry("requestedDevice", "metal"),
                Map.entry("executionBackend", "cpu"),
                Map.entry("executionDeviceName", "CPU"),
                Map.entry("executionAccelerated", false),
                Map.entry("requestedDeviceAvailable", false),
                Map.entry("executionFallback", true),
                Map.entry("trainBatchCount", 2L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainSamplesPerSecond", 0.0),
                Map.entry("trainAverageBatchMillis", 500.0)))
                .performanceGate();

        TrainingReportPerformanceGateArtifacts.ArtifactBundle bundle =
                GollekDlTrainingFacade.writeTrainingReportPerformanceGateArtifacts(tempDir, result);
        TrainingReportPerformanceGateArtifacts.ArtifactInspection inspection =
                GollekDlTrainingFacade.readTrainingReportPerformanceGateArtifacts(tempDir);
        TrainingReportPerformanceGateArtifacts.ArtifactVerification verification =
                GollekDlTrainingFacade.verifyTrainingReportPerformanceGateArtifacts(bundle);

        assertFalse(bundle.passed());
        assertFalse(inspection.passed());
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));
        assertFalse(bundle.artifact().hasManifest());
        assertEquals(bundle.artifactMap(), bundle.toMap().get("artifact"));
        assertEquals(bundle.artifactMap(), inspection.artifactMap());
        assertTrue(inspection.markdown().contains("# Gollek Trainer Performance Gate"));
        assertTrue(inspection.junitXml().contains("gollek-trainer-performance-gate"));
        assertTrue(verification.passed(), verification.message());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        verification.requirePassed();

        Files.writeString(bundle.markdownFile(), inspection.markdown() + "\n<!-- stale -->\n");
        TrainingReportPerformanceGateArtifacts.ArtifactVerification stale =
                GollekDlTrainingFacade.verifyTrainingReportPerformanceGateArtifacts(tempDir);
        assertFalse(stale.passed());
        assertFalse(stale.markdownMatchesJson());
        assertThrows(IllegalStateException.class, stale::requirePassed);

        TrainingReportPerformanceGateArtifacts.ArtifactBundle refreshed =
                GollekDlTrainingFacade.refreshTrainingReportPerformanceGateArtifacts(tempDir);
        assertTrue(GollekDlTrainingFacade.verifyTrainingReportPerformanceGateArtifacts(refreshed).passed());
    }

    private static TrainingReport report(Map<String, Object> metadataValues) {
        java.util.LinkedHashMap<String, Object> metadata = new java.util.LinkedHashMap<>(metadataValues);
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
