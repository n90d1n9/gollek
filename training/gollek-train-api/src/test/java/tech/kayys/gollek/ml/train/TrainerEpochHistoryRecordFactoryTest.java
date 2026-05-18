package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerEpochHistoryRecordFactoryTest {

    @Test
    void buildsTrainRecordFromTelemetrySnapshots() {
        ThroughputSnapshot throughput = new ThroughputSnapshot(2L, 4L, 8L, 4L, 2_000_000_000L);
        Map<String, Double> metrics = Map.of("mae", 1.5);
        Map<String, Object> details = Map.of("detail", Map.of("tp", 3));

        TrainerEpochHistory.TrainRecord record = TrainerEpochHistoryRecordFactory.train(
                3,
                0.75,
                0.01,
                12,
                4,
                new TrainerOptimizationMetadata.GradientDiagnostics(
                        10.0,
                        8.0,
                        4.0,
                        3.0,
                        2,
                        16L,
                        true),
                new TrainerOptimizationMetadata.ParameterDiagnostics(
                        6.0,
                        2.5,
                        4,
                        20L),
                new TrainerEpochHistoryRecordFactory.MixedPrecisionDiagnostics(
                        true,
                        128.0,
                        false,
                        1),
                throughput,
                metrics,
                details);

        assertEquals(3, record.epoch());
        assertEquals(0.75, record.trainLoss());
        assertEquals(0.01, record.learningRate());
        assertEquals(12, record.optimizerStepCount());
        assertEquals(4, record.schedulerStepCount());
        assertEquals(10.0, record.gradientL2NormBeforeClip());
        assertEquals(8.0, record.gradientL2Norm());
        assertEquals(4.0, record.gradientMaxAbsBeforeClip());
        assertEquals(3.0, record.gradientMaxAbs());
        assertEquals(2, record.gradientParameterCount());
        assertEquals(16L, record.gradientValueCount());
        assertTrue(record.gradientClipped());
        assertEquals(6.0, record.parameterL2Norm());
        assertEquals(2.5, record.parameterMaxAbs());
        assertEquals(4, record.parameterCount());
        assertEquals(20L, record.parameterValueCount());
        assertTrue(record.mixedPrecision());
        assertEquals(128.0, record.mixedPrecisionLossScale());
        assertFalse(record.mixedPrecisionOverflowDetected());
        assertEquals(1, record.mixedPrecisionOverflowSkipCount());
        assertSame(throughput, record.trainThroughput());
        assertSame(metrics, record.trainMetrics());
        assertSame(details, record.trainMetricDetails());
    }

    @Test
    void buildsValidationRecordFromTelemetrySnapshots() {
        ThroughputSnapshot throughput = new ThroughputSnapshot(1L, 2L, 4L, 2L, 1_000_000_000L);
        Map<String, Double> metrics = Map.of("accuracy", 0.8);
        Map<String, Object> details = Map.of("confusion", Map.of("tp", 2));

        TrainerEpochHistory.ValidationRecord record = TrainerEpochHistoryRecordFactory.validation(
                2,
                0.4,
                0.005,
                9,
                throughput,
                metrics,
                details,
                0.8,
                "validationMetric.accuracy",
                "MAX");

        assertEquals(2, record.epoch());
        assertEquals(0.4, record.validationLoss());
        assertEquals(0.005, record.learningRate());
        assertEquals(9, record.schedulerStepCount());
        assertSame(throughput, record.validationThroughput());
        assertSame(metrics, record.validationMetrics());
        assertSame(details, record.validationMetricDetails());
        assertEquals(0.8, record.bestModelMonitorValue());
        assertEquals("validationMetric.accuracy", record.bestModelMonitorLabel());
        assertEquals("MAX", record.bestModelMonitorMode());
    }
}
