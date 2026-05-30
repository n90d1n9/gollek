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
                        1.25,
                        1.0,
                        2.5,
                        2.0,
                        2,
                        16L,
                        4L,
                        0.25,
                        0.8,
                        true),
                new TrainerOptimizationMetadata.ParameterDiagnostics(
                        6.0,
                        2.5,
                        0.8,
                        Math.sqrt(36.0 / 20.0),
                        4,
                        20L,
                        1L,
                        0.05),
                new TrainerOptimizationMetadata.UpdateDiagnostics(
                        true,
                        0.6,
                        0.25,
                        0.08,
                        0.1,
                        4,
                        20L,
                        18L,
                        0.9),
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
        assertEquals(1.25, record.gradientMeanAbsBeforeClip());
        assertEquals(1.0, record.gradientMeanAbs());
        assertEquals(2.5, record.gradientRmsBeforeClip());
        assertEquals(2.0, record.gradientRms());
        assertEquals(2, record.gradientParameterCount());
        assertEquals(16L, record.gradientValueCount());
        assertEquals(4L, record.gradientZeroCount());
        assertEquals(0.25, record.gradientZeroFraction());
        assertEquals(16L, record.gradientFiniteCount());
        assertEquals(0L, record.gradientNonFiniteCount());
        assertEquals(0L, record.gradientNanCount());
        assertEquals(0L, record.gradientPositiveInfinityCount());
        assertEquals(0L, record.gradientNegativeInfinityCount());
        assertEquals(0.0, record.gradientNonFiniteFraction());
        assertEquals(0.8, record.gradientClipScale());
        assertTrue(record.gradientClipped());
        assertEquals(6.0, record.parameterL2Norm());
        assertEquals(2.5, record.parameterMaxAbs());
        assertEquals(0.8, record.parameterMeanAbs());
        assertEquals(Math.sqrt(36.0 / 20.0), record.parameterRms());
        assertEquals(4, record.parameterCount());
        assertEquals(20L, record.parameterValueCount());
        assertEquals(1L, record.parameterZeroCount());
        assertEquals(0.05, record.parameterZeroFraction());
        assertEquals(20L, record.parameterFiniteCount());
        assertEquals(0L, record.parameterNonFiniteCount());
        assertEquals(0L, record.parameterNanCount());
        assertEquals(0L, record.parameterPositiveInfinityCount());
        assertEquals(0L, record.parameterNegativeInfinityCount());
        assertEquals(0.0, record.parameterNonFiniteFraction());
        assertEquals(8.0 / 6.0, record.gradientToParameterL2Ratio(), 1e-12);
        assertEquals(3.0 / 2.5, record.gradientToParameterMaxAbsRatio(), 1e-12);
        assertEquals(1.0 / 0.8, record.gradientToParameterMeanAbsRatio(), 1e-12);
        assertEquals(2.0 / Math.sqrt(36.0 / 20.0), record.gradientToParameterRmsRatio(), 1e-12);
        assertTrue(record.parameterUpdateDiagnosticsEnabled());
        assertEquals(0.6, record.parameterUpdateL2Norm());
        assertEquals(0.25, record.parameterUpdateMaxAbs());
        assertEquals(0.08, record.parameterUpdateMeanAbs());
        assertEquals(0.1, record.parameterUpdateRms());
        assertEquals(4, record.parameterUpdateCount());
        assertEquals(20L, record.parameterUpdateValueCount());
        assertEquals(18L, record.parameterUpdateZeroCount());
        assertEquals(0.9, record.parameterUpdateZeroFraction());
        assertEquals(20L, record.parameterUpdateFiniteCount());
        assertEquals(0L, record.parameterUpdateNonFiniteCount());
        assertEquals(0L, record.parameterUpdateNanCount());
        assertEquals(0L, record.parameterUpdatePositiveInfinityCount());
        assertEquals(0L, record.parameterUpdateNegativeInfinityCount());
        assertEquals(0.0, record.parameterUpdateNonFiniteFraction());
        assertEquals(0.6 / 6.0, record.parameterUpdateToParameterL2Ratio(), 1e-12);
        assertEquals(0.25 / 2.5, record.parameterUpdateToParameterMaxAbsRatio(), 1e-12);
        assertEquals(0.08 / 0.8, record.parameterUpdateToParameterMeanAbsRatio(), 1e-12);
        assertEquals(0.1 / Math.sqrt(36.0 / 20.0), record.parameterUpdateToParameterRmsRatio(), 1e-12);
        assertEquals(0.6 / 8.0, record.parameterUpdateToGradientL2Ratio(), 1e-12);
        assertEquals(0.25 / 3.0, record.parameterUpdateToGradientMaxAbsRatio(), 1e-12);
        assertEquals(0.08 / 1.0, record.parameterUpdateToGradientMeanAbsRatio(), 1e-12);
        assertEquals(0.1 / 2.0, record.parameterUpdateToGradientRmsRatio(), 1e-12);
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
