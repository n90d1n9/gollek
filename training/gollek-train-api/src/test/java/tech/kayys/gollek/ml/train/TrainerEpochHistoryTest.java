package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerEpochHistoryTest {

    @Test
    void recordsTrainAndValidationIntoSingleFlattenedEpochRow() {
        TrainerEpochHistory history = new TrainerEpochHistory();

        history.recordTrain(new TrainerEpochHistory.TrainRecord(
                0,
                5.0,
                0.1,
                2,
                3,
                10.0,
                8.0,
                4.0,
                3.0,
                1.5,
                1.25,
                2.5,
                2.0,
                2,
                12L,
                3L,
                0.25,
                12L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.8,
                true,
                6.0,
                2.5,
                0.9,
                Math.sqrt(36.0 / 20.0),
                4,
                20L,
                2L,
                0.1,
                20L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                8.0 / 6.0,
                3.0 / 2.5,
                1.25 / 0.9,
                2.0 / Math.sqrt(36.0 / 20.0),
                true,
                0.6,
                0.25,
                0.08,
                0.1,
                4,
                20L,
                18L,
                0.9,
                20L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.6 / 6.0,
                0.25 / 2.5,
                0.08 / 0.9,
                0.1 / Math.sqrt(36.0 / 20.0),
                0.6 / 8.0,
                0.25 / 3.0,
                0.08 / 1.25,
                0.1 / 2.0,
                true,
                16.0,
                false,
                0,
                new ThroughputSnapshot(2L, 4L, 8L, 4L, 2_000_000_000L),
                Map.of("mae", 2.0),
                Map.of("detail", Map.of("tp", 3L))));
        history.recordValidation(new TrainerEpochHistory.ValidationRecord(
                0,
                4.0,
                0.05,
                4,
                new ThroughputSnapshot(1L, 2L, 4L, 2L, 1_000_000_000L),
                Map.of("mae", 1.5),
                Map.of("detail", Map.of("tp", 2L)),
                1.5,
                "validationMetric.mae",
                "MIN"));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(1, rows.size());
        Map<String, Object> row = rows.get(0);

        assertEquals(0, row.get("epoch"));
        assertEquals(5.0, row.get("trainLoss"));
        assertEquals(4.0, row.get("validationLoss"));
        assertEquals(0.05, row.get("learningRate"));
        assertEquals(2, row.get("optimizerStepCount"));
        assertEquals(4, row.get("schedulerStepCount"));
        assertEquals(-1.0, row.get("generalizationGap"));
        assertEquals(0.8, row.get("validationToTrainLossRatio"));
        assertEquals(Boolean.FALSE, row.get("validationLossAboveTrainLoss"));
        assertEquals(2.0, row.get("trainMetric.mae"));
        assertEquals(1.5, row.get("validationMetric.mae"));
        assertEquals("validationMetric.mae", row.get("bestModelMonitor"));
        assertEquals("MIN", row.get("bestModelMonitorMode"));
        assertEquals(1.5, row.get("bestModelMonitorValue"));
        assertEquals(Boolean.TRUE, row.get("bestModelMonitorProgressAvailable"));
        assertEquals(1.5, row.get("bestModelMonitorBestValue"));
        assertEquals(0, row.get("bestModelMonitorBestEpoch"));
        assertEquals(0.0, row.get("bestModelMonitorDistanceFromBest"));
        assertEquals(0, row.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.TRUE, row.get("bestModelMonitorIsBest"));
        assertEquals(2L, row.get("trainBatchCount"));
        assertEquals(1L, row.get("validationBatchCount"));
        assertEquals(1.5, row.get("gradientMeanAbsBeforeClip"));
        assertEquals(1.25, row.get("gradientMeanAbs"));
        assertEquals(2.5, row.get("gradientRmsBeforeClip"));
        assertEquals(2.0, row.get("gradientRms"));
        assertEquals(3L, row.get("gradientZeroCount"));
        assertEquals(0.25, row.get("gradientZeroFraction"));
        assertEquals(12L, row.get("gradientFiniteCount"));
        assertEquals(0L, row.get("gradientNonFiniteCount"));
        assertEquals(0.0, row.get("gradientNonFiniteFraction"));
        assertEquals(0.8, row.get("gradientClipScale"));
        assertEquals(0.9, row.get("parameterMeanAbs"));
        assertEquals(Math.sqrt(36.0 / 20.0), (double) row.get("parameterRms"), 1e-12);
        assertEquals(2L, row.get("parameterZeroCount"));
        assertEquals(0.1, row.get("parameterZeroFraction"));
        assertEquals(20L, row.get("parameterFiniteCount"));
        assertEquals(0L, row.get("parameterNonFiniteCount"));
        assertEquals(0.0, row.get("parameterNonFiniteFraction"));
        assertEquals(8.0 / 6.0, (double) row.get("gradientToParameterL2Ratio"), 1e-12);
        assertEquals(3.0 / 2.5, (double) row.get("gradientToParameterMaxAbsRatio"), 1e-12);
        assertEquals(1.25 / 0.9, (double) row.get("gradientToParameterMeanAbsRatio"), 1e-12);
        assertEquals(2.0 / Math.sqrt(36.0 / 20.0), (double) row.get("gradientToParameterRmsRatio"), 1e-12);
        assertEquals(Boolean.TRUE, row.get("parameterUpdateDiagnosticsEnabled"));
        assertEquals(0.6, row.get("parameterUpdateL2Norm"));
        assertEquals(0.25, row.get("parameterUpdateMaxAbs"));
        assertEquals(0.08, row.get("parameterUpdateMeanAbs"));
        assertEquals(0.1, row.get("parameterUpdateRms"));
        assertEquals(4, row.get("parameterUpdateCount"));
        assertEquals(20L, row.get("parameterUpdateValueCount"));
        assertEquals(18L, row.get("parameterUpdateZeroCount"));
        assertEquals(0.9, row.get("parameterUpdateZeroFraction"));
        assertEquals(20L, row.get("parameterUpdateFiniteCount"));
        assertEquals(0L, row.get("parameterUpdateNonFiniteCount"));
        assertEquals(0.0, row.get("parameterUpdateNonFiniteFraction"));
        assertEquals(0.6 / 6.0, (double) row.get("parameterUpdateToParameterL2Ratio"), 1e-12);
        assertEquals(0.25 / 2.5, (double) row.get("parameterUpdateToParameterMaxAbsRatio"), 1e-12);
        assertEquals(0.08 / 0.9, (double) row.get("parameterUpdateToParameterMeanAbsRatio"), 1e-12);
        assertEquals(0.1 / Math.sqrt(36.0 / 20.0), (double) row.get("parameterUpdateToParameterRmsRatio"), 1e-12);
        assertEquals(0.6 / 8.0, (double) row.get("parameterUpdateToGradientL2Ratio"), 1e-12);
        assertEquals(0.25 / 3.0, (double) row.get("parameterUpdateToGradientMaxAbsRatio"), 1e-12);
        assertEquals(0.08 / 1.25, (double) row.get("parameterUpdateToGradientMeanAbsRatio"), 1e-12);
        assertEquals(0.1 / 2.0, (double) row.get("parameterUpdateToGradientRmsRatio"), 1e-12);
        assertEquals(16.0, row.get("mixedPrecisionLossScale"));
    }

    @Test
    void snapshotIsImmutableAndNestedMapsAreCopied() {
        TrainerEpochHistory history = new TrainerEpochHistory();
        Map<String, Object> detail = new LinkedHashMap<>();
        detail.put("tp", 1L);

        history.recordTrain(new TrainerEpochHistory.TrainRecord(
                1,
                1.0,
                0.1,
                1,
                1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                1.0,
                false,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
                Double.NaN,
                false,
                0,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of("loss", 1.0),
                Map.of("detail", detail)));

        List<Map<String, Object>> rows = history.snapshot();
        Map<String, Object> row = rows.get(0);
        @SuppressWarnings("unchecked")
        Map<String, Object> details = (Map<String, Object>) row.get("trainMetricDetails.detail");

        detail.put("fp", 2L);

        assertEquals(Map.of("tp", 1L), details);
        assertThrows(UnsupportedOperationException.class, () -> rows.add(Map.of()));
        assertThrows(UnsupportedOperationException.class, () -> row.put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> details.put("x", 1));
    }

    @Test
    void recordsBestModelMonitorProgressBetweenEpochs() {
        TrainerEpochHistory history = new TrainerEpochHistory();

        history.recordValidation(validationRecordWithMonitor(
                0,
                1.0,
                0.70,
                "validationMetric.accuracy",
                "MAX"));
        history.recordValidation(validationRecordWithMonitor(
                1,
                1.1,
                0.68,
                "validationMetric.accuracy",
                "MAX"));
        history.recordValidation(validationRecordWithMonitor(
                2,
                0.9,
                0.74,
                "validationMetric.accuracy",
                "MAX"));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(3, rows.size());
        Map<String, Object> firstEpoch = rows.get(0);
        assertEquals(Boolean.TRUE, firstEpoch.get("bestModelMonitorProgressAvailable"));
        assertEquals(0.70, firstEpoch.get("bestModelMonitorBestValue"));
        assertEquals(0, firstEpoch.get("bestModelMonitorBestEpoch"));
        assertEquals(0.0, firstEpoch.get("bestModelMonitorDistanceFromBest"));
        assertEquals(0, firstEpoch.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.TRUE, firstEpoch.get("bestModelMonitorIsBest"));

        Map<String, Object> secondEpoch = rows.get(1);
        assertEquals(0.70, secondEpoch.get("bestModelMonitorBestValue"));
        assertEquals(0, secondEpoch.get("bestModelMonitorBestEpoch"));
        assertEquals(0.02, (double) secondEpoch.get("bestModelMonitorDistanceFromBest"), 1e-12);
        assertEquals(1, secondEpoch.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.FALSE, secondEpoch.get("bestModelMonitorIsBest"));

        Map<String, Object> thirdEpoch = rows.get(2);
        assertEquals(0.74, thirdEpoch.get("bestModelMonitorBestValue"));
        assertEquals(2, thirdEpoch.get("bestModelMonitorBestEpoch"));
        assertEquals(0.0, thirdEpoch.get("bestModelMonitorDistanceFromBest"));
        assertEquals(0, thirdEpoch.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.TRUE, thirdEpoch.get("bestModelMonitorIsBest"));
    }

    @Test
    void recordsLossTrendBetweenEpochs() {
        TrainerEpochHistory history = new TrainerEpochHistory();

        history.recordTrain(trainRecord(0, 5.0));
        history.recordValidation(validationRecord(0, 4.0));
        history.recordTrain(trainRecord(1, 3.0));
        history.recordValidation(validationRecord(1, 4.5));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(2, rows.size());
        Map<String, Object> firstEpoch = rows.get(0);
        assertFalse(firstEpoch.containsKey("trainLossDelta"));
        assertFalse(firstEpoch.containsKey("validationLossDelta"));
        assertEquals(5.0, firstEpoch.get("trainLossBest"));
        assertEquals(0, firstEpoch.get("trainLossBestEpoch"));
        assertEquals(0, firstEpoch.get("trainLossNonImprovingStreak"));
        assertEquals(Boolean.TRUE, firstEpoch.get("trainLossBestAtEpoch"));
        assertEquals(4.0, firstEpoch.get("validationLossBest"));
        assertEquals(0, firstEpoch.get("validationLossBestEpoch"));
        assertEquals(0, firstEpoch.get("validationLossNonImprovingStreak"));
        assertEquals(Boolean.TRUE, firstEpoch.get("validationLossBestAtEpoch"));
        assertEquals(Boolean.TRUE, firstEpoch.get("validationLossProgressAvailable"));
        assertEquals(0.0, firstEpoch.get("validationLossDeltaFromBest"));
        assertEquals(1.0, firstEpoch.get("validationLossRatioToBest"));
        assertEquals(0, firstEpoch.get("epochsSinceBestValidation"));
        assertEquals(Boolean.TRUE, firstEpoch.get("validationLossIsBest"));
        assertFalse(firstEpoch.containsKey("generalizationGapDelta"));

        Map<String, Object> secondEpoch = rows.get(1);
        assertEquals(-2.0, secondEpoch.get("trainLossDelta"));
        assertEquals(Boolean.TRUE, secondEpoch.get("trainLossImproved"));
        assertEquals(3.0, secondEpoch.get("trainLossBest"));
        assertEquals(1, secondEpoch.get("trainLossBestEpoch"));
        assertEquals(0, secondEpoch.get("trainLossNonImprovingStreak"));
        assertEquals(Boolean.TRUE, secondEpoch.get("trainLossBestAtEpoch"));
        assertEquals(0.5, secondEpoch.get("validationLossDelta"));
        assertEquals(Boolean.FALSE, secondEpoch.get("validationLossImproved"));
        assertEquals(4.0, secondEpoch.get("validationLossBest"));
        assertEquals(0, secondEpoch.get("validationLossBestEpoch"));
        assertEquals(1, secondEpoch.get("validationLossNonImprovingStreak"));
        assertEquals(Boolean.FALSE, secondEpoch.get("validationLossBestAtEpoch"));
        assertEquals(Boolean.TRUE, secondEpoch.get("validationLossProgressAvailable"));
        assertEquals(0.5, secondEpoch.get("validationLossDeltaFromBest"));
        assertEquals(4.5 / 4.0, (double) secondEpoch.get("validationLossRatioToBest"), 1e-12);
        assertEquals(1, secondEpoch.get("epochsSinceBestValidation"));
        assertEquals(Boolean.FALSE, secondEpoch.get("validationLossIsBest"));
        assertEquals(1.5, secondEpoch.get("generalizationGap"));
        assertEquals(2.5, secondEpoch.get("generalizationGapDelta"));
        assertEquals("increasing", secondEpoch.get("generalizationGapTrend"));
        assertEquals(Boolean.TRUE, secondEpoch.get("generalizationGapIncreasing"));
    }

    @Test
    void recordsRollingLossDiagnosticsBetweenEpochs() {
        TrainerEpochHistory history = new TrainerEpochHistory();

        history.recordTrain(trainRecord(0, 5.0));
        history.recordValidation(validationRecord(0, 4.0));
        history.recordTrain(trainRecord(1, 3.0));
        history.recordValidation(validationRecord(1, 4.5));
        history.recordTrain(trainRecord(2, 2.0));
        history.recordValidation(validationRecord(2, 5.0));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(3, rows.size());
        Map<String, Object> secondEpoch = rows.get(1);
        assertEquals(Boolean.FALSE, secondEpoch.get("trainLossSlopeAvailable"));
        assertEquals(Boolean.TRUE, secondEpoch.get("trainLossWindowStatsAvailable"));
        assertEquals(2, secondEpoch.get("trainLossWindowStatsSize"));

        Map<String, Object> thirdEpoch = rows.get(2);
        double trainMean = (5.0 + 3.0 + 2.0) / 3.0;
        double trainStdDev = Math.sqrt((
                Math.pow(5.0 - trainMean, 2)
                        + Math.pow(3.0 - trainMean, 2)
                        + Math.pow(2.0 - trainMean, 2))
                / 3.0);
        double validationMean = (4.0 + 4.5 + 5.0) / 3.0;
        double validationStdDev = Math.sqrt((
                Math.pow(4.0 - validationMean, 2)
                        + Math.pow(4.5 - validationMean, 2)
                        + Math.pow(5.0 - validationMean, 2))
                / 3.0);

        assertEquals(Boolean.TRUE, thirdEpoch.get("trainLossSlopeAvailable"));
        assertEquals(-1.5, (double) thirdEpoch.get("trainLossSlopePerEpoch"), 1e-12);
        assertEquals("improving", thirdEpoch.get("trainLossTrend"));
        assertEquals(3, thirdEpoch.get("trainLossSlopeWindowSize"));
        assertEquals(Boolean.TRUE, thirdEpoch.get("trainLossWindowStatsAvailable"));
        assertEquals(3, thirdEpoch.get("trainLossWindowStatsSize"));
        assertEquals(trainMean, (double) thirdEpoch.get("trainLossWindowMean"), 1e-12);
        assertEquals(trainStdDev, (double) thirdEpoch.get("trainLossWindowStdDev"), 1e-12);
        assertEquals(
                trainStdDev / trainMean,
                (double) thirdEpoch.get("trainLossWindowCoefficientOfVariation"),
                1e-12);
        assertEquals(Boolean.TRUE, thirdEpoch.get("validationLossSlopeAvailable"));
        assertEquals(0.5, (double) thirdEpoch.get("validationLossSlopePerEpoch"), 1e-12);
        assertEquals("regressing", thirdEpoch.get("validationLossTrend"));
        assertEquals(3, thirdEpoch.get("validationLossSlopeWindowSize"));
        assertEquals(Boolean.TRUE, thirdEpoch.get("validationLossWindowStatsAvailable"));
        assertEquals(3, thirdEpoch.get("validationLossWindowStatsSize"));
        assertEquals(validationMean, (double) thirdEpoch.get("validationLossWindowMean"), 1e-12);
        assertEquals(validationStdDev, (double) thirdEpoch.get("validationLossWindowStdDev"), 1e-12);
        assertEquals(1.0, thirdEpoch.get("validationLossDeltaFromBest"));
        assertEquals(5.0 / 4.0, (double) thirdEpoch.get("validationLossRatioToBest"), 1e-12);
        assertEquals(2, thirdEpoch.get("epochsSinceBestValidation"));
        assertEquals(Boolean.FALSE, thirdEpoch.get("validationLossIsBest"));
        assertEquals(
                validationStdDev / validationMean,
                (double) thirdEpoch.get("validationLossWindowCoefficientOfVariation"),
                1e-12);
    }

    @Test
    void replaceWithLoadedRowsRecomputesDerivedLossDiagnostics() {
        TrainerEpochHistory history = new TrainerEpochHistory();
        Map<String, Object> third = new LinkedHashMap<>();
        third.put("epoch", 2);
        third.put("trainLoss", 2.0);
        third.put("validationLoss", 5.0);
        putMonitor(third, 5.0);
        third.put("source", "legacy");
        Map<String, Object> first = new LinkedHashMap<>();
        first.put("epoch", 0);
        first.put("trainLoss", 5.0);
        first.put("validationLoss", 4.0);
        putMonitor(first, 4.0);
        Map<String, Object> second = new LinkedHashMap<>();
        second.put("epoch", 1);
        second.put("trainLoss", 3.0);
        second.put("validationLoss", 4.5);
        putMonitor(second, 4.5);

        history.replaceWith(List.of(third, first, second));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(3, rows.size());
        assertEquals(0, rows.get(0).get("epoch"));
        assertEquals(1, rows.get(1).get("epoch"));
        assertEquals(2, rows.get(2).get("epoch"));
        assertEquals("legacy", rows.get(2).get("source"));

        Map<String, Object> secondEpoch = rows.get(1);
        assertEquals(-2.0, secondEpoch.get("trainLossDelta"));
        assertEquals(Boolean.TRUE, secondEpoch.get("trainLossImproved"));
        assertEquals(0.5, secondEpoch.get("validationLossDelta"));
        assertEquals(Boolean.FALSE, secondEpoch.get("validationLossImproved"));
        assertEquals(0.5, secondEpoch.get("validationLossDeltaFromBest"));
        assertEquals(1, secondEpoch.get("epochsSinceBestValidation"));
        assertEquals(Boolean.FALSE, secondEpoch.get("validationLossIsBest"));
        assertEquals(4.0, secondEpoch.get("bestModelMonitorBestValue"));
        assertEquals(0, secondEpoch.get("bestModelMonitorBestEpoch"));
        assertEquals(0.5, secondEpoch.get("bestModelMonitorDistanceFromBest"));
        assertEquals(1, secondEpoch.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.FALSE, secondEpoch.get("bestModelMonitorIsBest"));
        assertEquals(2.5, secondEpoch.get("generalizationGapDelta"));
        assertEquals("increasing", secondEpoch.get("generalizationGapTrend"));

        Map<String, Object> thirdEpoch = rows.get(2);
        assertEquals(-1.0, thirdEpoch.get("trainLossDelta"));
        assertEquals(2.0, thirdEpoch.get("trainLossBest"));
        assertEquals(2, thirdEpoch.get("trainLossBestEpoch"));
        assertEquals(Boolean.TRUE, thirdEpoch.get("trainLossSlopeAvailable"));
        assertEquals(-1.5, (double) thirdEpoch.get("trainLossSlopePerEpoch"), 1e-12);
        assertEquals("improving", thirdEpoch.get("trainLossTrend"));
        assertEquals(1.5, thirdEpoch.get("generalizationGapDelta"));
        assertEquals("increasing", thirdEpoch.get("generalizationGapTrend"));
        assertEquals(Boolean.TRUE, thirdEpoch.get("validationLossSlopeAvailable"));
        assertEquals(0.5, (double) thirdEpoch.get("validationLossSlopePerEpoch"), 1e-12);
        assertEquals("regressing", thirdEpoch.get("validationLossTrend"));
        assertEquals(1.0, thirdEpoch.get("validationLossDeltaFromBest"));
        assertEquals(5.0 / 4.0, (double) thirdEpoch.get("validationLossRatioToBest"), 1e-12);
        assertEquals(2, thirdEpoch.get("epochsSinceBestValidation"));
        assertEquals(Boolean.FALSE, thirdEpoch.get("validationLossIsBest"));
        assertEquals(4.0, thirdEpoch.get("bestModelMonitorBestValue"));
        assertEquals(0, thirdEpoch.get("bestModelMonitorBestEpoch"));
        assertEquals(1.0, thirdEpoch.get("bestModelMonitorDistanceFromBest"));
        assertEquals(2, thirdEpoch.get("epochsSinceBestModelMonitor"));
        assertEquals(Boolean.FALSE, thirdEpoch.get("bestModelMonitorIsBest"));
    }

    @Test
    void replaceWithLoadedRowsCopiesInputAndValidationCanMerge() {
        TrainerEpochHistory history = new TrainerEpochHistory();
        Map<String, Object> loaded = new LinkedHashMap<>();
        loaded.put("epoch", 2);
        loaded.put("trainLoss", 3.0);

        history.replaceWith(List.of(loaded));
        loaded.put("epoch", 99);
        history.recordValidation(new TrainerEpochHistory.ValidationRecord(
                2,
                2.0,
                0.01,
                5,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of(),
                Map.of(),
                Double.NaN,
                "validation_loss",
                "MIN"));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(1, rows.size());
        assertEquals(2, rows.get(0).get("epoch"));
        assertEquals(3.0, rows.get(0).get("trainLoss"));
        assertEquals(2.0, rows.get(0).get("validationLoss"));
        assertEquals(-1.0, rows.get(0).get("generalizationGap"));
        assertEquals(2.0 / 3.0, (double) rows.get(0).get("validationToTrainLossRatio"), 1e-12);
        assertEquals(Boolean.FALSE, rows.get(0).get("validationLossAboveTrainLoss"));
    }

    private static TrainerEpochHistory.TrainRecord trainRecord(int epoch, double trainLoss) {
        return new TrainerEpochHistory.TrainRecord(
                epoch,
                trainLoss,
                0.1,
                1,
                1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                1.0,
                false,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                0L,
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
                Double.NaN,
                false,
                0,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of(),
                Map.of());
    }

    private static TrainerEpochHistory.ValidationRecord validationRecord(int epoch, double validationLoss) {
        return validationRecordWithMonitor(epoch, validationLoss, Double.NaN, "validation_loss", "MIN");
    }

    private static TrainerEpochHistory.ValidationRecord validationRecordWithMonitor(
            int epoch,
            double validationLoss,
            double monitorValue,
            String monitorLabel,
            String monitorMode) {
        return new TrainerEpochHistory.ValidationRecord(
                epoch,
                validationLoss,
                0.1,
                1,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of(),
                Map.of(),
                monitorValue,
                monitorLabel,
                monitorMode);
    }

    private static void putMonitor(Map<String, Object> row, double value) {
        row.put("bestModelMonitor", "validation_loss");
        row.put("bestModelMonitorMode", "MIN");
        row.put("bestModelMonitorValue", value);
    }
}
