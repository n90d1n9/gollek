package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Owns per-epoch training history rows and snapshot isolation.
 */
final class TrainerEpochHistory {
    private final List<Map<String, Object>> rows = new ArrayList<>();

    void replaceWith(List<Map<String, Object>> loadedRows) {
        rows.clear();
        if (loadedRows == null || loadedRows.isEmpty()) {
            return;
        }
        for (Map<String, Object> row : loadedRows) {
            rows.add(new LinkedHashMap<>(TrainingHistoryCsv.copyRow(row)));
        }
        rows.sort(Comparator.comparingInt(TrainerEpochHistoryDiagnostics::epochOf));
        TrainerEpochHistoryDiagnostics.recompute(rows);
    }

    int size() {
        return rows.size();
    }

    List<Map<String, Object>> snapshot() {
        if (rows.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> snapshot = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            snapshot.add(Collections.unmodifiableMap(TrainingHistoryCsv.copyRow(row)));
        }
        return List.copyOf(snapshot);
    }

    void recordTrain(TrainRecord record) {
        Map<String, Object> row = row(record.epoch());
        row.put("epoch", record.epoch());
        row.put("trainLoss", record.trainLoss());
        row.put("learningRate", record.learningRate());
        row.put("optimizerStepCount", record.optimizerStepCount());
        row.put("schedulerStepCount", record.schedulerStepCount());
        TrainerEpochHistoryDiagnostics.putTrain(row, rows, record.epoch());
        row.put("gradientL2NormBeforeClip", record.gradientL2NormBeforeClip());
        row.put("gradientL2Norm", record.gradientL2Norm());
        row.put("gradientMaxAbsBeforeClip", record.gradientMaxAbsBeforeClip());
        row.put("gradientMaxAbs", record.gradientMaxAbs());
        row.put("gradientMeanAbsBeforeClip", record.gradientMeanAbsBeforeClip());
        row.put("gradientMeanAbs", record.gradientMeanAbs());
        row.put("gradientRmsBeforeClip", record.gradientRmsBeforeClip());
        row.put("gradientRms", record.gradientRms());
        row.put("gradientParameterCount", record.gradientParameterCount());
        row.put("gradientValueCount", record.gradientValueCount());
        row.put("gradientZeroCount", record.gradientZeroCount());
        row.put("gradientZeroFraction", record.gradientZeroFraction());
        row.put("gradientFiniteCount", record.gradientFiniteCount());
        row.put("gradientNonFiniteCount", record.gradientNonFiniteCount());
        row.put("gradientNanCount", record.gradientNanCount());
        row.put("gradientPositiveInfinityCount", record.gradientPositiveInfinityCount());
        row.put("gradientNegativeInfinityCount", record.gradientNegativeInfinityCount());
        row.put("gradientNonFiniteFraction", record.gradientNonFiniteFraction());
        row.put("gradientClipScale", record.gradientClipScale());
        row.put("gradientClipped", record.gradientClipped());
        row.put("parameterL2Norm", record.parameterL2Norm());
        row.put("parameterMaxAbs", record.parameterMaxAbs());
        row.put("parameterMeanAbs", record.parameterMeanAbs());
        row.put("parameterRms", record.parameterRms());
        row.put("parameterCount", record.parameterCount());
        row.put("parameterValueCount", record.parameterValueCount());
        row.put("parameterZeroCount", record.parameterZeroCount());
        row.put("parameterZeroFraction", record.parameterZeroFraction());
        row.put("parameterFiniteCount", record.parameterFiniteCount());
        row.put("parameterNonFiniteCount", record.parameterNonFiniteCount());
        row.put("parameterNanCount", record.parameterNanCount());
        row.put("parameterPositiveInfinityCount", record.parameterPositiveInfinityCount());
        row.put("parameterNegativeInfinityCount", record.parameterNegativeInfinityCount());
        row.put("parameterNonFiniteFraction", record.parameterNonFiniteFraction());
        row.put("gradientToParameterL2Ratio", record.gradientToParameterL2Ratio());
        row.put("gradientToParameterMaxAbsRatio", record.gradientToParameterMaxAbsRatio());
        row.put("gradientToParameterMeanAbsRatio", record.gradientToParameterMeanAbsRatio());
        row.put("gradientToParameterRmsRatio", record.gradientToParameterRmsRatio());
        row.put("parameterUpdateDiagnosticsEnabled", record.parameterUpdateDiagnosticsEnabled());
        row.put("parameterUpdateL2Norm", record.parameterUpdateL2Norm());
        row.put("parameterUpdateMaxAbs", record.parameterUpdateMaxAbs());
        row.put("parameterUpdateMeanAbs", record.parameterUpdateMeanAbs());
        row.put("parameterUpdateRms", record.parameterUpdateRms());
        row.put("parameterUpdateCount", record.parameterUpdateCount());
        row.put("parameterUpdateValueCount", record.parameterUpdateValueCount());
        row.put("parameterUpdateZeroCount", record.parameterUpdateZeroCount());
        row.put("parameterUpdateZeroFraction", record.parameterUpdateZeroFraction());
        row.put("parameterUpdateFiniteCount", record.parameterUpdateFiniteCount());
        row.put("parameterUpdateNonFiniteCount", record.parameterUpdateNonFiniteCount());
        row.put("parameterUpdateNanCount", record.parameterUpdateNanCount());
        row.put("parameterUpdatePositiveInfinityCount", record.parameterUpdatePositiveInfinityCount());
        row.put("parameterUpdateNegativeInfinityCount", record.parameterUpdateNegativeInfinityCount());
        row.put("parameterUpdateNonFiniteFraction", record.parameterUpdateNonFiniteFraction());
        row.put("parameterUpdateToParameterL2Ratio", record.parameterUpdateToParameterL2Ratio());
        row.put("parameterUpdateToParameterMaxAbsRatio", record.parameterUpdateToParameterMaxAbsRatio());
        row.put("parameterUpdateToParameterMeanAbsRatio", record.parameterUpdateToParameterMeanAbsRatio());
        row.put("parameterUpdateToParameterRmsRatio", record.parameterUpdateToParameterRmsRatio());
        row.put("parameterUpdateToGradientL2Ratio", record.parameterUpdateToGradientL2Ratio());
        row.put("parameterUpdateToGradientMaxAbsRatio", record.parameterUpdateToGradientMaxAbsRatio());
        row.put("parameterUpdateToGradientMeanAbsRatio", record.parameterUpdateToGradientMeanAbsRatio());
        row.put("parameterUpdateToGradientRmsRatio", record.parameterUpdateToGradientRmsRatio());
        row.put("mixedPrecisionEnabled", record.mixedPrecision());
        if (record.mixedPrecision()) {
            row.put("mixedPrecisionLossScale", record.mixedPrecisionLossScale());
            row.put("mixedPrecisionOverflowDetected", record.mixedPrecisionOverflowDetected());
            row.put("mixedPrecisionOverflowSkipCount", record.mixedPrecisionOverflowSkipCount());
        }
        TrainerThroughputStats.putPhaseMetadata(row, "train", record.trainThroughput());
        row.put("trainMetrics", record.trainMetrics());
        row.put("trainMetricDetails", record.trainMetricDetails());
        flatten(row, "trainMetric.", record.trainMetrics());
        flatten(row, "trainMetricDetails.", record.trainMetricDetails());
        addIfNew(row);
    }

    void recordValidation(ValidationRecord record) {
        Map<String, Object> row = row(record.epoch());
        row.put("epoch", record.epoch());
        row.put("validationLoss", record.validationLoss());
        row.put("learningRate", record.learningRate());
        row.put("schedulerStepCount", record.schedulerStepCount());
        TrainerThroughputStats.putPhaseMetadata(row, "validation", record.validationThroughput());
        row.put("validationMetrics", record.validationMetrics());
        row.put("validationMetricDetails", record.validationMetricDetails());
        flatten(row, "validationMetric.", record.validationMetrics());
        flatten(row, "validationMetricDetails.", record.validationMetricDetails());
        if (Double.isFinite(record.bestModelMonitorValue())) {
            row.put("bestModelMonitor", record.bestModelMonitorLabel());
            row.put("bestModelMonitorMode", record.bestModelMonitorMode());
            row.put("bestModelMonitorValue", record.bestModelMonitorValue());
        }
        TrainerEpochHistoryDiagnostics.putValidation(row, rows, record.epoch());
        addIfNew(row);
    }

    private Map<String, Object> row(int epoch) {
        for (int i = rows.size() - 1; i >= 0; i--) {
            Map<String, Object> row = rows.get(i);
            Object rowEpoch = row.get("epoch");
            if (rowEpoch instanceof Number number && number.intValue() == epoch) {
                return row;
            }
        }
        return new LinkedHashMap<>();
    }

    private void addIfNew(Map<String, Object> row) {
        if (!rows.contains(row)) {
            rows.add(row);
        }
    }

    private static void flatten(Map<String, Object> target, String prefix, Map<String, ?> source) {
        for (Map.Entry<String, ?> entry : source.entrySet()) {
            target.put(prefix + entry.getKey(), entry.getValue());
        }
    }

    record TrainRecord(
            int epoch,
            double trainLoss,
            double learningRate,
            int optimizerStepCount,
            int schedulerStepCount,
            double gradientL2NormBeforeClip,
            double gradientL2Norm,
            double gradientMaxAbsBeforeClip,
            double gradientMaxAbs,
            double gradientMeanAbsBeforeClip,
            double gradientMeanAbs,
            double gradientRmsBeforeClip,
            double gradientRms,
            int gradientParameterCount,
            long gradientValueCount,
            long gradientZeroCount,
            double gradientZeroFraction,
            long gradientFiniteCount,
            long gradientNonFiniteCount,
            long gradientNanCount,
            long gradientPositiveInfinityCount,
            long gradientNegativeInfinityCount,
            double gradientNonFiniteFraction,
            double gradientClipScale,
            boolean gradientClipped,
            double parameterL2Norm,
            double parameterMaxAbs,
            double parameterMeanAbs,
            double parameterRms,
            int parameterCount,
            long parameterValueCount,
            long parameterZeroCount,
            double parameterZeroFraction,
            long parameterFiniteCount,
            long parameterNonFiniteCount,
            long parameterNanCount,
            long parameterPositiveInfinityCount,
            long parameterNegativeInfinityCount,
            double parameterNonFiniteFraction,
            double gradientToParameterL2Ratio,
            double gradientToParameterMaxAbsRatio,
            double gradientToParameterMeanAbsRatio,
            double gradientToParameterRmsRatio,
            boolean parameterUpdateDiagnosticsEnabled,
            double parameterUpdateL2Norm,
            double parameterUpdateMaxAbs,
            double parameterUpdateMeanAbs,
            double parameterUpdateRms,
            int parameterUpdateCount,
            long parameterUpdateValueCount,
            long parameterUpdateZeroCount,
            double parameterUpdateZeroFraction,
            long parameterUpdateFiniteCount,
            long parameterUpdateNonFiniteCount,
            long parameterUpdateNanCount,
            long parameterUpdatePositiveInfinityCount,
            long parameterUpdateNegativeInfinityCount,
            double parameterUpdateNonFiniteFraction,
            double parameterUpdateToParameterL2Ratio,
            double parameterUpdateToParameterMaxAbsRatio,
            double parameterUpdateToParameterMeanAbsRatio,
            double parameterUpdateToParameterRmsRatio,
            double parameterUpdateToGradientL2Ratio,
            double parameterUpdateToGradientMaxAbsRatio,
            double parameterUpdateToGradientMeanAbsRatio,
            double parameterUpdateToGradientRmsRatio,
            boolean mixedPrecision,
            double mixedPrecisionLossScale,
            boolean mixedPrecisionOverflowDetected,
            int mixedPrecisionOverflowSkipCount,
            ThroughputSnapshot trainThroughput,
            Map<String, Double> trainMetrics,
            Map<String, Object> trainMetricDetails) {
    }

    record ValidationRecord(
            int epoch,
            double validationLoss,
            double learningRate,
            int schedulerStepCount,
            ThroughputSnapshot validationThroughput,
            Map<String, Double> validationMetrics,
            Map<String, Object> validationMetricDetails,
            double bestModelMonitorValue,
            String bestModelMonitorLabel,
            String bestModelMonitorMode) {
    }
}
