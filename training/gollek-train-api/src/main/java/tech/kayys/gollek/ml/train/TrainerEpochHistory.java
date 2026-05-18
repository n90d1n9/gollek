package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
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
        row.put("gradientL2NormBeforeClip", record.gradientL2NormBeforeClip());
        row.put("gradientL2Norm", record.gradientL2Norm());
        row.put("gradientMaxAbsBeforeClip", record.gradientMaxAbsBeforeClip());
        row.put("gradientMaxAbs", record.gradientMaxAbs());
        row.put("gradientParameterCount", record.gradientParameterCount());
        row.put("gradientValueCount", record.gradientValueCount());
        row.put("gradientClipped", record.gradientClipped());
        row.put("parameterL2Norm", record.parameterL2Norm());
        row.put("parameterMaxAbs", record.parameterMaxAbs());
        row.put("parameterCount", record.parameterCount());
        row.put("parameterValueCount", record.parameterValueCount());
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
            int gradientParameterCount,
            long gradientValueCount,
            boolean gradientClipped,
            double parameterL2Norm,
            double parameterMaxAbs,
            int parameterCount,
            long parameterValueCount,
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
