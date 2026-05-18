package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Builds epoch history records from trainer telemetry snapshots.
 */
final class TrainerEpochHistoryRecordFactory {
    private TrainerEpochHistoryRecordFactory() {
    }

    static TrainerEpochHistory.TrainRecord train(
            int epoch,
            double trainLoss,
            double learningRate,
            int optimizerStepCount,
            int schedulerStepCount,
            TrainerOptimizationMetadata.GradientDiagnostics gradients,
            TrainerOptimizationMetadata.ParameterDiagnostics parameters,
            MixedPrecisionDiagnostics mixedPrecision,
            ThroughputSnapshot trainThroughput,
            Map<String, Double> trainMetrics,
            Map<String, Object> trainMetricDetails) {
        return new TrainerEpochHistory.TrainRecord(
                epoch,
                trainLoss,
                learningRate,
                optimizerStepCount,
                schedulerStepCount,
                gradients.l2NormBeforeClip(),
                gradients.l2Norm(),
                gradients.maxAbsBeforeClip(),
                gradients.maxAbs(),
                gradients.parameterCount(),
                gradients.valueCount(),
                gradients.clipped(),
                parameters.l2Norm(),
                parameters.maxAbs(),
                parameters.count(),
                parameters.valueCount(),
                mixedPrecision.enabled(),
                mixedPrecision.lossScale(),
                mixedPrecision.overflowDetected(),
                mixedPrecision.overflowSkipCount(),
                trainThroughput,
                trainMetrics,
                trainMetricDetails);
    }

    static TrainerEpochHistory.ValidationRecord validation(
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
        return new TrainerEpochHistory.ValidationRecord(
                epoch,
                validationLoss,
                learningRate,
                schedulerStepCount,
                validationThroughput,
                validationMetrics,
                validationMetricDetails,
                bestModelMonitorValue,
                bestModelMonitorLabel,
                bestModelMonitorMode);
    }

    record MixedPrecisionDiagnostics(
            boolean enabled,
            double lossScale,
            boolean overflowDetected,
            int overflowSkipCount) {
    }
}
