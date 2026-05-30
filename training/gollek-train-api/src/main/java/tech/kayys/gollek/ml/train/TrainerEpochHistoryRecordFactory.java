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
            TrainerOptimizationMetadata.UpdateDiagnostics updates,
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
                gradients.meanAbsBeforeClip(),
                gradients.meanAbs(),
                gradients.rmsBeforeClip(),
                gradients.rms(),
                gradients.parameterCount(),
                gradients.valueCount(),
                gradients.zeroCount(),
                gradients.zeroFraction(),
                gradients.finiteCount(),
                gradients.nonFiniteCount(),
                gradients.nanCount(),
                gradients.positiveInfinityCount(),
                gradients.negativeInfinityCount(),
                gradients.nonFiniteFraction(),
                gradients.clipScale(),
                gradients.clipped(),
                parameters.l2Norm(),
                parameters.maxAbs(),
                parameters.meanAbs(),
                parameters.rms(),
                parameters.count(),
                parameters.valueCount(),
                parameters.zeroCount(),
                parameters.zeroFraction(),
                parameters.finiteCount(),
                parameters.nonFiniteCount(),
                parameters.nanCount(),
                parameters.positiveInfinityCount(),
                parameters.negativeInfinityCount(),
                parameters.nonFiniteFraction(),
                TrainerOptimizationMetadata.ratio(gradients.l2Norm(), parameters.l2Norm()),
                TrainerOptimizationMetadata.ratio(gradients.maxAbs(), parameters.maxAbs()),
                TrainerOptimizationMetadata.ratio(gradients.meanAbs(), parameters.meanAbs()),
                TrainerOptimizationMetadata.ratio(gradients.rms(), parameters.rms()),
                updates.enabled(),
                updates.l2Norm(),
                updates.maxAbs(),
                updates.meanAbs(),
                updates.rms(),
                updates.count(),
                updates.valueCount(),
                updates.zeroCount(),
                updates.zeroFraction(),
                updates.finiteCount(),
                updates.nonFiniteCount(),
                updates.nanCount(),
                updates.positiveInfinityCount(),
                updates.negativeInfinityCount(),
                updates.nonFiniteFraction(),
                TrainerOptimizationMetadata.ratio(updates.l2Norm(), parameters.l2Norm()),
                TrainerOptimizationMetadata.ratio(updates.maxAbs(), parameters.maxAbs()),
                TrainerOptimizationMetadata.ratio(updates.meanAbs(), parameters.meanAbs()),
                TrainerOptimizationMetadata.ratio(updates.rms(), parameters.rms()),
                TrainerOptimizationMetadata.ratio(updates.l2Norm(), gradients.l2Norm()),
                TrainerOptimizationMetadata.ratio(updates.maxAbs(), gradients.maxAbs()),
                TrainerOptimizationMetadata.ratio(updates.meanAbs(), gradients.meanAbs()),
                TrainerOptimizationMetadata.ratio(updates.rms(), gradients.rms()),
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
