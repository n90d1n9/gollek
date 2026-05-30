package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.Optimizer;

/**
 * Owns gradient accumulation, optimizer stepping, and optimizer telemetry.
 */
final class TrainerOptimizationRuntime {
    private final Optimizer optimizer;
    private final TrainerOptimizerStepRunner stepRunner;
    private final int gradientAccumulationSteps;
    private final boolean mixedPrecision;
    private final GradScaler gradScaler;
    private final BooleanSupplier nonFiniteDetected;
    private int pendingGradientAccumulationBatches;
    private int optimizerStepCount;
    private int mixedPrecisionOverflowSkipCount;
    private boolean latestMixedPrecisionOverflowDetected;
    private double latestMixedPrecisionLossScale = Double.NaN;
    private TrainerOptimizationMetadata.GradientDiagnostics latestGradientDiagnostics =
            new TrainerOptimizationMetadata.GradientDiagnostics(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0L, 0L, 0.0,
                    0L, 0L, 0L, 0L, 0L, 0.0, 1.0, false);
    private TrainerOptimizationMetadata.ParameterDiagnostics latestParameterDiagnostics =
            new TrainerOptimizationMetadata.ParameterDiagnostics(
                    0.0, 0.0, 0.0, 0.0, 0, 0L, 0L, 0.0, 0L, 0L, 0L, 0L, 0L, 0.0);
    private TrainerOptimizationMetadata.UpdateDiagnostics latestParameterUpdateDiagnostics =
            TrainerOptimizationMetadata.UpdateDiagnostics.disabled();

    TrainerOptimizationRuntime(
            Optimizer optimizer,
            TrainerOptimizerStepRunner stepRunner,
            int gradientAccumulationSteps,
            boolean mixedPrecision,
            GradScaler gradScaler,
            BooleanSupplier nonFiniteDetected) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
        this.stepRunner = Objects.requireNonNull(stepRunner, "stepRunner must not be null");
        this.gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
        this.mixedPrecision = mixedPrecision;
        this.gradScaler = gradScaler;
        this.nonFiniteDetected = Objects.requireNonNull(nonFiniteDetected, "nonFiniteDetected must not be null");
        if (gradScaler != null) {
            this.latestMixedPrecisionLossScale = gradScaler.getScale();
        }
    }

    boolean shouldZeroGradBeforeBackward() {
        return pendingGradientAccumulationBatches == 0;
    }

    void afterBackwardBatch() {
        pendingGradientAccumulationBatches++;
        if (pendingGradientAccumulationBatches >= gradientAccumulationSteps) {
            applyOptimizerStep();
        }
    }

    boolean hasPendingGradients() {
        return pendingGradientAccumulationBatches > 0;
    }

    void flush() {
        if (hasPendingGradients()) {
            applyOptimizerStep();
        }
    }

    void discardPendingGradients() {
        optimizer.zeroGrad();
        pendingGradientAccumulationBatches = 0;
    }

    void restoreOptimizerStepCount(int optimizerStepCount) {
        this.optimizerStepCount = Math.max(0, optimizerStepCount);
    }

    void restoreMixedPrecisionState(int overflowSkipCount, double lossScale, boolean overflowDetected) {
        this.mixedPrecisionOverflowSkipCount = Math.max(0, overflowSkipCount);
        this.latestMixedPrecisionLossScale = lossScale;
        this.latestMixedPrecisionOverflowDetected = overflowDetected;
    }

    int gradientAccumulationSteps() {
        return gradientAccumulationSteps;
    }

    int pendingGradientAccumulationBatches() {
        return pendingGradientAccumulationBatches;
    }

    int optimizerStepCount() {
        return optimizerStepCount;
    }

    int mixedPrecisionOverflowSkipCount() {
        return mixedPrecisionOverflowSkipCount;
    }

    double latestMixedPrecisionLossScale() {
        return latestMixedPrecisionLossScale;
    }

    boolean latestMixedPrecisionOverflowDetected() {
        return latestMixedPrecisionOverflowDetected;
    }

    TrainerOptimizationMetadata.GradientDiagnostics latestGradientDiagnostics() {
        return latestGradientDiagnostics;
    }

    TrainerOptimizationMetadata.ParameterDiagnostics latestParameterDiagnostics() {
        return latestParameterDiagnostics;
    }

    TrainerOptimizationMetadata.UpdateDiagnostics latestParameterUpdateDiagnostics() {
        return latestParameterUpdateDiagnostics;
    }

    TrainerEpochHistoryRecordFactory.MixedPrecisionDiagnostics latestMixedPrecisionDiagnostics() {
        return new TrainerEpochHistoryRecordFactory.MixedPrecisionDiagnostics(
                mixedPrecision,
                latestMixedPrecisionLossScale,
                latestMixedPrecisionOverflowDetected,
                mixedPrecisionOverflowSkipCount);
    }

    Map<String, Object> gradScalerStateSnapshot() {
        if (gradScaler == null || !gradScaler.supportsStateDict()) {
            return Map.of();
        }
        return TrainerMetadataSupport.stateSnapshot(gradScaler.stateDict());
    }

    private void applyOptimizerStep() {
        if (!hasPendingGradients()) {
            return;
        }
        try {
            TrainerOptimizerStepRunner.StepResult result =
                    stepRunner.step(pendingGradientAccumulationBatches);
            if (!result.attempted()) {
                return;
            }
            if (result.overflowSkipped()) {
                latestMixedPrecisionOverflowDetected = result.overflowDetected();
                mixedPrecisionOverflowSkipCount++;
                latestMixedPrecisionLossScale = result.lossScale();
                if (result.pendingGradientsCleared()) {
                    pendingGradientAccumulationBatches = 0;
                }
                return;
            }
            if (result.optimizerStepApplied()) {
                optimizerStepCount++;
                updateTensorDiagnostics(
                        result.gradientBeforeClip(),
                        result.gradientAfterClip(),
                        result.gradientClipScale(),
                        result.gradientClipped(),
                        result.parameterUpdateDiagnosticsEnabled(),
                        result.parameterUpdates(),
                        result.parametersAfterStep());
            }
            if (result.mixedPrecisionUsed()) {
                latestMixedPrecisionLossScale = result.lossScale();
                latestMixedPrecisionOverflowDetected = result.overflowDetected();
            }
            if (result.pendingGradientsCleared()) {
                pendingGradientAccumulationBatches = 0;
            }
        } catch (RuntimeException error) {
            if (nonFiniteDetected.getAsBoolean()) {
                discardPendingGradients();
            }
            throw error;
        }
    }

    private void updateTensorDiagnostics(
            TensorDiagnostics gradientBeforeClip,
            TensorDiagnostics gradientAfterClip,
            double gradientClipScale,
            boolean gradientClipped,
            boolean parameterUpdateDiagnosticsEnabled,
            TensorDiagnostics parameterUpdates,
            TensorDiagnostics parametersAfterStep) {
        latestGradientDiagnostics = new TrainerOptimizationMetadata.GradientDiagnostics(
                gradientBeforeClip.l2Norm(),
                gradientAfterClip.l2Norm(),
                gradientBeforeClip.maxAbs(),
                gradientAfterClip.maxAbs(),
                gradientBeforeClip.meanAbs(),
                gradientAfterClip.meanAbs(),
                gradientBeforeClip.rms(),
                gradientAfterClip.rms(),
                gradientAfterClip.tensorCount(),
                gradientAfterClip.valueCount(),
                gradientAfterClip.zeroCount(),
                gradientAfterClip.zeroFraction(),
                gradientAfterClip.finiteCount(),
                gradientAfterClip.nonFiniteCount(),
                gradientAfterClip.nanCount(),
                gradientAfterClip.positiveInfinityCount(),
                gradientAfterClip.negativeInfinityCount(),
                gradientAfterClip.nonFiniteFraction(),
                gradientClipScale,
                gradientClipped);
        latestParameterDiagnostics = new TrainerOptimizationMetadata.ParameterDiagnostics(
                parametersAfterStep.l2Norm(),
                parametersAfterStep.maxAbs(),
                parametersAfterStep.meanAbs(),
                parametersAfterStep.rms(),
                parametersAfterStep.tensorCount(),
                parametersAfterStep.valueCount(),
                parametersAfterStep.zeroCount(),
                parametersAfterStep.zeroFraction(),
                parametersAfterStep.finiteCount(),
                parametersAfterStep.nonFiniteCount(),
                parametersAfterStep.nanCount(),
                parametersAfterStep.positiveInfinityCount(),
                parametersAfterStep.negativeInfinityCount(),
                parametersAfterStep.nonFiniteFraction());
        latestParameterUpdateDiagnostics = parameterUpdateDiagnosticsEnabled
                ? TrainerOptimizationMetadata.UpdateDiagnostics.enabled(parameterUpdates)
                : TrainerOptimizationMetadata.UpdateDiagnostics.disabled();
    }
}
