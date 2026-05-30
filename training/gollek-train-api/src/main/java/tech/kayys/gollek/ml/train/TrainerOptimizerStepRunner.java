package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.GradientClipper;
import tech.kayys.gollek.ml.optim.Optimizer;

/**
 * Applies one accumulated optimizer step and returns telemetry for the trainer.
 */
final class TrainerOptimizerStepRunner {
    private final Optimizer optimizer;
    private final GradScaler gradScaler;
    private final TrainerGradientClipConfig gradientClip;
    private final TrainerBatchGuards.FailureRecorder failures;
    private final Runnable batchSchedulerStep;
    private final boolean parameterUpdateDiagnostics;

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            double gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep) {
        this(optimizer, gradScaler, TrainerGradientClipConfig.norm(gradientClip), failures, batchSchedulerStep, false);
    }

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            double gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep,
            boolean parameterUpdateDiagnostics) {
        this(
                optimizer,
                gradScaler,
                TrainerGradientClipConfig.norm(gradientClip),
                failures,
                batchSchedulerStep,
                parameterUpdateDiagnostics);
    }

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            TrainerGradientClipConfig gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep,
            boolean parameterUpdateDiagnostics) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
        this.gradScaler = gradScaler;
        this.gradientClip = Objects.requireNonNull(gradientClip, "gradientClip must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.batchSchedulerStep = Objects.requireNonNull(batchSchedulerStep, "batchSchedulerStep must not be null");
        this.parameterUpdateDiagnostics = parameterUpdateDiagnostics;
    }

    StepResult step(int pendingGradientAccumulationBatches) {
        if (pendingGradientAccumulationBatches <= 0) {
            return StepResult.noPendingGradients();
        }
        TrainerTensorDiagnostics.scaleGradients(
                optimizer.parameters(),
                1.0f / pendingGradientAccumulationBatches);
        if (gradScaler != null && gradScaler.unscaleAndCheck(optimizer)) {
            double scaleBeforeUpdate = gradScaler.getScale();
            gradScaler.update();
            optimizer.zeroGrad();
            return StepResult.overflowSkipped(
                    scaleBeforeUpdate,
                    gradScaler.getScale());
        }

        TensorDiagnostics gradientBeforeClip = TrainerTensorDiagnostics.gradients(optimizer.parameters());
        TrainerTensorDiagnostics.requireFinite(gradientBeforeClip, "train", "gradient", true, failures);
        double gradientClipScale = 1.0;
        boolean gradientClipped = false;
        if (gradientClip.normEnabled()) {
            GradientClipper.ClipResult clipResult =
                    GradientClipper.clipByNormDetailed(optimizer.parameters(), (float) gradientClip.normThreshold());
            gradientClipScale = clipResult.scale();
            gradientClipped = clipResult.clipped();
        }
        if (gradientClip.valueEnabled()) {
            TensorDiagnostics afterNormClip = TrainerTensorDiagnostics.gradients(optimizer.parameters());
            if (afterNormClip.maxAbs() > gradientClip.valueThreshold()) {
                GradientClipper.clipByValue(
                        optimizer.parameters(),
                        (float) -gradientClip.valueThreshold(),
                        (float) gradientClip.valueThreshold());
                gradientClipped = true;
            }
        }
        TensorDiagnostics gradientAfterClip = TrainerTensorDiagnostics.gradients(optimizer.parameters());
        TrainerTensorDiagnostics.requireFinite(gradientAfterClip, "train", "clipped-gradient", true, failures);

        double lossScaleBeforeUpdate = gradScaler == null ? Double.NaN : gradScaler.getScale();
        List<float[]> parametersBeforeStep = parameterUpdateDiagnostics
                ? TrainerTensorDiagnostics.snapshotParameters(optimizer.parameters())
                : List.of();
        if (gradScaler == null) {
            optimizer.step();
        } else {
            gradScaler.step(optimizer);
        }
        TensorDiagnostics parameterUpdates = parameterUpdateDiagnostics
                ? TrainerTensorDiagnostics.parameterUpdates(optimizer.parameters(), parametersBeforeStep)
                : TensorDiagnostics.empty();
        TensorDiagnostics parametersAfterStep = TrainerTensorDiagnostics.parameters(optimizer.parameters());
        TrainerTensorDiagnostics.requireFinite(parametersAfterStep, "train", "parameter", false, failures);

        batchSchedulerStep.run();
        double lossScale = Double.NaN;
        boolean overflowDetected = false;
        if (gradScaler != null) {
            gradScaler.update();
            lossScale = gradScaler.getScale();
            overflowDetected = gradScaler.overflowDetected();
        }
        optimizer.zeroGrad();
        return StepResult.optimizerStepped(
                gradScaler != null,
                lossScaleBeforeUpdate,
                lossScale,
                overflowDetected,
                gradientBeforeClip,
                gradientAfterClip,
                gradientClipScale,
                gradientClipped,
                parameterUpdateDiagnostics,
                parameterUpdates,
                parametersAfterStep);
    }

    record StepResult(
            boolean attempted,
            boolean pendingGradientsCleared,
            boolean optimizerStepApplied,
            boolean overflowSkipped,
            boolean mixedPrecisionUsed,
            double lossScaleBeforeUpdate,
            double lossScale,
            boolean overflowDetected,
            TensorDiagnostics gradientBeforeClip,
            TensorDiagnostics gradientAfterClip,
            double gradientClipScale,
            boolean gradientClipped,
            boolean parameterUpdateDiagnosticsEnabled,
            TensorDiagnostics parameterUpdates,
            TensorDiagnostics parametersAfterStep) {
        static StepResult noPendingGradients() {
            return new StepResult(
                    false,
                    false,
                    false,
                    false,
                    false,
                    Double.NaN,
                    Double.NaN,
                    false,
                    null,
                    null,
                    Double.NaN,
                    false,
                    false,
                    null,
                    null);
        }

        static StepResult overflowSkipped(double lossScaleBeforeUpdate, double lossScale) {
            return new StepResult(
                    true,
                    true,
                    false,
                    true,
                    true,
                    lossScaleBeforeUpdate,
                    lossScale,
                    true,
                    null,
                    null,
                    Double.NaN,
                    false,
                    false,
                    null,
                    null);
        }

        static StepResult optimizerStepped(
                boolean mixedPrecisionUsed,
                double lossScaleBeforeUpdate,
                double lossScale,
                boolean overflowDetected,
                TensorDiagnostics gradientBeforeClip,
                TensorDiagnostics gradientAfterClip,
                double gradientClipScale,
                boolean gradientClipped,
                boolean parameterUpdateDiagnosticsEnabled,
                TensorDiagnostics parameterUpdates,
                TensorDiagnostics parametersAfterStep) {
            return new StepResult(
                    true,
                    true,
                    true,
                    false,
                    mixedPrecisionUsed,
                    mixedPrecisionUsed ? lossScaleBeforeUpdate : Double.NaN,
                    lossScale,
                    overflowDetected,
                    gradientBeforeClip,
                    gradientAfterClip,
                    gradientClipScale,
                    gradientClipped,
                    parameterUpdateDiagnosticsEnabled,
                    parameterUpdates,
                    parametersAfterStep);
        }
    }
}
