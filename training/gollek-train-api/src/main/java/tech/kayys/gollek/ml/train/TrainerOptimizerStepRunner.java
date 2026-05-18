package tech.kayys.gollek.ml.train;

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
    private final double gradientClip;
    private final TrainerBatchGuards.FailureRecorder failures;
    private final Runnable batchSchedulerStep;

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            double gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
        this.gradScaler = gradScaler;
        this.gradientClip = Math.max(0.0, gradientClip);
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.batchSchedulerStep = Objects.requireNonNull(batchSchedulerStep, "batchSchedulerStep must not be null");
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
        if (gradientClip > 0.0) {
            GradientClipper.clipByNorm(optimizer.parameters(), (float) gradientClip);
        }
        TensorDiagnostics gradientAfterClip = TrainerTensorDiagnostics.gradients(optimizer.parameters());
        TrainerTensorDiagnostics.requireFinite(gradientAfterClip, "train", "clipped-gradient", true, failures);

        double lossScaleBeforeUpdate = gradScaler == null ? Double.NaN : gradScaler.getScale();
        if (gradScaler == null) {
            optimizer.step();
        } else {
            gradScaler.step(optimizer);
        }
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
                    null);
        }

        static StepResult optimizerStepped(
                boolean mixedPrecisionUsed,
                double lossScaleBeforeUpdate,
                double lossScale,
                boolean overflowDetected,
                TensorDiagnostics gradientBeforeClip,
                TensorDiagnostics gradientAfterClip,
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
                    parametersAfterStep);
        }
    }
}
