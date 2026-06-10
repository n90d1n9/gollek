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
    private final TrainerParameterUpdateDiagnosticsPolicy parameterUpdateDiagnostics;
    private final TrainerRuntimeProfiler profiler;
    private int optimizerStepIndex;

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
                TrainerParameterUpdateDiagnosticsPolicy.of(parameterUpdateDiagnostics, 1),
                new TrainerRuntimeProfiler());
    }

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            TrainerGradientClipConfig gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep,
            boolean parameterUpdateDiagnostics) {
        this(
                optimizer,
                gradScaler,
                gradientClip,
                failures,
                batchSchedulerStep,
                TrainerParameterUpdateDiagnosticsPolicy.of(parameterUpdateDiagnostics, 1),
                new TrainerRuntimeProfiler());
    }

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            TrainerGradientClipConfig gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep,
            boolean parameterUpdateDiagnostics,
            TrainerRuntimeProfiler profiler) {
        this(
                optimizer,
                gradScaler,
                gradientClip,
                failures,
                batchSchedulerStep,
                TrainerParameterUpdateDiagnosticsPolicy.of(parameterUpdateDiagnostics, 1),
                profiler);
    }

    TrainerOptimizerStepRunner(
            Optimizer optimizer,
            GradScaler gradScaler,
            TrainerGradientClipConfig gradientClip,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable batchSchedulerStep,
            TrainerParameterUpdateDiagnosticsPolicy parameterUpdateDiagnostics,
            TrainerRuntimeProfiler profiler) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
        this.gradScaler = gradScaler;
        this.gradientClip = Objects.requireNonNull(gradientClip, "gradientClip must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.batchSchedulerStep = Objects.requireNonNull(batchSchedulerStep, "batchSchedulerStep must not be null");
        this.parameterUpdateDiagnostics = Objects.requireNonNull(
                parameterUpdateDiagnostics,
                "parameterUpdateDiagnostics must not be null");
        this.profiler = Objects.requireNonNull(profiler, "profiler must not be null");
    }

    StepResult step(int pendingGradientAccumulationBatches) {
        if (pendingGradientAccumulationBatches <= 0) {
            return StepResult.noPendingGradients();
        }
        profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_ACCUMULATION_SCALE,
                () -> TrainerTensorDiagnostics.scaleGradients(
                        optimizer.parameters(),
                        1.0f / pendingGradientAccumulationBatches));
        boolean overflow = gradScaler != null && profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_AMP_UNSCALE_CHECK,
                () -> gradScaler.unscaleAndCheck(optimizer));
        if (overflow) {
            double scaleBeforeUpdate = gradScaler.getScale();
            profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_AMP_OVERFLOW_UPDATE, () -> gradScaler.update());
            profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_ZERO_GRAD, () -> optimizer.zeroGrad());
            return StepResult.overflowSkipped(
                    scaleBeforeUpdate,
                    gradScaler.getScale());
        }

        TensorDiagnostics gradientBeforeClip = profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_DIAGNOSTICS_BEFORE_CLIP,
                () -> TrainerTensorDiagnostics.gradients(optimizer.parameters()));
        profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_VALIDATE_BEFORE_CLIP,
                () -> TrainerTensorDiagnostics.requireFinite(gradientBeforeClip, "train", "gradient", true, failures));
        double gradientClipScale = 1.0;
        boolean gradientClipped = false;
        if (gradientClip.normEnabled()) {
            GradientClipper.ClipResult clipResult = profiler.time(
                    TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_NORM_CLIP,
                    () -> GradientClipper.clipByNormDetailed(
                            optimizer.parameters(),
                            (float) gradientClip.normThreshold()));
            gradientClipScale = clipResult.scale();
            gradientClipped = clipResult.clipped();
        }
        if (gradientClip.valueEnabled()) {
            TensorDiagnostics afterNormClip = profiler.time(
                    TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_DIAGNOSTICS_AFTER_CLIP,
                    () -> TrainerTensorDiagnostics.gradients(optimizer.parameters()));
            if (afterNormClip.maxAbs() > gradientClip.valueThreshold()) {
                profiler.time(
                        TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_VALUE_CLIP,
                        () -> GradientClipper.clipByValue(
                                optimizer.parameters(),
                                (float) -gradientClip.valueThreshold(),
                                (float) gradientClip.valueThreshold()));
                gradientClipped = true;
            }
        }
        TensorDiagnostics gradientAfterClip = profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_DIAGNOSTICS_AFTER_CLIP,
                () -> TrainerTensorDiagnostics.gradients(optimizer.parameters()));
        profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_VALIDATE_AFTER_CLIP,
                () -> TrainerTensorDiagnostics.requireFinite(
                        gradientAfterClip,
                        "train",
                        "clipped-gradient",
                        true,
                        failures));

        double lossScaleBeforeUpdate = gradScaler == null ? Double.NaN : gradScaler.getScale();
        int nextOptimizerStepIndex = optimizerStepIndex + 1;
        boolean captureParameterUpdates = parameterUpdateDiagnostics.shouldCapture(nextOptimizerStepIndex);
        List<float[]> parametersBeforeStep = captureParameterUpdates
                ? profiler.time(
                        TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_SNAPSHOT,
                        () -> TrainerTensorDiagnostics.snapshotParameters(optimizer.parameters()))
                : List.of();
        if (gradScaler == null) {
            profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, () -> optimizer.step());
        } else {
            profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, () -> gradScaler.step(optimizer));
        }
        TensorDiagnostics parameterUpdates = captureParameterUpdates
                ? profiler.time(
                        TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_UPDATE_DIAGNOSTICS,
                        () -> TrainerTensorDiagnostics.parameterUpdates(optimizer.parameters(), parametersBeforeStep))
                : TensorDiagnostics.empty();
        TensorDiagnostics parametersAfterStep = profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_DIAGNOSTICS,
                () -> TrainerTensorDiagnostics.parameters(optimizer.parameters()));
        profiler.time(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_VALIDATE,
                () -> TrainerTensorDiagnostics.requireFinite(parametersAfterStep, "train", "parameter", false, failures));

        profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_SCHEDULER_STEP, batchSchedulerStep);
        double lossScale = Double.NaN;
        boolean overflowDetected = false;
        if (gradScaler != null) {
            profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_AMP_UPDATE, () -> gradScaler.update());
            lossScale = gradScaler.getScale();
            overflowDetected = gradScaler.overflowDetected();
        }
        profiler.time(TrainerRuntimeProfiler.Phase.OPTIMIZER_ZERO_GRAD, () -> optimizer.zeroGrad());
        optimizerStepIndex = nextOptimizerStepIndex;
        return StepResult.optimizerStepped(
                gradScaler != null,
                lossScaleBeforeUpdate,
                lossScale,
                overflowDetected,
                gradientBeforeClip,
                gradientAfterClip,
                gradientClipScale,
                gradientClipped,
                captureParameterUpdates,
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
