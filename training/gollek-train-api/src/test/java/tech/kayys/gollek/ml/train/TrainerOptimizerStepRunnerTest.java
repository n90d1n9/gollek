package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.SGD;

class TrainerOptimizerStepRunnerTest {

    @Test
    void appliesAccumulationScaleBeforeOptimizerStepAndReportsDiagnostics() {
        Parameter parameter = parameter(10f);
        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();
        AtomicInteger schedulerSteps = new AtomicInteger();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                0.0,
                new RecordingFailures(),
                schedulerSteps::incrementAndGet).step(2);

        assertTrue(result.attempted());
        assertTrue(result.pendingGradientsCleared());
        assertTrue(result.optimizerStepApplied());
        assertFalse(result.overflowSkipped());
        assertFalse(result.mixedPrecisionUsed());
        assertEquals(8.0f, parameter.data().data()[0], 1e-6f);
        assertNull(parameter.grad());
        assertEquals(1, schedulerSteps.get());
        assertEquals(2.0, result.gradientBeforeClip().l2Norm(), 1e-6);
        assertEquals(2.0, result.gradientAfterClip().l2Norm(), 1e-6);
        assertEquals(1.0, result.gradientClipScale(), 1e-6);
        assertFalse(result.gradientClipped());
        assertFalse(result.parameterUpdateDiagnosticsEnabled());
        assertEquals(0.0, result.parameterUpdates().l2Norm(), 1e-6);
        assertEquals(8.0, result.parametersAfterStep().l2Norm(), 1e-6);
    }

    @Test
    void canReportExactParameterUpdateDiagnosticsWhenEnabled() {
        Parameter parameter = parameter(10f);
        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                0.0,
                new RecordingFailures(),
                () -> {
                },
                true).step(2);

        assertTrue(result.optimizerStepApplied());
        assertTrue(result.parameterUpdateDiagnosticsEnabled());
        assertEquals(8.0f, parameter.data().data()[0], 1e-6f);
        assertEquals(1, result.parameterUpdates().tensorCount());
        assertEquals(1L, result.parameterUpdates().valueCount());
        assertEquals(0L, result.parameterUpdates().zeroCount());
        assertEquals(2.0, result.parameterUpdates().sumAbs(), 1e-6);
        assertEquals(2.0, result.parameterUpdates().l2Norm(), 1e-6);
        assertEquals(2.0, result.parameterUpdates().maxAbs(), 1e-6);
        assertEquals(2.0, result.parameterUpdates().meanAbs(), 1e-6);
        assertEquals(2.0, result.parameterUpdates().rms(), 1e-6);
    }

    @Test
    void recordsOptimizerRuntimeProfilePhases() {
        Parameter parameter = parameter(10f);
        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                TrainerGradientClipConfig.value(0.5),
                new RecordingFailures(),
                () -> { },
                true,
                profiler).step(1);

        assertTrue(result.optimizerStepApplied());
        assertEquals(1L, profiler.snapshot(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_GRADIENT_ACCUMULATION_SCALE).count());
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP).count());
        assertEquals(1L, profiler.snapshot(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_UPDATE_DIAGNOSTICS).count());
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.OPTIMIZER_ZERO_GRAD).count());
        assertTrue(profiler.toMetadata("runtimeProfile")
                .containsKey("runtimeProfile.optimizer.step.totalMillis"));
    }

    @Test
    void capturesParameterUpdateDiagnosticsOnConfiguredIntervalOnly() {
        Parameter parameter = parameter(10f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();
        TrainerOptimizerStepRunner runner = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                TrainerGradientClipConfig.DISABLED,
                new RecordingFailures(),
                () -> { },
                TrainerParameterUpdateDiagnosticsPolicy.of(true, 2),
                profiler);

        parameter.data().backward(GradTensor.of(new float[] {1f}, 1));
        TrainerOptimizerStepRunner.StepResult first = runner.step(1);
        parameter.data().backward(GradTensor.of(new float[] {1f}, 1));
        TrainerOptimizerStepRunner.StepResult second = runner.step(1);

        assertTrue(first.optimizerStepApplied());
        assertFalse(first.parameterUpdateDiagnosticsEnabled());
        assertTrue(second.optimizerStepApplied());
        assertTrue(second.parameterUpdateDiagnosticsEnabled());
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_SNAPSHOT).count());
        assertEquals(1L, profiler.snapshot(
                TrainerRuntimeProfiler.Phase.OPTIMIZER_PARAMETER_UPDATE_DIAGNOSTICS).count());
    }

    @Test
    void skipsOptimizerAndSchedulerWhenMixedPrecisionOverflowIsDetected() {
        Parameter parameter = parameter(2f);
        parameter.data().backward(GradTensor.of(new float[] {Float.POSITIVE_INFINITY}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        GradScaler scaler = GradScaler.builder()
                .initScale(4.0)
                .backoffFactor(0.25)
                .build();
        AtomicInteger schedulerSteps = new AtomicInteger();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                scaler,
                0.0,
                new RecordingFailures(),
                schedulerSteps::incrementAndGet).step(1);

        assertTrue(result.attempted());
        assertTrue(result.pendingGradientsCleared());
        assertFalse(result.optimizerStepApplied());
        assertTrue(result.overflowSkipped());
        assertTrue(result.mixedPrecisionUsed());
        assertTrue(result.overflowDetected());
        assertEquals(4.0, result.lossScaleBeforeUpdate(), 1e-9);
        assertEquals(1.0, result.lossScale(), 1e-9);
        assertEquals(2.0f, parameter.data().data()[0], 1e-6f);
        assertNull(parameter.grad());
        assertEquals(0, schedulerSteps.get());
    }

    @Test
    void clipsGradientNormBeforeApplyingOptimizerStep() {
        Parameter parameter = parameter(10f);
        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                1.0,
                new RecordingFailures(),
                () -> { }).step(1);

        assertTrue(result.optimizerStepApplied());
        assertEquals(4.0, result.gradientBeforeClip().l2Norm(), 1e-6);
        assertTrue(result.gradientAfterClip().l2Norm() <= 1.000001);
        assertTrue(result.gradientClipped());
        assertEquals(1.0 / (4.0 + 1e-6), result.gradientClipScale(), 1e-7);
        assertEquals(9.0f, parameter.data().data()[0], 1e-5f);
    }

    @Test
    void clipsGradientValueBeforeApplyingOptimizerStep() {
        Parameter parameter = parameter(10f);
        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                TrainerGradientClipConfig.value(0.5),
                new RecordingFailures(),
                () -> { },
                false).step(1);

        assertTrue(result.optimizerStepApplied());
        assertEquals(4.0, result.gradientBeforeClip().maxAbs(), 1e-6);
        assertEquals(0.5, result.gradientAfterClip().maxAbs(), 1e-6);
        assertTrue(result.gradientClipped());
        assertEquals(1.0, result.gradientClipScale(), 1e-6);
        assertEquals(9.5f, parameter.data().data()[0], 1e-6f);
    }

    @Test
    void noopsWhenThereAreNoPendingAccumulationBatches() {
        Parameter parameter = parameter(5f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        AtomicInteger schedulerSteps = new AtomicInteger();

        TrainerOptimizerStepRunner.StepResult result = new TrainerOptimizerStepRunner(
                optimizer,
                null,
                0.0,
                new RecordingFailures(),
                schedulerSteps::incrementAndGet).step(0);

        assertFalse(result.attempted());
        assertFalse(result.pendingGradientsCleared());
        assertFalse(result.optimizerStepApplied());
        assertFalse(result.parameterUpdateDiagnosticsEnabled());
        assertNull(result.parameterUpdates());
        assertEquals(5.0f, parameter.data().data()[0], 1e-6f);
        assertEquals(0, schedulerSteps.get());
    }

    private static Parameter parameter(float value) {
        return new Parameter(GradTensor.of(new float[] {value}, 1));
    }

    private static final class RecordingFailures implements TrainerBatchGuards.FailureRecorder {
        @Override
        public String invalidBatch(
                String phase,
                String reason,
                String message,
                boolean optimizerStepSkipped) {
            throw new AssertionError("not used");
        }

        @Override
        public String invalidLossShape(
                String phase,
                String shape,
                long elements,
                boolean optimizerStepSkipped) {
            throw new AssertionError("not used");
        }

        @Override
        public String nonFinite(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped) {
            return phase + " " + label + " must be finite, got " + value;
        }

        @Override
        public void discardPendingGradients() {
            throw new AssertionError("not used");
        }
    }
}
