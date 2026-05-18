package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.SGD;

class TrainerOptimizationRuntimeTest {

    @Test
    void waitsForConfiguredAccumulationBeforeStepping() {
        Parameter parameter = parameter(10f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();
        AtomicInteger schedulerSteps = new AtomicInteger();
        TrainerOptimizationRuntime runtime = runtime(
                optimizer,
                null,
                2,
                new RecordingFailures(),
                schedulerSteps::incrementAndGet,
                () -> false);

        parameter.data().backward(GradTensor.of(new float[] {4f}, 1));
        runtime.afterBackwardBatch();

        assertFalse(runtime.shouldZeroGradBeforeBackward());
        assertTrue(runtime.hasPendingGradients());
        assertEquals(1, runtime.pendingGradientAccumulationBatches());
        assertEquals(0, runtime.optimizerStepCount());
        assertEquals(10.0f, parameter.data().data()[0], 1e-6f);

        parameter.data().backward(GradTensor.of(new float[] {2f}, 1));
        runtime.afterBackwardBatch();

        assertTrue(runtime.shouldZeroGradBeforeBackward());
        assertFalse(runtime.hasPendingGradients());
        assertEquals(0, runtime.pendingGradientAccumulationBatches());
        assertEquals(1, runtime.optimizerStepCount());
        assertEquals(7.0f, parameter.data().data()[0], 1e-6f);
        assertEquals(1, schedulerSteps.get());
        assertEquals(3.0, runtime.latestGradientDiagnostics().l2Norm(), 1e-6);
        assertEquals(7.0, runtime.latestParameterDiagnostics().l2Norm(), 1e-6);
    }

    @Test
    void flushAppliesPartialAccumulationStep() {
        Parameter parameter = parameter(8f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 1.0f).build();
        TrainerOptimizationRuntime runtime = runtime(
                optimizer,
                null,
                4,
                new RecordingFailures(),
                () -> {
                },
                () -> false);

        parameter.data().backward(GradTensor.of(new float[] {3f}, 1));
        runtime.afterBackwardBatch();
        runtime.flush();

        assertFalse(runtime.hasPendingGradients());
        assertEquals(1, runtime.optimizerStepCount());
        assertEquals(5.0f, parameter.data().data()[0], 1e-6f);
        assertNull(parameter.grad());
    }

    @Test
    void mixedPrecisionOverflowClearsPendingGradientsAndRecordsScale() {
        Parameter parameter = parameter(2f);
        parameter.data().backward(GradTensor.of(new float[] {Float.POSITIVE_INFINITY}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        GradScaler scaler = GradScaler.builder()
                .initScale(4.0)
                .backoffFactor(0.25)
                .build();
        AtomicInteger schedulerSteps = new AtomicInteger();
        TrainerOptimizationRuntime runtime = runtime(
                optimizer,
                scaler,
                1,
                new RecordingFailures(),
                schedulerSteps::incrementAndGet,
                () -> false);

        runtime.afterBackwardBatch();

        assertFalse(runtime.hasPendingGradients());
        assertEquals(0, runtime.optimizerStepCount());
        assertEquals(1, runtime.mixedPrecisionOverflowSkipCount());
        assertTrue(runtime.latestMixedPrecisionOverflowDetected());
        assertEquals(1.0, runtime.latestMixedPrecisionLossScale(), 1e-9);
        assertEquals(2.0f, parameter.data().data()[0], 1e-6f);
        assertNull(parameter.grad());
        assertEquals(0, schedulerSteps.get());
    }

    @Test
    void discardPendingGradientsClearsAccumulatedState() {
        Parameter parameter = parameter(6f);
        parameter.data().backward(GradTensor.of(new float[] {2f}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.5f).build();
        TrainerOptimizationRuntime runtime = runtime(
                optimizer,
                null,
                2,
                new RecordingFailures(),
                () -> {
                },
                () -> false);

        runtime.afterBackwardBatch();
        runtime.discardPendingGradients();

        assertFalse(runtime.hasPendingGradients());
        assertEquals(0, runtime.pendingGradientAccumulationBatches());
        assertNull(parameter.grad());
        assertEquals(6.0f, parameter.data().data()[0], 1e-6f);
    }

    @Test
    void nonFiniteOptimizerFailureDiscardsPendingGradientsBeforeRethrow() {
        Parameter parameter = parameter(4f);
        parameter.data().backward(GradTensor.of(new float[] {Float.NaN}, 1));
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.5f).build();
        AtomicBoolean nonFiniteDetected = new AtomicBoolean(false);
        RecordingFailures failures = new RecordingFailures(nonFiniteDetected);
        TrainerOptimizationRuntime runtime = runtime(
                optimizer,
                null,
                1,
                failures,
                () -> {
                },
                nonFiniteDetected::get);

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                runtime::afterBackwardBatch);

        assertEquals("train gradient must be finite, got NaN", error.getMessage());
        assertFalse(runtime.hasPendingGradients());
        assertNull(parameter.grad());
        assertEquals(0, runtime.optimizerStepCount());
    }

    @Test
    void restoresResumeCountersAndMixedPrecisionSnapshot() {
        TrainerOptimizationRuntime runtime = runtime(
                SGD.builder(List.of(parameter(1f)), 0.1f).build(),
                null,
                1,
                new RecordingFailures(),
                () -> {
                },
                () -> false);

        runtime.restoreOptimizerStepCount(9);
        runtime.restoreMixedPrecisionState(3, 16.0, true);

        assertEquals(9, runtime.optimizerStepCount());
        assertEquals(3, runtime.mixedPrecisionOverflowSkipCount());
        assertEquals(16.0, runtime.latestMixedPrecisionLossScale(), 1e-9);
        assertTrue(runtime.latestMixedPrecisionOverflowDetected());
    }

    private static TrainerOptimizationRuntime runtime(
            Optimizer optimizer,
            GradScaler gradScaler,
            int accumulationSteps,
            TrainerBatchGuards.FailureRecorder failures,
            Runnable schedulerStep,
            java.util.function.BooleanSupplier nonFiniteDetected) {
        return new TrainerOptimizationRuntime(
                optimizer,
                new TrainerOptimizerStepRunner(
                        optimizer,
                        gradScaler,
                        0.0,
                        failures,
                        schedulerStep),
                accumulationSteps,
                gradScaler != null,
                gradScaler,
                nonFiniteDetected);
    }

    private static Parameter parameter(float value) {
        return new Parameter(GradTensor.of(new float[] {value}, 1));
    }

    private static class RecordingFailures implements TrainerBatchGuards.FailureRecorder {
        private final AtomicBoolean nonFiniteDetected;

        RecordingFailures() {
            this(new AtomicBoolean(false));
        }

        RecordingFailures(AtomicBoolean nonFiniteDetected) {
            this.nonFiniteDetected = nonFiniteDetected;
        }

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
            nonFiniteDetected.set(true);
            return phase + " " + label + " must be finite, got " + value;
        }

        @Override
        public void discardPendingGradients() {
            throw new AssertionError("not used");
        }
    }
}
