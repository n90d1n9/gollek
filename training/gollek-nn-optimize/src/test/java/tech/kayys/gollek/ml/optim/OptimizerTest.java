package tech.kayys.gollek.ml.optim;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class OptimizerTest {

    @Test
    void testSGDStep() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f, -1f}, 2));
        var optimizer = SGD.builder(List.of(parameter), 0.01f).build();

        optimizer.step();

        assertArrayEquals(new float[] {0.995f, -1.99f}, parameter.data().data(), 1e-6f);
    }

    @Test
    void testAdamStep() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        var optimizer = Adam.builder(List.of(parameter), 0.001f).build();

        optimizer.step();

        assertArrayEquals(new float[] {0.999f, -1.999f}, parameter.data().data(), 1e-5f);
    }

    @Test
    void testAdamWStep() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        var optimizer = AdamW.builder(List.of(parameter), 0.001f).build();

        optimizer.step();

        assertArrayEquals(new float[] {0.99899f, -1.99898f}, parameter.data().data(), 1e-5f);
    }

    @Test
    void testZeroGrad() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f, -1f}, 2));
        var optimizer = SGD.builder(List.of(parameter), 0.01f).build();

        assertNotNull(parameter.grad());

        optimizer.zeroGrad();

        var grad = parameter.grad();
        if (grad != null) {
            for (float v : grad.data()) assertEquals(0f, v, 1e-8f);
        }
    }

    @Test
    void testRMSpropStep() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        var optimizer = RMSprop.builder(List.of(parameter), 0.01f).build();

        optimizer.step();

        assertNotEquals(1f, parameter.data().data()[0], 1e-6f);
        assertNotEquals(-2f, parameter.data().data()[1], 1e-6f);
    }

    @Test
    void optimizersRejectInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class,
                () -> SGD.builder(List.of(parameter(new float[] {1f})), Float.NaN).build());
        assertThrows(IllegalArgumentException.class,
                () -> SGD.builder(List.of(parameter(new float[] {1f})), 0.1f).momentum(Float.POSITIVE_INFINITY).build());
        assertThrows(IllegalArgumentException.class,
                () -> SGD.builder(List.of(parameter(new float[] {1f})), 0.1f).weightDecay(-0.1f).build());

        assertThrows(IllegalArgumentException.class,
                () -> Adam.builder(List.of(parameter(new float[] {1f})), 0.1f).betas(1.0f, 0.999f).build());
        assertThrows(IllegalArgumentException.class,
                () -> Adam.builder(List.of(parameter(new float[] {1f})), 0.1f).eps(0.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> AdamW.builder(List.of(parameter(new float[] {1f})), 0.1f).weightDecay(Float.NaN).build());
        assertThrows(IllegalArgumentException.class,
                () -> RMSprop.builder(List.of(parameter(new float[] {1f})), 0.1f).alpha(1.0f).build());

        assertThrows(IllegalArgumentException.class,
                () -> new Adagrad(List.of(parameter(new float[] {1f})), 0.1f, Float.NaN, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new Adadelta(List.of(parameter(new float[] {1f})), -0.1f, 0.95f, 1e-6f));
        assertThrows(IllegalArgumentException.class,
                () -> new Lion(List.of(parameter(new float[] {1f})), 0.1f, -0.1f, 0.99f, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new LAMB(List.of(parameter(new float[] {1f})), 0.1f, 0.9f, 0.999f, 0.0f, 0.01f));
    }

    @Test
    void optimizerStepsRejectNonFiniteGradientsWithoutMutatingState() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {Float.NaN, 1f}, 2));
        var optimizer = AdamW.builder(List.of(parameter), 0.001f).build();

        assertThrows(IllegalStateException.class, optimizer::step);

        assertArrayEquals(new float[] {1f, -2f}, parameter.data().data(), 1e-7f);
        assertEquals(0, optimizer.stateDict().get("step"));
    }

    @Test
    void adamWeightDecayDoesNotMutateGradientBuffer() {
        Parameter parameter = parameter(new float[] {2f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = Adam.builder(List.of(parameter), 0.01f).weightDecay(0.1f).build();

        optimizer.step();

        assertEquals(0.5f, parameter.grad().data()[0], 1e-7f);
        assertEquals(1.99f, parameter.data().data()[0], 1e-6f);
    }

    @Test
    void adagradWeightDecayAppliesAcrossWholeTensor() {
        float[] values = new float[32];
        float[] zeroGrad = new float[32];
        float[] expected = new float[32];
        for (int i = 0; i < values.length; i++) {
            values[i] = i + 1f;
            expected[i] = values[i] - 0.1f;
        }
        Parameter parameter = parameter(values);
        parameter.data().backward(GradTensor.of(zeroGrad, zeroGrad.length));
        var optimizer = new Adagrad(List.of(parameter), 0.1f, 1e-8f, 0.1f);

        optimizer.step();

        assertArrayEquals(expected, parameter.data().data(), 1e-5f);
    }

    @Test
    void gradientClipperRejectsInvalidBoundsAndNonFiniteGradients() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {Float.POSITIVE_INFINITY, 1f}, 2));

        assertThrows(IllegalArgumentException.class,
                () -> GradientClipper.clipByNorm(List.of(parameter), 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> GradientClipper.clipByValue(List.of(parameter), 1.0f, -1.0f));
        assertThrows(IllegalStateException.class,
                () -> GradientClipper.clipByNorm(List.of(parameter), 1.0f));
    }

    @Test
    void testAdamStateDictRestoresMomentContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = Adam.builder(List.of(original), 0.001f).build();

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals(1, state.get("step"));
        assertEquals("Adam", state.get("optimizer"));

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = Adam.builder(List.of(restored), 0.001f).build();
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);

        var mismatched = Adam.builder(List.of(parameter(new float[] {0f, 0f})), 0.001f)
                .betas(0.8f, 0.999f)
                .build();
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testRMSpropStateDictRestoresAccumulatorContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = RMSprop.builder(List.of(original), 0.01f).momentum(0.9f).build();

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals("RMSprop", state.get("optimizer"));
        assertTrue(state.containsKey("squareAvg"));
        assertTrue(state.containsKey("velocity"));

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = RMSprop.builder(List.of(restored), 0.01f).momentum(0.9f).build();
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.5f, -0.25f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.5f, -0.25f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);

        var mismatched = RMSprop.builder(List.of(parameter(new float[] {0f, 0f})), 0.01f)
                .momentum(0.0f)
                .build();
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testLRSchedulerStep() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        var optimizer = Adam.builder(List.of(parameter), 0.01f).build();
        var scheduler = new CosineAnnealingLR(optimizer, 10, 0.0f);

        scheduler.step();

        assertEquals(0.01f, scheduler.getLr(), 1e-6f);
    }

    @Test
    void schedulersRejectInvalidConfigurationAndCheckpointState() {
        assertThrows(IllegalArgumentException.class,
                () -> new StepLR(optimizer(0.1f), 1, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new CosineAnnealingLR(optimizer(0.1f), 10, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new CosineAnnealingLR(optimizer(0.1f), 10, 0.2f));
        assertThrows(IllegalArgumentException.class,
                () -> new WarmupCosineScheduler(optimizer(0.1f), 1, 10, Float.POSITIVE_INFINITY, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new ReduceLROnPlateau(
                        optimizer(0.1f), ReduceLROnPlateau.Mode.MIN, Float.NaN, 1, 0.0, 0, 0.0f));

        StepLR step = new StepLR(optimizer(0.1f), 2, 0.5f);
        Map<String, Object> stepState = new HashMap<>(step.stateDict());
        stepState.put("currentLr", Float.NaN);
        assertThrows(IllegalArgumentException.class, () -> step.loadStateDict(stepState));

        CosineAnnealingLR cosine = new CosineAnnealingLR(optimizer(0.1f), 10, 0.0f);
        Map<String, Object> cosineState = new HashMap<>(cosine.stateDict());
        cosineState.put("step", -1);
        assertThrows(IllegalArgumentException.class, () -> cosine.loadStateDict(cosineState));

        WarmupCosineScheduler warmup = new WarmupCosineScheduler(optimizer(0.1f), 2, 10, 0.1f, 0.0f);
        Map<String, Object> warmupState = new HashMap<>(warmup.stateDict());
        warmupState.put("currentStep", -1);
        assertThrows(IllegalArgumentException.class, () -> warmup.loadStateDict(warmupState));

        ReduceLROnPlateau plateau = new ReduceLROnPlateau(
                optimizer(0.1f), ReduceLROnPlateau.Mode.MIN, 0.5f, 1, 0.0, 0, 0.0f);
        Map<String, Object> plateauState = new HashMap<>(plateau.stateDict());
        plateauState.put("bestMetric", Double.POSITIVE_INFINITY);
        assertThrows(IllegalArgumentException.class, () -> plateau.loadStateDict(plateauState));
    }

    @Test
    void testWarmupCosineStartsAtZeroAndRestoresState() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        var optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        var scheduler = new WarmupCosineScheduler(optimizer, 2, 4, 0.1f, 0.01f);

        assertEquals(0.0f, optimizer.learningRate(), 1e-7f);
        scheduler.step();
        assertEquals(0.05f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.1f, optimizer.learningRate(), 1e-6f);

        Map<String, Object> state = scheduler.stateDict();
        var restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var restored = new WarmupCosineScheduler(restoredOptimizer, 2, 4, 0.1f, 0.01f);
        restored.loadStateDict(state);

        assertEquals(2, restored.currentStep());
        assertEquals(0.1f, restoredOptimizer.learningRate(), 1e-6f);
    }

    @Test
    void testReduceLROnPlateauReducesAfterStaleMetricAndRestoresState() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        var optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        var scheduler = new ReduceLROnPlateau(
                optimizer, ReduceLROnPlateau.Mode.MIN, 0.5f, 1, 0.0, 0, 0.01f);

        scheduler.step(1.0);
        assertEquals(0.1f, optimizer.learningRate(), 1e-6f);
        assertEquals(1.0, scheduler.bestMetric(), 1e-12);
        scheduler.step(1.2);
        assertEquals(0.1f, optimizer.learningRate(), 1e-6f);
        assertEquals(1, scheduler.badSteps());
        scheduler.step(1.1);
        assertEquals(0.05f, optimizer.learningRate(), 1e-6f);
        assertEquals(1, scheduler.reductionCount());

        Map<String, Object> state = scheduler.stateDict();
        var restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var restored = new ReduceLROnPlateau(
                restoredOptimizer, ReduceLROnPlateau.Mode.MIN, 0.5f, 1, 0.0, 0, 0.01f);
        restored.loadStateDict(state);

        assertEquals(3, restored.stepCount());
        assertEquals(0.05f, restoredOptimizer.learningRate(), 1e-6f);
        assertEquals(1, restored.reductionCount());
        assertEquals(1.0, restored.bestMetric(), 1e-12);
    }

    private static Parameter parameter(float[] values) {
        return new Parameter(GradTensor.of(values, values.length));
    }

    private static Optimizer optimizer(float lr) {
        return SGD.builder(List.of(parameter(new float[] {1f})), lr).build();
    }
}
