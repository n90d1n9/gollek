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
    void testNAdamUsesNesterovMomentAndRestoresState() {
        Parameter original = parameter(new float[] {1f, -2f});
        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        var optimizer = NAdam.builder(List.of(original), 0.001f).build();

        optimizer.step();

        assertTrue(optimizer.supportsStateDict());
        assertEquals(1, optimizer.stepCount());
        assertArrayEquals(new float[] {0.9981f, -1.9981f}, original.data().data(), 1e-6f);

        Map<String, Object> state = optimizer.stateDict();
        optimizer.zeroGrad();
        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = NAdam.builder(List.of(restored), 0.001f).build();
        restoredOptimizer.loadStateDict(state);

        assertEquals(1, restoredOptimizer.stepCount());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
    }

    @Test
    void testNAdamDecoupledWeightDecayDoesNotMutateGradientBuffer() {
        Parameter parameter = parameter(new float[] {2f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = NAdam.builder(List.of(parameter), 0.01f)
                .weightDecay(0.1f)
                .decoupledWeightDecay(true)
                .build();

        optimizer.step();

        assertEquals(0.5f, parameter.grad().data()[0], 1e-7f);
        assertTrue(parameter.data().data()[0] < 2.0f);
    }

    @Test
    void testLookaheadSynchronizesFromInitialSlowWeightsAndRestoresState() {
        Parameter original = parameter(new float[] {1f});
        var base = SGD.builder(List.of(original), 0.1f).momentum(0.9f).build();
        var optimizer = new Lookahead(base, 2, 0.5f);

        original.data().backward(GradTensor.of(new float[] {1f}, 1));
        optimizer.step();
        assertEquals(0.9f, original.data().data()[0], 1e-6f);
        optimizer.zeroGrad();

        original.data().backward(GradTensor.of(new float[] {1f}, 1));
        optimizer.step();
        assertEquals(0.855f, original.data().data()[0], 1e-6f);
        assertEquals(2, optimizer.stepCount());
        assertTrue(optimizer.supportsStateDict());

        Map<String, Object> state = optimizer.stateDict();
        optimizer.zeroGrad();
        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = new Lookahead(
                SGD.builder(List.of(restored), 0.1f).momentum(0.9f).build(),
                2,
                0.5f);
        restoredOptimizer.loadStateDict(state);

        assertEquals(2, restoredOptimizer.stepCount());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        original.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        restored.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
    }

    @Test
    void testLAMBStateDictRestoresMomentContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = new LAMB(List.of(original), 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals("LAMB", state.get("optimizer"));
        assertEquals(1, optimizer.stepCount());

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = new LAMB(List.of(restored), 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-6f);
    }

    @Test
    void testLionStateDictRestoresMomentumContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = new Lion(List.of(original), 0.01f, 0.9f, 0.99f, 0.01f);

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals("Lion", state.get("optimizer"));
        assertEquals(1, optimizer.stepCount());

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = new Lion(List.of(restored), 0.01f, 0.9f, 0.99f, 0.01f);
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
    }

    @Test
    void testAdagradStateDictRestoresAccumulatorContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = new Adagrad(List.of(original), 0.1f, 1e-8f, 0.01f);

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals("Adagrad", state.get("optimizer"));
        assertEquals(1, optimizer.stepCount());

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = new Adagrad(List.of(restored), 0.1f, 1e-8f, 0.01f);
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
    }

    @Test
    void testAdadeltaStateDictRestoresAccumulatorContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        var optimizer = new Adadelta(List.of(original), 1.0f, 0.95f, 1e-6f);

        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        optimizer.step();
        optimizer.zeroGrad();

        Map<String, Object> state = optimizer.stateDict();
        assertTrue(optimizer.supportsStateDict());
        assertEquals("Adadelta", state.get("optimizer"));
        assertEquals(1, optimizer.stepCount());

        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = new Adadelta(List.of(restored), 1.0f, 0.95f, 1e-6f);
        restoredOptimizer.loadStateDict(state);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
    }

    @Test
    void testRAdamStartsUnrectifiedAndRestoresMomentContinuity() {
        Parameter original = parameter(new float[] {1f, -2f});
        original.data().backward(GradTensor.of(new float[] {1f, -1f}, 2));
        var optimizer = RAdam.builder(List.of(original), 0.001f).build();

        optimizer.step();

        assertTrue(optimizer.supportsStateDict());
        assertEquals(1, optimizer.stepCount());
        assertArrayEquals(new float[] {0.999f, -1.999f}, original.data().data(), 1e-6f);

        Map<String, Object> state = optimizer.stateDict();
        optimizer.zeroGrad();
        Parameter restored = parameter(original.data().data().clone());
        var restoredOptimizer = RAdam.builder(List.of(restored), 0.001f).build();
        restoredOptimizer.loadStateDict(state);

        assertEquals(1, restoredOptimizer.stepCount());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        original.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        restored.data().backward(GradTensor.of(new float[] {0.25f, -0.5f}, 2));
        optimizer.step();
        restoredOptimizer.step();

        assertArrayEquals(original.data().data(), restored.data().data(), 1e-7f);
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
                () -> NAdam.builder(List.of(parameter(new float[] {1f})), 0.1f).betas(0.9f, 1.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> NAdam.builder(List.of(parameter(new float[] {1f})), 0.1f).eps(0.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> RAdam.builder(List.of(parameter(new float[] {1f})), 0.1f).betas(0.9f, 1.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> RAdam.builder(List.of(parameter(new float[] {1f})), 0.1f).eps(0.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> RMSprop.builder(List.of(parameter(new float[] {1f})), 0.1f).alpha(1.0f).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Lookahead(SGD.builder(List.of(parameter(new float[] {1f})), 0.1f).build(), 0, 0.5f));
        assertThrows(IllegalArgumentException.class,
                () -> new Lookahead(SGD.builder(List.of(parameter(new float[] {1f})), 0.1f).build(), 5, 0.0f));

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
    void gradientClipperReportsDetailedNormTelemetry() {
        Parameter first = parameter(new float[] {1f, -2f});
        Parameter second = parameter(new float[] {3f, -4f});
        first.data().backward(GradTensor.of(new float[] {3f, 4f}, 2));
        second.data().backward(GradTensor.of(new float[] {0f, 12f}, 2));

        var result = GradientClipper.clipByNormDetailed(List.of(first, second), 5.0f);

        assertTrue(result.clipped());
        assertEquals(13.0f, result.preClipNorm(), 1e-6f);
        assertEquals(5.0f / (13.0f + 1e-6f), result.scale(), 1e-7f);
        assertEquals(5.0f, result.maxNorm(), 1e-6f);
        assertEquals(5.0f, result.postClipNorm(), 1e-5f);
        assertEquals(result.postClipNorm(), GradientClipper.totalNorm(List.of(first, second)), 1e-6f);

        float previousNorm = GradientClipper.clipByNorm(List.of(first, second), 10.0f);
        assertEquals(result.postClipNorm(), previousNorm, 1e-6f);
    }

    @Test
    void gradientClipperDetailedResultReportsNoopWhenBelowThreshold() {
        Parameter parameter = parameter(new float[] {1f, -2f});
        parameter.data().backward(GradTensor.of(new float[] {3f, 4f}, 2));

        var result = GradientClipper.clipByNormDetailed(List.of(parameter), 10.0f);

        assertFalse(result.clipped());
        assertEquals(5.0f, result.preClipNorm(), 1e-6f);
        assertEquals(5.0f, result.postClipNorm(), 1e-6f);
        assertEquals(1.0f, result.scale(), 1e-6f);
        assertArrayEquals(new float[] {3f, 4f}, parameter.grad().data(), 1e-6f);
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
    void testNAdamStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = NAdam.builder(List.of(parameter), 0.001f)
                .betas(0.9f, 0.99f)
                .weightDecay(0.01f)
                .decoupledWeightDecay(true)
                .build();

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatchedBeta = NAdam.builder(List.of(parameter(new float[] {0f})), 0.001f)
                .betas(0.8f, 0.99f)
                .weightDecay(0.01f)
                .decoupledWeightDecay(true)
                .build();
        assertThrows(IllegalArgumentException.class, () -> mismatchedBeta.loadStateDict(state));

        var mismatchedDecayMode = NAdam.builder(List.of(parameter(new float[] {0f})), 0.001f)
                .betas(0.9f, 0.99f)
                .weightDecay(0.01f)
                .build();
        assertThrows(IllegalArgumentException.class, () -> mismatchedDecayMode.loadStateDict(state));
    }

    @Test
    void testLookaheadStateDictRejectsMismatchedWrapperConfig() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {1f}, 1));
        var optimizer = new Lookahead(SGD.builder(List.of(parameter), 0.1f).build(), 2, 0.5f);

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = new Lookahead(
                SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build(),
                3,
                0.5f);
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testLAMBStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = new LAMB(List.of(parameter), 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = new LAMB(List.of(parameter(new float[] {0f})), 0.01f, 0.8f, 0.999f, 1e-6f, 0.01f);
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testLionStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = new Lion(List.of(parameter), 0.01f, 0.9f, 0.99f, 0.01f);

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = new Lion(List.of(parameter(new float[] {0f})), 0.01f, 0.8f, 0.99f, 0.01f);
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testAdagradStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = new Adagrad(List.of(parameter), 0.1f, 1e-8f, 0.01f);

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = new Adagrad(List.of(parameter(new float[] {0f})), 0.1f, 1e-6f, 0.01f);
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testAdadeltaStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = new Adadelta(List.of(parameter), 1.0f, 0.95f, 1e-6f);

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = new Adadelta(List.of(parameter(new float[] {0f})), 1.0f, 0.90f, 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> mismatched.loadStateDict(state));
    }

    @Test
    void testRAdamStateDictRejectsMismatchedHyperparameters() {
        Parameter parameter = parameter(new float[] {1f});
        parameter.data().backward(GradTensor.of(new float[] {0.5f}, 1));
        var optimizer = RAdam.builder(List.of(parameter), 0.001f)
                .betas(0.9f, 0.99f)
                .build();

        optimizer.step();
        Map<String, Object> state = optimizer.stateDict();

        var mismatched = RAdam.builder(List.of(parameter(new float[] {0f})), 0.001f)
                .betas(0.8f, 0.99f)
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
    void testCosineAnnealingWarmRestartsCyclesAndRestoresState() {
        var optimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var scheduler = new CosineAnnealingWarmRestartsLR(optimizer, 4, 2, 0.01f);

        assertTrue(scheduler.supportsStateDict());
        assertEquals(4, scheduler.firstCycleSteps());
        assertEquals(2, scheduler.cycleMultiplier());
        assertEquals(0.1f, scheduler.initialLr(), 1e-7f);
        assertEquals(0.01f, scheduler.minLr(), 1e-7f);

        scheduler.step();
        assertEquals(0.1f, optimizer.learningRate(), 1e-7f);
        scheduler.step();
        assertEquals(0.0775f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.0325f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.01f, optimizer.learningRate(), 1e-7f);
        assertEquals(1, scheduler.cycleIndex());
        assertEquals(0, scheduler.cycleStep());
        assertEquals(8, scheduler.cycleLength());

        scheduler.step();
        assertEquals(0.1f, optimizer.learningRate(), 1e-7f);

        Map<String, Object> state = scheduler.stateDict();
        var restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var restored = new CosineAnnealingWarmRestartsLR(restoredOptimizer, 4, 2, 0.01f);
        restored.loadStateDict(state);

        assertEquals(5, restored.currentStep());
        assertEquals(1, restored.cycleIndex());
        assertEquals(1, restored.cycleStep());
        assertEquals(8, restored.cycleLength());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        scheduler.step();
        restored.step();
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);
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
                () -> new CosineAnnealingWarmRestartsLR(optimizer(0.1f), 0, 2, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new CosineAnnealingWarmRestartsLR(optimizer(0.1f), 4, 0, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new CosineAnnealingWarmRestartsLR(optimizer(0.1f), 4, 2, 0.2f));
        assertThrows(IllegalArgumentException.class,
                () -> new ExponentialLR(optimizer(0.1f), 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new WarmupCosineScheduler(optimizer(0.1f), 1, 10, Float.POSITIVE_INFINITY, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new OneCycleLR(
                        optimizer(0.1f), 10, 0.1f, 1.0f, 25.0f, 10_000.0f,
                        OneCycleLR.AnnealStrategy.COSINE));
        assertThrows(IllegalArgumentException.class,
                () -> new SequentialLR(optimizer(0.1f), List.of()));
        assertThrows(IllegalArgumentException.class, () -> {
            Optimizer optimizer = optimizer(0.1f);
            new SequentialLR(optimizer, List.of(
                    new ExponentialLR(optimizer, 0.5f),
                    new ExponentialLR(optimizer, 0.1f)));
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Optimizer optimizer = optimizer(0.1f);
            new SequentialLR(optimizer, List.of(
                    new ExponentialLR(optimizer, 0.5f),
                    new ExponentialLR(optimizer, 0.1f)), 0);
        });
        assertThrows(IllegalArgumentException.class, () -> new SequentialLR(
                optimizer(0.1f),
                List.of(new ExponentialLR(optimizer(0.1f), 0.5f)), 1));
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

        CosineAnnealingWarmRestartsLR warmRestarts =
                new CosineAnnealingWarmRestartsLR(optimizer(0.1f), 4, 2, 0.0f);
        Map<String, Object> warmRestartsState = new HashMap<>(warmRestarts.stateDict());
        warmRestartsState.put("cycleStep", 4);
        assertThrows(IllegalArgumentException.class, () -> warmRestarts.loadStateDict(warmRestartsState));

        ExponentialLR exponential = new ExponentialLR(optimizer(0.1f), 0.5f);
        Map<String, Object> exponentialState = new HashMap<>(exponential.stateDict());
        exponentialState.put("gamma", 0.25f);
        assertThrows(IllegalArgumentException.class, () -> exponential.loadStateDict(exponentialState));

        Optimizer sequentialOptimizer = optimizer(0.1f);
        SequentialLR sequential = new SequentialLR(sequentialOptimizer, List.of(
                new ExponentialLR(sequentialOptimizer, 0.5f),
                new ExponentialLR(sequentialOptimizer, 0.1f)), 2);
        Map<String, Object> sequentialState = new HashMap<>(sequential.stateDict());
        sequentialState.put("milestones", List.of(3));
        assertThrows(IllegalArgumentException.class, () -> sequential.loadStateDict(sequentialState));

        WarmupCosineScheduler warmup = new WarmupCosineScheduler(optimizer(0.1f), 2, 10, 0.1f, 0.0f);
        Map<String, Object> warmupState = new HashMap<>(warmup.stateDict());
        warmupState.put("currentStep", -1);
        assertThrows(IllegalArgumentException.class, () -> warmup.loadStateDict(warmupState));

        OneCycleLR oneCycle = new OneCycleLR(
                optimizer(0.1f), 10, 0.1f, 0.3f, 10.0f, 100.0f, OneCycleLR.AnnealStrategy.LINEAR);
        Map<String, Object> oneCycleState = new HashMap<>(oneCycle.stateDict());
        oneCycleState.put("annealStrategy", "COSINE");
        assertThrows(IllegalArgumentException.class, () -> oneCycle.loadStateDict(oneCycleState));

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
    void testExponentialLRDecaysAndRestoresState() {
        var optimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var scheduler = new ExponentialLR(optimizer, 0.5f);

        assertTrue(scheduler.supportsStateDict());
        assertEquals(0, scheduler.currentStep());
        assertEquals(0.1f, scheduler.initialLr(), 1e-7f);
        assertEquals(0.5f, scheduler.gamma(), 1e-7f);
        assertEquals(0.1f, scheduler.getLr(), 1e-7f);

        scheduler.step();
        assertEquals(0.05f, optimizer.learningRate(), 1e-7f);
        scheduler.step();
        assertEquals(0.025f, optimizer.learningRate(), 1e-7f);

        Map<String, Object> state = scheduler.stateDict();
        var restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var restored = new ExponentialLR(restoredOptimizer, 0.5f);
        restored.loadStateDict(state);

        assertEquals(2, restored.currentStep());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        scheduler.step();
        restored.step();
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);
        assertEquals(0.0125f, restoredOptimizer.learningRate(), 1e-7f);
    }

    @Test
    void testSequentialLRSwitchesSchedulersAndRestoresState() {
        Optimizer optimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var first = new ExponentialLR(optimizer, 0.5f);
        var second = new ExponentialLR(optimizer, 0.1f);
        var scheduler = new SequentialLR(optimizer, List.of(first, second), 2);

        assertTrue(scheduler.supportsStateDict());
        assertEquals(0, scheduler.activeIndex());
        assertArrayEquals(new int[] {2}, scheduler.milestones());

        scheduler.step();
        assertEquals(0.05f, optimizer.learningRate(), 1e-7f);
        assertEquals(0, scheduler.activeIndex());
        scheduler.step();
        assertEquals(0.025f, optimizer.learningRate(), 1e-7f);
        scheduler.step();
        assertEquals(0.01f, optimizer.learningRate(), 1e-7f);
        assertEquals(1, scheduler.activeIndex());
        assertEquals(second, scheduler.activeScheduler());

        Map<String, Object> state = scheduler.stateDict();
        Optimizer restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.1f).build();
        var restored = new SequentialLR(restoredOptimizer, List.of(
                new ExponentialLR(restoredOptimizer, 0.5f),
                new ExponentialLR(restoredOptimizer, 0.1f)), 2);
        restored.loadStateDict(state);

        assertEquals(3, restored.currentStep());
        assertEquals(1, restored.activeIndex());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        scheduler.step();
        restored.step();
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);
        assertEquals(0.001f, restoredOptimizer.learningRate(), 1e-7f);
    }

    @Test
    void testOneCycleLRWarmsUpDecaysAndRestoresState() {
        var optimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.5f).build();
        var scheduler = new OneCycleLR(
                optimizer, 10, 0.1f, 0.3f, 10.0f, 100.0f, OneCycleLR.AnnealStrategy.LINEAR);

        assertTrue(scheduler.supportsStateDict());
        assertEquals(3, scheduler.warmupSteps());
        assertEquals(0.01f, scheduler.initialLr(), 1e-7f);
        assertEquals(0.0001f, scheduler.minLr(), 1e-7f);
        assertEquals(0.01f, optimizer.learningRate(), 1e-7f);

        scheduler.step();
        assertEquals(0.04f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.07f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.1f, optimizer.learningRate(), 1e-6f);
        scheduler.step();
        assertEquals(0.08572857f, optimizer.learningRate(), 1e-6f);

        Map<String, Object> state = scheduler.stateDict();
        var restoredOptimizer = SGD.builder(List.of(parameter(new float[] {0f})), 0.5f).build();
        var restored = new OneCycleLR(
                restoredOptimizer, 10, 0.1f, 0.3f, 10.0f, 100.0f, OneCycleLR.AnnealStrategy.LINEAR);
        restored.loadStateDict(state);

        assertEquals(4, restored.currentStep());
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        scheduler.step();
        restored.step();
        assertEquals(optimizer.learningRate(), restoredOptimizer.learningRate(), 1e-7f);

        while (scheduler.currentStep() < scheduler.totalSteps()) {
            scheduler.step();
        }
        assertEquals(0.0001f, optimizer.learningRate(), 1e-7f);
        scheduler.step();
        assertEquals(0.0001f, optimizer.learningRate(), 1e-7f);
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
