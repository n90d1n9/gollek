package tech.kayys.gollek.ml.optim;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GradScalerTest {

    @Test
    void scaleReturnsDifferentiableLossAndUnscaleRestoresParameterGradient() {
        Parameter parameter = parameter(2f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        GradScaler scaler = GradScaler.builder().initScale(8.0).build();

        GradTensor loss = parameter.data().mul(3f);
        GradTensor scaledLoss = scaler.scale(loss);

        assertEquals(48f, scaledLoss.item(), 1e-6f);

        scaledLoss.backward();

        assertArrayEquals(new float[] {24f}, parameter.grad().data(), 1e-6f);
        assertFalse(scaler.unscaleAndCheck(optimizer));
        assertFalse(scaler.overflowDetected());
        assertArrayEquals(new float[] {3f}, parameter.grad().data(), 1e-6f);

        scaler.step(optimizer);

        assertArrayEquals(new float[] {1.7f}, parameter.data().data(), 1e-6f);
    }

    @Test
    void stepIsSkippedAndScaleBacksOffWhenGradientOverflows() {
        Parameter parameter = parameter(2f);
        Optimizer optimizer = SGD.builder(List.of(parameter), 0.1f).build();
        GradScaler scaler = GradScaler.builder()
                .initScale(4.0)
                .backoffFactor(0.25)
                .build();

        parameter.data().backward(GradTensor.of(new float[] {Float.POSITIVE_INFINITY}, 1));

        assertTrue(scaler.unscaleAndCheck(optimizer));
        assertTrue(scaler.overflowDetected());

        scaler.step(optimizer);
        scaler.update();

        assertArrayEquals(new float[] {2f}, parameter.data().data(), 1e-6f);
        assertEquals(1.0, scaler.getScale(), 1e-9);
    }

    @Test
    void scaleGrowsAfterConfiguredSuccessfulSteps() {
        GradScaler scaler = GradScaler.builder()
                .initScale(2.0)
                .growthFactor(3.0)
                .growthInterval(2)
                .build();

        scaler.update();
        assertEquals(2.0, scaler.getScale(), 1e-9);

        scaler.update();
        assertEquals(6.0, scaler.getScale(), 1e-9);
    }

    @Test
    void scaleGrowthIsCappedAtFloatRepresentableLimit() {
        GradScaler scaler = GradScaler.builder()
                .initScale(Float.MAX_VALUE / 2.0)
                .growthFactor(4.0)
                .growthInterval(1)
                .build();

        scaler.update();

        assertEquals(Float.MAX_VALUE, scaler.getScale(), Math.ulp((double) Float.MAX_VALUE));
    }

    @Test
    void builderRejectsInvalidScalePolicy() {
        assertThrows(IllegalArgumentException.class, () -> GradScaler.builder().initScale(0.0));
        assertThrows(IllegalArgumentException.class, () -> GradScaler.builder().initScale(Double.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> GradScaler.builder().growthFactor(1.0));
        assertThrows(IllegalArgumentException.class, () -> GradScaler.builder().backoffFactor(1.0));
        assertThrows(IllegalArgumentException.class, () -> GradScaler.builder().growthInterval(0));
    }

    @Test
    void stateDictRestoresLossScaleContinuity() {
        GradScaler scaler = GradScaler.builder()
                .initScale(4.0)
                .growthFactor(2.0)
                .backoffFactor(0.5)
                .growthInterval(2)
                .build();

        scaler.update();
        Map<String, Object> state = scaler.stateDict();

        GradScaler restored = GradScaler.builder()
                .initScale(4.0)
                .growthFactor(2.0)
                .backoffFactor(0.5)
                .growthInterval(2)
                .build();

        assertTrue(restored.supportsStateDict());

        restored.loadStateDict(state);
        assertEquals(4.0, restored.getScale(), 1e-9);
        assertFalse(restored.overflowDetected());

        restored.update();
        assertEquals(8.0, restored.getScale(), 1e-9);
    }

    @Test
    void stateDictRejectsInvalidPayloads() {
        GradScaler scaler = GradScaler.builder()
                .initScale(4.0)
                .growthFactor(2.0)
                .backoffFactor(0.5)
                .growthInterval(2)
                .build();

        Map<String, Object> state = scaler.stateDict();

        assertThrows(IllegalArgumentException.class,
                () -> scaler.loadStateDict(with(state, "scaler", "OtherScaler")));
        assertThrows(IllegalArgumentException.class,
                () -> scaler.loadStateDict(with(state, "scale", Double.NaN)));
        assertThrows(IllegalArgumentException.class,
                () -> scaler.loadStateDict(with(state, "growthFactor", 3.0)));
        assertThrows(IllegalArgumentException.class,
                () -> scaler.loadStateDict(with(state, "stepsWithoutOverflow", -1)));
        assertThrows(IllegalArgumentException.class,
                () -> scaler.loadStateDict(with(state, "overflowDetected", "maybe")));
    }

    @Test
    void unscaleRejectsInvalidInputsAndDoesNotPartiallyMutateOnOverflow() {
        Parameter parameter = parameter(8f, 16f);
        GradScaler scaler = GradScaler.builder().initScale(4.0).build();

        parameter.data().backward(GradTensor.of(new float[] {8f, Float.POSITIVE_INFINITY}, 2));

        assertThrows(IllegalArgumentException.class,
                () -> scaler.unscaleAndCheckParameters(Arrays.asList(parameter, null)));
        assertTrue(scaler.unscaleAndCheckParameters(List.of(parameter)));
        assertTrue(scaler.overflowDetected());
        assertArrayEquals(new float[] {8f, Float.POSITIVE_INFINITY}, parameter.grad().data(), 0f);

        assertThrows(NullPointerException.class, () -> scaler.step(null));
    }

    private static Map<String, Object> with(Map<String, Object> source, String key, Object value) {
        Map<String, Object> copy = new HashMap<>(source);
        copy.put(key, value);
        return copy;
    }

    private static Parameter parameter(float value) {
        return new Parameter(GradTensor.of(new float[] {value}, 1));
    }

    private static Parameter parameter(float... values) {
        return new Parameter(GradTensor.of(values.clone(), values.length));
    }
}
