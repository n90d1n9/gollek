package tech.kayys.gollek.ml.nn.layer;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class LayerValidationTest {

    @Test
    void dropoutRejectsNonFiniteProbabilities() {
        assertThrows(IllegalArgumentException.class, () -> new Dropout(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(Float.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(1.1f));
    }

    @Test
    void dropoutAtOneZerosOutputAndGradientDeterministically() {
        var dropout = new Dropout(1f);
        GradTensor input = GradTensor.of(new float[] {1f, -2f, 3f}, 3).requiresGrad(true);
        GradTensor output = dropout.forward(input);

        output.sum().backward();

        assertArrayEquals(new float[] {0f, 0f, 0f}, output.data(), 0f);
        assertNotNull(input.grad());
        assertArrayEquals(new float[] {0f, 0f, 0f}, input.grad().data(), 0f);
    }

    @Test
    void layerGroupNormValidatesConfigurationAndInputChannels() {
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(0, 4));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(2, 0));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(3, 4));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(2, 4, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(2, 4, 0f));

        var groupNorm = new GroupNorm(2, 4);
        assertThrows(IllegalArgumentException.class, () -> groupNorm.forward(GradTensor.zeros(1, 4, 2)));
        assertThrows(IllegalArgumentException.class, () -> groupNorm.forward(GradTensor.zeros(1, 2, 1, 2)));
    }
}
