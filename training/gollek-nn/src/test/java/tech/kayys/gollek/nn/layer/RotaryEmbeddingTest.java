package tech.kayys.gollek.ml.nn.layer;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class RotaryEmbeddingTest {

    @Test
    void rotaryEmbeddingBackpropagatesToInput() {
        var rope = new RotaryEmbedding(4, 8);
        float[] values = new float[] {
                0.2f, -0.4f, 0.6f, 0.8f,
                -0.3f, 0.5f, 0.7f, -0.2f,
                0.9f, -0.1f, -0.6f, 0.4f
        };
        float[] weights = new float[] {
                0.7f, -0.2f, 0.5f, -0.9f,
                -0.4f, 0.3f, 0.8f, 0.1f,
                0.2f, -0.7f, -0.5f, 0.6f
        };
        GradTensor input = GradTensor.of(values, 1, 1, 3, 4).requiresGrad(true);

        rope.apply(input)
                .mul(GradTensor.of(weights, 1, 1, 3, 4))
                .sum()
                .backward();

        assertNotNull(input.grad());
        assertArrayEquals(
                finiteDifferenceInputGradient(rope, values, weights),
                input.grad().data(),
                2e-3f);
    }

    @Test
    void rotaryEmbeddingValidatesShapeAndSequenceBounds() {
        var rope = new RotaryEmbedding(4, 2);

        assertThrows(IllegalArgumentException.class, () -> rope.apply(GradTensor.zeros(1, 2, 4)));
        assertThrows(IllegalArgumentException.class, () -> rope.apply(GradTensor.zeros(1, 1, 1, 2)));
        assertThrows(IllegalArgumentException.class, () -> rope.apply(GradTensor.zeros(1, 1, 3, 4)));
    }

    private static float[] finiteDifferenceInputGradient(RotaryEmbedding rope, float[] values, float[] weights) {
        float[] grad = new float[values.length];
        float eps = 1e-3f;
        for (int i = 0; i < values.length; i++) {
            float[] plus = values.clone();
            float[] minus = values.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (weightedForwardSum(rope, plus, weights) - weightedForwardSum(rope, minus, weights)) / (2f * eps);
        }
        return grad;
    }

    private static float weightedForwardSum(RotaryEmbedding rope, float[] values, float[] weights) {
        float[] output = rope.apply(GradTensor.of(values, 1, 1, 3, 4)).data();
        float sum = 0f;
        for (int i = 0; i < output.length; i++) {
            sum += output[i] * weights[i];
        }
        return sum;
    }
}
