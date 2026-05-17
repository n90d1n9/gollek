package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class GroupNormTest {

    @Test
    void rootGroupNormBackpropagatesThroughInputAndAffineParameters() {
        var groupNorm = new GroupNorm(2, 4);
        float[] values = new float[] {
                0.2f, -0.4f,
                0.6f, 0.8f,
                -0.3f, 0.5f,
                0.7f, -0.2f
        };
        float[] weights = new float[] {
                0.7f, -0.2f,
                0.5f, -0.9f,
                -0.4f, 0.3f,
                0.8f, 0.1f
        };
        GradTensor input = GradTensor.of(values, 1, 4, 1, 2).requiresGrad(true);

        groupNorm.forward(input)
                .mul(GradTensor.of(weights, 1, 4, 1, 2))
                .sum()
                .backward();

        assertNotNull(input.grad());
        assertArrayEquals(
                finiteDifferenceInputGradient(groupNorm, values, weights),
                input.grad().data(),
                4e-3f);
        assertNotNull(groupNorm.namedParameters().get("weight").grad());
        assertNotNull(groupNorm.namedParameters().get("bias").grad());
        assertArrayEquals(new float[] {0.5f, -0.4f, -0.1f, 0.9f},
                groupNorm.namedParameters().get("bias").grad().data(),
                1e-6f);
    }

    @Test
    void rootGroupNormValidatesConfigurationAndInputShape() {
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(0, 4));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(3, 4));
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(2, 4, Float.NaN));

        var groupNorm = new GroupNorm(2, 4);
        assertThrows(IllegalArgumentException.class, () -> groupNorm.forward(GradTensor.zeros(1, 4, 2)));
        assertThrows(IllegalArgumentException.class, () -> groupNorm.forward(GradTensor.zeros(1, 2, 1, 2)));
    }

    private static float[] finiteDifferenceInputGradient(GroupNorm groupNorm, float[] values, float[] weights) {
        float[] grad = new float[values.length];
        float eps = 1e-3f;
        for (int i = 0; i < values.length; i++) {
            float[] plus = values.clone();
            float[] minus = values.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (weightedForwardSum(groupNorm, plus, weights) - weightedForwardSum(groupNorm, minus, weights))
                    / (2f * eps);
        }
        return grad;
    }

    private static float weightedForwardSum(GroupNorm groupNorm, float[] values, float[] weights) {
        float[] output = groupNorm.forward(GradTensor.of(values, 1, 4, 1, 2)).data();
        float sum = 0f;
        for (int i = 0; i < output.length; i++) {
            sum += output[i] * weights[i];
        }
        return sum;
    }
}
