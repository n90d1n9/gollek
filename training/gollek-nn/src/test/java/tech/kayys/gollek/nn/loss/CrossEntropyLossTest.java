package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class CrossEntropyLossTest {

    @Test
    void weightedCrossEntropyUsesSampleWeightMeanForLossAndGradient() {
        var loss = new CrossEntropyLoss(new float[] {1f, 1f, 4f});
        float[] logitsData = new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f,
                0f, 0f, 0f
        };
        GradTensor logits = GradTensor.of(logitsData, 3, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {0f, 2f, 2f}, 3);

        GradTensor output = loss.compute(logits, targets);
        output.backward();

        assertEquals(Math.log(3.0), output.item(), 1e-6);
        assertArrayEquals(
                finiteDifferenceGradient(loss, logitsData, targets),
                logits.grad().data(),
                2e-3f);
        assertArrayEquals(new float[] {
                -2f / 27f, 1f / 27f, 1f / 27f,
                4f / 27f, 4f / 27f, -8f / 27f,
                4f / 27f, 4f / 27f, -8f / 27f
        }, logits.grad().data(), 1e-6f);
    }

    @Test
    void crossEntropyRejectsInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new CrossEntropyLoss(new float[0]));
        assertThrows(IllegalArgumentException.class, () -> new CrossEntropyLoss(new float[] {1.0f, 0.0f}));

        CrossEntropyLoss loss = new CrossEntropyLoss();
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(3), GradTensor.zeros(3)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(0, 3), GradTensor.zeros(0)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 0), GradTensor.zeros(1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2, 3), GradTensor.zeros(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2, 3), GradTensor.zeros(3)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(
                        GradTensor.of(new float[] {0.0f, Float.POSITIVE_INFINITY, 1.0f}, 1, 3),
                        GradTensor.of(new float[] {0.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> new CrossEntropyLoss(new float[] {1.0f, 2.0f})
                        .compute(GradTensor.zeros(1, 3), GradTensor.zeros(1)));
    }

    private static float[] finiteDifferenceGradient(CrossEntropyLoss loss, float[] logitsData, GradTensor targets) {
        float[] grad = new float[logitsData.length];
        float eps = 1e-3f;
        for (int i = 0; i < logitsData.length; i++) {
            float[] plus = logitsData.clone();
            float[] minus = logitsData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.compute(GradTensor.of(plus, 3, 3), targets).item()
                    - loss.compute(GradTensor.of(minus, 3, 3), targets).item()) / (2f * eps);
        }
        return grad;
    }
}
