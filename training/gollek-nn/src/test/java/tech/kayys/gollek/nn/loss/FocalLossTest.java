package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FocalLossTest {

    @Test
    void focalLossBackpropagatesFiniteDifferenceGradients() {
        FocalLoss loss = new FocalLoss(1.3f, new float[] {1.0f, 2.0f, 0.5f});
        float[] logitsData = {
                0.4f, -0.8f, 1.2f,
                -0.3f, 0.9f, 0.1f
        };
        GradTensor logits = GradTensor.of(logitsData.clone(), 2, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {2.0f, 0.0f}, 2);

        loss.compute(logits, targets).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, logitsData, targets),
                logits.grad().data(),
                3e-3f);
    }

    @Test
    void focalLossRejectsInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(-0.1f));
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(2.0f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(2.0f, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(2.0f, new float[0]));
        assertThrows(IllegalArgumentException.class, () -> new FocalLoss(2.0f, new float[] {1.0f, 0.0f}));

        FocalLoss loss = new FocalLoss();
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(0, 3), GradTensor.zeros(0)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 0), GradTensor.zeros(1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2, 3), GradTensor.zeros(3)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(
                        GradTensor.of(new float[] {0.0f, Float.POSITIVE_INFINITY, 1.0f}, 1, 3),
                        GradTensor.of(new float[] {0.0f}, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 3), GradTensor.of(new float[] {1.5f}, 1)));
    }

    private static float[] finiteDifferenceGradient(FocalLoss loss, float[] logitsData, GradTensor targets) {
        float[] grad = new float[logitsData.length];
        float eps = 1e-3f;
        for (int i = 0; i < logitsData.length; i++) {
            float[] plus = logitsData.clone();
            float[] minus = logitsData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.compute(GradTensor.of(plus, 2, 3), targets).item()
                    - loss.compute(GradTensor.of(minus, 2, 3), targets).item()) / (2.0f * eps);
        }
        return grad;
    }
}
