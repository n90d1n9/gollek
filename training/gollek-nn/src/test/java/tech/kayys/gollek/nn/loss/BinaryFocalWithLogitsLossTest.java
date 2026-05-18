package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BinaryFocalWithLogitsLossTest {

    @Test
    void positiveWeightedBinaryFocalBackpropagatesFiniteDifferenceGradients() {
        BinaryFocalWithLogitsLoss loss = new BinaryFocalWithLogitsLoss(
                1.5f,
                0.35f,
                new float[] {2.0f, 0.5f, 3.0f});
        float[] logitsData = {
                -0.7f, 0.2f, 1.1f,
                0.4f, -1.2f, 0.8f
        };
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f
        }, 2, 3);
        GradTensor logits = GradTensor.of(logitsData.clone(), 2, 3).requiresGrad(true);

        loss.compute(logits, targets).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, logitsData, targets),
                logits.grad().data(),
                3e-3f);
    }

    @Test
    void binaryFocalRejectsInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new BinaryFocalWithLogitsLoss(Float.NaN, 0.25f));
        assertThrows(IllegalArgumentException.class, () -> new BinaryFocalWithLogitsLoss(-0.1f, 0.25f));
        assertThrows(IllegalArgumentException.class, () -> new BinaryFocalWithLogitsLoss(2.0f, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> new BinaryFocalWithLogitsLoss(2.0f, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> new BinaryFocalWithLogitsLoss(2.0f, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new BinaryFocalWithLogitsLoss(2.0f, 0.25f, new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new BinaryFocalWithLogitsLoss(2.0f, 0.25f, new float[] {1.0f, Float.NaN}));

        BinaryFocalWithLogitsLoss loss = new BinaryFocalWithLogitsLoss(2.0f, 0.25f, new float[] {1.0f, 2.0f});
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2, 3), GradTensor.zeros(2, 3)));

        BinaryFocalWithLogitsLoss defaultLoss = new BinaryFocalWithLogitsLoss();
        assertThrows(IllegalArgumentException.class,
                () -> defaultLoss.compute(GradTensor.zeros(2), GradTensor.zeros(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> defaultLoss.compute(GradTensor.zeros(0), GradTensor.zeros(0)));
        assertThrows(IllegalArgumentException.class,
                () -> defaultLoss.compute(
                        GradTensor.of(new float[] {0.0f, Float.NEGATIVE_INFINITY}, 2),
                        GradTensor.of(new float[] {0.0f, 1.0f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> defaultLoss.compute(
                        GradTensor.zeros(2),
                        GradTensor.of(new float[] {0.0f, 0.5f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> defaultLoss.compute(
                        GradTensor.zeros(2),
                        GradTensor.of(new float[] {0.0f, Float.NaN}, 2)));
    }

    private static float[] finiteDifferenceGradient(
            BinaryFocalWithLogitsLoss loss,
            float[] logitsData,
            GradTensor targets) {
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
