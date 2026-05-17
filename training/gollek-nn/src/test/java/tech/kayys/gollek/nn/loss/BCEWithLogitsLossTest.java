package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BCEWithLogitsLossTest {

    @Test
    void stableBceWithLogitsKeepsExtremeWrongPredictionLoss() {
        BCEWithLogitsLoss loss = new BCEWithLogitsLoss();
        GradTensor logits = GradTensor.of(new float[] {-100f, 100f}, 2);
        GradTensor targets = GradTensor.of(new float[] {1f, 0f}, 2);

        assertEquals(100.0f, loss.compute(logits, targets).item(), 1e-5f);
    }

    @Test
    void positiveWeightedBceWithLogitsGradientMatchesFiniteDifference() {
        BCEWithLogitsLoss loss = new BCEWithLogitsLoss(new float[] {2.0f, 4.0f});
        float[] logitsData = new float[] {
                0f, 0f,
                0f, 0f
        };
        GradTensor logits = GradTensor.of(logitsData, 2, 2).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1f, 0f, 0f, 1f}, 2, 2);

        GradTensor output = loss.compute(logits, targets);
        output.backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, logitsData, targets),
                logits.grad().data(),
                2e-3f);
        assertArrayEquals(new float[] {
                -0.25f, 0.125f,
                0.125f, -0.5f
        }, logits.grad().data(), 1e-6f);
    }

    @Test
    void bceWithLogitsRejectsNonBinaryOrNonFiniteTargets() {
        BCEWithLogitsLoss loss = new BCEWithLogitsLoss();
        GradTensor logits = GradTensor.zeros(2);

        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(logits, GradTensor.of(new float[] {0f, 0.5f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(logits, GradTensor.of(new float[] {0f, Float.NaN}, 2)));
    }

    private static float[] finiteDifferenceGradient(
            BCEWithLogitsLoss loss, float[] logitsData, GradTensor targets) {
        float[] grad = new float[logitsData.length];
        float eps = 1e-3f;
        for (int i = 0; i < logitsData.length; i++) {
            float[] plus = logitsData.clone();
            float[] minus = logitsData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.compute(GradTensor.of(plus, 2, 2), targets).item()
                    - loss.compute(GradTensor.of(minus, 2, 2), targets).item()) / (2f * eps);
        }
        return grad;
    }
}
