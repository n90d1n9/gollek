package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTCLossTest {

    @Test
    void ctcLossBackpropagatesLogProbabilityGradient() {
        CTCLoss loss = new CTCLoss();
        float[] logProbsData = new float[] {
                log(0.6f), log(0.3f), log(0.1f),
                log(0.2f), log(0.6f), log(0.2f),
                log(0.4f), log(0.4f), log(0.2f)
        };
        GradTensor logProbs = GradTensor.of(logProbsData, 3, 1, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1.0f}, 1, 1);

        loss.forward(logProbs, targets, new int[] {3}, new int[] {1}).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, logProbsData, targets),
                logProbs.grad().data(),
                2e-3f);
    }

    @Test
    void ctcLossRejectsBlankTargetsAndImpossibleAlignments() {
        CTCLoss loss = new CTCLoss();
        GradTensor logProbs = GradTensor.zeros(2, 1, 3);

        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(logProbs, GradTensor.of(new float[] {0.0f}, 1, 1),
                        new int[] {2}, new int[] {1}));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(logProbs, GradTensor.of(new float[] {1.0f, 1.0f}, 1, 2),
                        new int[] {2}, new int[] {2}));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {Float.NaN, 0.0f, 0.0f}, 1, 1, 3),
                        GradTensor.of(new float[] {1.0f}, 1, 1),
                        new int[] {1}, new int[] {1}));
    }

    private static float[] finiteDifferenceGradient(CTCLoss loss, float[] logProbsData, GradTensor targets) {
        float[] grad = new float[logProbsData.length];
        float eps = 1e-3f;
        for (int i = 0; i < logProbsData.length; i++) {
            float[] plus = logProbsData.clone();
            float[] minus = logProbsData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.forward(GradTensor.of(plus, 3, 1, 3), targets, new int[] {3}, new int[] {1}).item()
                    - loss.forward(GradTensor.of(minus, 3, 1, 3), targets, new int[] {3}, new int[] {1}).item())
                    / (2f * eps);
        }
        return grad;
    }

    private static float log(float value) {
        return (float) Math.log(value);
    }
}
