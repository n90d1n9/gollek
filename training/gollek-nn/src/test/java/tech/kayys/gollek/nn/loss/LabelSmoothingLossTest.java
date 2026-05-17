package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class LabelSmoothingLossTest {

    @Test
    void labelSmoothingBackpropagatesSmoothedCrossEntropyGradient() {
        LabelSmoothingLoss loss = new LabelSmoothingLoss(0.2f);
        float[] logitsData = new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f
        };
        GradTensor logits = GradTensor.of(logitsData, 2, 3).requiresGrad(true);
        GradTensor labels = GradTensor.of(new float[] {0f, 2f}, 2);

        GradTensor output = loss.forward(logits, labels);
        output.backward();

        assertEquals(Math.log(3.0), output.item(), 1e-6);
        assertArrayEquals(
                finiteDifferenceGradient(loss, logitsData, labels),
                logits.grad().data(),
                2e-3f);
        assertArrayEquals(new float[] {
                -4f / 15f, 2f / 15f, 2f / 15f,
                2f / 15f, 2f / 15f, -4f / 15f
        }, logits.grad().data(), 1e-6f);
    }

    private static float[] finiteDifferenceGradient(
            LabelSmoothingLoss loss, float[] logitsData, GradTensor labels) {
        float[] grad = new float[logitsData.length];
        float eps = 1e-3f;
        for (int i = 0; i < logitsData.length; i++) {
            float[] plus = logitsData.clone();
            float[] minus = logitsData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.forward(GradTensor.of(plus, 2, 3), labels).item()
                    - loss.forward(GradTensor.of(minus, 2, 3), labels).item()) / (2f * eps);
        }
        return grad;
    }
}
