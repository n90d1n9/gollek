package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DiceLossTest {

    @Test
    void diceLossBackpropagatesProbabilityGradient() {
        DiceLoss loss = new DiceLoss(1.0f);
        float[] predData = new float[] {0.2f, 0.7f, 0.4f, 0.9f};
        GradTensor pred = GradTensor.of(predData, 4).requiresGrad(true);
        GradTensor target = GradTensor.of(new float[] {0f, 1f, 1f, 0f}, 4);

        GradTensor output = loss.forward(pred, target);
        output.backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, predData, target),
                pred.grad().data(),
                2e-3f);
    }

    @Test
    void diceLossRejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new DiceLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new DiceLoss(-0.1f));

        DiceLoss loss = new DiceLoss();
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(2), GradTensor.zeros(1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0.2f, 1.2f}, 2),
                        GradTensor.of(new float[] {0f, 1f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0.2f, 0.8f}, 2),
                        GradTensor.of(new float[] {0f, Float.NaN}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0.2f, 0.8f}, 2),
                        GradTensor.of(new float[] {0f, 0.5f}, 2)));
    }

    private static float[] finiteDifferenceGradient(DiceLoss loss, float[] predData, GradTensor target) {
        float[] grad = new float[predData.length];
        float eps = 1e-3f;
        for (int i = 0; i < predData.length; i++) {
            float[] plus = predData.clone();
            float[] minus = predData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.forward(GradTensor.of(plus, 4), target).item()
                    - loss.forward(GradTensor.of(minus, 4), target).item()) / (2f * eps);
        }
        return grad;
    }
}
