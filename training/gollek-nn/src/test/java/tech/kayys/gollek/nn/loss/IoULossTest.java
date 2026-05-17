package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class IoULossTest {

    @Test
    void iouLossBackpropagatesPredictedBoxGradient() {
        IoULoss loss = new IoULoss();
        float[] predData = new float[] {
                0.2f, 0.1f, 2.2f, 1.8f,
                1.0f, 0.7f, 3.1f, 2.8f
        };
        GradTensor pred = GradTensor.of(predData, 2, 4).requiresGrad(true);
        GradTensor target = GradTensor.of(new float[] {
                0.0f, 0.0f, 2.0f, 2.0f,
                0.5f, 0.5f, 2.5f, 2.5f
        }, 2, 4);

        GradTensor output = loss.forward(pred, target);
        output.backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, predData, target),
                pred.grad().data(),
                3e-3f);
    }

    @Test
    void iouLossRejectsInvalidBoxInputs() {
        IoULoss loss = new IoULoss();
        GradTensor target = GradTensor.of(new float[] {0f, 0f, 1f, 1f}, 1, 4);

        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(4), GradTensor.zeros(4)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(1, 4), GradTensor.zeros(2, 4)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 0f, 1f}, 1, 3),
                        GradTensor.of(new float[] {0f, 0f, 1f}, 1, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 0f, Float.NaN, 1f}, 1, 4), target));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {1f, 0f, 0f, 1f}, 1, 4), target));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 0f, 1f, 1f}, 1, 4),
                        GradTensor.of(new float[] {0f, 1f, 1f, 0f}, 1, 4)));
    }

    private static float[] finiteDifferenceGradient(IoULoss loss, float[] predData, GradTensor target) {
        float[] grad = new float[predData.length];
        float eps = 1e-3f;
        for (int i = 0; i < predData.length; i++) {
            float[] plus = predData.clone();
            float[] minus = predData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (loss.forward(GradTensor.of(plus, 2, 4), target).item()
                    - loss.forward(GradTensor.of(minus, 2, 4), target).item()) / (2f * eps);
        }
        return grad;
    }
}
