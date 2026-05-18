package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ContrastiveLossTest {

    @Test
    void contrastiveLossBackpropagatesEmbeddingGradients() {
        ContrastiveLoss loss = new ContrastiveLoss(1.0f);
        float[] x1Data = new float[] {
                0.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 0.0f
        };
        float[] x2Data = new float[] {
                0.4f, 0.2f,
                1.6f, 1.0f,
                2.0f, 0.0f
        };
        GradTensor labels = GradTensor.of(new float[] {1.0f, 0.0f, 0.0f}, 3);
        GradTensor x1 = GradTensor.of(x1Data, 3, 2).requiresGrad(true);
        GradTensor x2 = GradTensor.of(x2Data, 3, 2).requiresGrad(true);

        loss.forward(x1, x2, labels).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, x1Data, x2Data, labels, Input.X1),
                x1.grad().data(),
                2e-3f);
        assertArrayEquals(
                finiteDifferenceGradient(loss, x1Data, x2Data, labels, Input.X2),
                x2.grad().data(),
                2e-3f);
    }

    @Test
    void inactiveNegativePairsProduceZeroGradients() {
        ContrastiveLoss loss = new ContrastiveLoss(1.0f);
        GradTensor x1 = GradTensor.of(new float[] {0.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor x2 = GradTensor.of(new float[] {2.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor labels = GradTensor.of(new float[] {0.0f}, 1);

        loss.forward(x1, x2, labels).backward();

        assertArrayEquals(new float[] {0.0f, 0.0f}, x1.grad().data(), 1e-7f);
        assertArrayEquals(new float[] {0.0f, 0.0f}, x2.grad().data(), 1e-7f);
    }

    @Test
    void contrastiveLossRejectsInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new ContrastiveLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new ContrastiveLoss(-0.1f));

        ContrastiveLoss loss = new ContrastiveLoss();
        GradTensor embeddings = GradTensor.zeros(2, 3);

        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(2), GradTensor.zeros(2), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(embeddings, GradTensor.zeros(2, 4), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(embeddings, embeddings, GradTensor.zeros(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(embeddings, embeddings, GradTensor.of(new float[] {0.0f, 0.5f}, 2)));
    }

    private static float[] finiteDifferenceGradient(
            ContrastiveLoss loss,
            float[] x1Data,
            float[] x2Data,
            GradTensor labels,
            Input input) {
        float[] source = input == Input.X1 ? x1Data : x2Data;
        float[] grad = new float[source.length];
        float eps = 1e-3f;
        for (int i = 0; i < source.length; i++) {
            float[] x1Plus = x1Data.clone();
            float[] x1Minus = x1Data.clone();
            float[] x2Plus = x2Data.clone();
            float[] x2Minus = x2Data.clone();
            if (input == Input.X1) {
                x1Plus[i] += eps;
                x1Minus[i] -= eps;
            } else {
                x2Plus[i] += eps;
                x2Minus[i] -= eps;
            }

            grad[i] = (loss.forward(
                            GradTensor.of(x1Plus, 3, 2),
                            GradTensor.of(x2Plus, 3, 2),
                            labels).item()
                    - loss.forward(
                            GradTensor.of(x1Minus, 3, 2),
                            GradTensor.of(x2Minus, 3, 2),
                            labels).item()) / (2f * eps);
        }
        return grad;
    }

    private enum Input {
        X1,
        X2
    }
}
