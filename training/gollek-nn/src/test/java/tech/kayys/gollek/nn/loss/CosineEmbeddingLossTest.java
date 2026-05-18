package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CosineEmbeddingLossTest {

    @Test
    void cosineEmbeddingLossBackpropagatesBothEmbeddingInputs() {
        CosineEmbeddingLoss loss = new CosineEmbeddingLoss(0.2f);
        float[] x1Data = {
                0.7f, -0.2f,
                0.3f, 0.8f,
                -0.4f, 0.9f
        };
        float[] x2Data = {
                0.2f, 0.6f,
                0.9f, 0.1f,
                0.5f, 0.4f
        };
        GradTensor target = GradTensor.of(new float[] {1.0f, -1.0f, -1.0f}, 3);
        GradTensor x1 = GradTensor.of(x1Data, 3, 2).requiresGrad(true);
        GradTensor x2 = GradTensor.of(x2Data, 3, 2).requiresGrad(true);

        loss.compute(x1, x2, target).backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, x1Data, x2Data, target, Input.X1),
                x1.grad().data(),
                2e-3f);
        assertArrayEquals(
                finiteDifferenceGradient(loss, x1Data, x2Data, target, Input.X2),
                x2.grad().data(),
                2e-3f);
    }

    @Test
    void inactiveNegativePairsProduceZeroGradients() {
        CosineEmbeddingLoss loss = new CosineEmbeddingLoss(0.0f);
        GradTensor x1 = GradTensor.of(new float[] {1.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor x2 = GradTensor.of(new float[] {-1.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor target = GradTensor.of(new float[] {-1.0f}, 1);

        loss.compute(x1, x2, target).backward();

        assertArrayEquals(new float[] {0.0f, 0.0f}, x1.grad().data(), 1e-7f);
        assertArrayEquals(new float[] {0.0f, 0.0f}, x2.grad().data(), 1e-7f);
    }

    @Test
    void cosineEmbeddingLossRejectsInvalidConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class, () -> new CosineEmbeddingLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new CosineEmbeddingLoss(-1.1f));
        assertThrows(IllegalArgumentException.class, () -> new CosineEmbeddingLoss(1.1f));

        CosineEmbeddingLoss loss = new CosineEmbeddingLoss();
        GradTensor embeddings = GradTensor.zeros(2, 3);

        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2), GradTensor.zeros(2), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(embeddings, GradTensor.zeros(2, 4), GradTensor.ones(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(0, 3), GradTensor.zeros(0, 3), GradTensor.zeros(0)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(embeddings, embeddings, GradTensor.zeros(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(embeddings, embeddings, GradTensor.of(new float[] {1.0f, 0.0f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(embeddings, embeddings, GradTensor.of(new float[] {1.0f, Float.NaN}, 2)));
    }

    private static float[] finiteDifferenceGradient(
            CosineEmbeddingLoss loss,
            float[] x1Data,
            float[] x2Data,
            GradTensor target,
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

            grad[i] = (loss.compute(
                            GradTensor.of(x1Plus, 3, 2),
                            GradTensor.of(x2Plus, 3, 2),
                            target).item()
                    - loss.compute(
                            GradTensor.of(x1Minus, 3, 2),
                            GradTensor.of(x2Minus, 3, 2),
                            target).item()) / (2f * eps);
        }
        return grad;
    }

    private enum Input {
        X1,
        X2
    }
}
