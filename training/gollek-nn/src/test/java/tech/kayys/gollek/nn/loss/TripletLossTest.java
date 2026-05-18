package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TripletLossTest {

    @Test
    void tripletLossBackpropagatesActiveTripletGradients() {
        TripletLoss loss = new TripletLoss(1.0f);
        float[] anchorData = new float[] {
                0.0f, 0.0f,
                1.0f, 1.0f
        };
        float[] positiveData = new float[] {
                0.5f, 0.2f,
                1.2f, 1.4f
        };
        float[] negativeData = new float[] {
                0.8f, 0.1f,
                1.5f, 1.2f
        };
        GradTensor anchor = GradTensor.of(anchorData, 2, 2).requiresGrad(true);
        GradTensor positive = GradTensor.of(positiveData, 2, 2).requiresGrad(true);
        GradTensor negative = GradTensor.of(negativeData, 2, 2).requiresGrad(true);

        GradTensor output = loss.forward(anchor, positive, negative);
        output.backward();

        assertArrayEquals(
                finiteDifferenceGradient(loss, anchorData, positiveData, negativeData, Input.ANCHOR),
                anchor.grad().data(),
                2e-3f);
        assertArrayEquals(
                finiteDifferenceGradient(loss, anchorData, positiveData, negativeData, Input.POSITIVE),
                positive.grad().data(),
                2e-3f);
        assertArrayEquals(
                finiteDifferenceGradient(loss, anchorData, positiveData, negativeData, Input.NEGATIVE),
                negative.grad().data(),
                2e-3f);
    }

    @Test
    void inactiveTripletsProduceZeroGradients() {
        TripletLoss loss = new TripletLoss(0.2f);
        GradTensor anchor = GradTensor.of(new float[] {0.0f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor positive = GradTensor.of(new float[] {0.1f, 0.0f}, 1, 2).requiresGrad(true);
        GradTensor negative = GradTensor.of(new float[] {10.0f, 10.0f}, 1, 2).requiresGrad(true);

        loss.forward(anchor, positive, negative).backward();

        assertArrayEquals(new float[] {0.0f, 0.0f}, anchor.grad().data(), 1e-7f);
        assertArrayEquals(new float[] {0.0f, 0.0f}, positive.grad().data(), 1e-7f);
        assertArrayEquals(new float[] {0.0f, 0.0f}, negative.grad().data(), 1e-7f);
    }

    @Test
    void tripletLossRejectsInvalidConfigurationAndShapes() {
        assertThrows(IllegalArgumentException.class, () -> new TripletLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new TripletLoss(-0.1f));

        TripletLoss loss = new TripletLoss();
        GradTensor embeddings = GradTensor.zeros(2, 3);

        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(2), GradTensor.zeros(2), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(embeddings, GradTensor.zeros(2, 4), embeddings));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(0, 3), GradTensor.zeros(0, 3), GradTensor.zeros(0, 3)));
    }

    private static float[] finiteDifferenceGradient(
            TripletLoss loss,
            float[] anchorData,
            float[] positiveData,
            float[] negativeData,
            Input input) {
        float[] source = switch (input) {
            case ANCHOR -> anchorData;
            case POSITIVE -> positiveData;
            case NEGATIVE -> negativeData;
        };
        float[] grad = new float[source.length];
        float eps = 1e-3f;
        for (int i = 0; i < source.length; i++) {
            float[] anchorPlus = anchorData.clone();
            float[] positivePlus = positiveData.clone();
            float[] negativePlus = negativeData.clone();
            float[] anchorMinus = anchorData.clone();
            float[] positiveMinus = positiveData.clone();
            float[] negativeMinus = negativeData.clone();

            switch (input) {
                case ANCHOR -> {
                    anchorPlus[i] += eps;
                    anchorMinus[i] -= eps;
                }
                case POSITIVE -> {
                    positivePlus[i] += eps;
                    positiveMinus[i] -= eps;
                }
                case NEGATIVE -> {
                    negativePlus[i] += eps;
                    negativeMinus[i] -= eps;
                }
            }

            grad[i] = (loss.forward(
                            GradTensor.of(anchorPlus, 2, 2),
                            GradTensor.of(positivePlus, 2, 2),
                            GradTensor.of(negativePlus, 2, 2)).item()
                    - loss.forward(
                            GradTensor.of(anchorMinus, 2, 2),
                            GradTensor.of(positiveMinus, 2, 2),
                            GradTensor.of(negativeMinus, 2, 2)).item()) / (2f * eps);
        }
        return grad;
    }

    private enum Input {
        ANCHOR,
        POSITIVE,
        NEGATIVE
    }
}
