package tech.kayys.gollek.ml.nn.loss;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class CausalLanguageModelingLossTest {

    @Test
    void computesMeanNextTokenLossAndSkipsIgnoredLabels() {
        float[] logitsData = {
                2f, 0f, -1f,
                0f, 1f, 3f,
                5f, 5f, 5f,
                -1f, 0f, 1f,
                9f, -9f, 3f,
                0f, 0f, 0f
        };
        GradTensor logits = GradTensor.of(logitsData, 2, 3, 3).requiresGrad(true);
        GradTensor labels = GradTensor.of(new float[] {
                0f, 2f, -100f,
                2f, -100f, 1f
        }, 2, 3);

        CausalLanguageModelingLoss loss = new CausalLanguageModelingLoss();
        GradTensor out = loss.compute(logits, labels);
        out.backward();

        Expected expected = expected(logitsData, labels.data(), 3, -100f);
        assertEquals(expected.loss(), out.item(), 1e-6);
        assertArrayEquals(expected.gradient(), logits.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {0f, 0f, 0f}, Arrays.copyOfRange(logits.grad().data(), 6, 9), 1e-6f);
        assertArrayEquals(new float[] {0f, 0f, 0f}, Arrays.copyOfRange(logits.grad().data(), 12, 15), 1e-6f);
    }

    @Test
    void supportsCustomIgnoreIndex() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f,
                1f, -1f
        }, 1, 2, 2).requiresGrad(true);
        GradTensor labels = GradTensor.of(new float[] {1f, -1f}, 1, 2);

        CausalLanguageModelingLoss loss = new CausalLanguageModelingLoss(-1f);
        GradTensor out = loss.forward(logits, labels);
        out.backward();

        assertEquals(Math.log(2.0), out.item(), 1e-6);
        assertArrayEquals(new float[] {0.5f, -0.5f, 0f, 0f}, logits.grad().data(), 1e-6f);
        assertEquals(-1f, loss.ignoreIndex(), 1e-6f);
    }

    @Test
    void rejectsInvalidConfigurationAndInputs() {
        CausalLanguageModelingLoss loss = new CausalLanguageModelingLoss();

        assertThrows(IllegalArgumentException.class, () -> new CausalLanguageModelingLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(2, 3), GradTensor.zeros(2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(0, 1, 3), GradTensor.zeros(0, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 0, 3), GradTensor.zeros(1, 0)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 1, 0), GradTensor.zeros(1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 2, 3), GradTensor.zeros(2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 1, 3), GradTensor.of(new float[] {-100f}, 1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(GradTensor.zeros(1, 1, 3), GradTensor.of(new float[] {1.5f}, 1, 1)));
        assertThrows(IndexOutOfBoundsException.class,
                () -> loss.compute(GradTensor.zeros(1, 1, 3), GradTensor.of(new float[] {3f}, 1, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.compute(
                        GradTensor.of(new float[] {0f, Float.POSITIVE_INFINITY, 1f}, 1, 1, 3),
                        GradTensor.of(new float[] {0f}, 1, 1)));
    }

    private static Expected expected(float[] logits, float[] labels, int vocab, float ignoreIndex) {
        float[] grad = new float[logits.length];
        float totalLoss = 0.0f;
        int valid = 0;
        for (int token = 0; token < labels.length; token++) {
            if (Float.compare(labels[token], ignoreIndex) == 0) {
                continue;
            }
            int target = (int) labels[token];
            int offset = token * vocab;
            float[] probabilities = softmax(logits, offset, vocab);
            totalLoss -= (float) Math.log(probabilities[target] + 1e-8f);
            valid++;
            for (int c = 0; c < vocab; c++) {
                grad[offset + c] = probabilities[c];
            }
            grad[offset + target] -= 1f;
        }
        for (int i = 0; i < grad.length; i++) {
            grad[i] /= valid;
        }
        return new Expected(totalLoss / valid, grad);
    }

    private static float[] softmax(float[] logits, int offset, int vocab) {
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < vocab; c++) {
            max = Math.max(max, logits[offset + c]);
        }
        float sum = 0.0f;
        float[] out = new float[vocab];
        for (int c = 0; c < vocab; c++) {
            out[c] = (float) Math.exp(logits[offset + c] - max);
            sum += out[c];
        }
        for (int c = 0; c < vocab; c++) {
            out[c] /= sum;
        }
        return out;
    }

    private record Expected(float loss, float[] gradient) {
    }
}
