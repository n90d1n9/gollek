package tech.kayys.gollek.ml.nn.backend;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CPUNNBackendTest {

    private final CPUNNBackend backend = new CPUNNBackend();

    @Test
    void depthwiseConv2dComputesPerChannelKernels() {
        GradTensor input = GradTensor.of(new float[] {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12,
                13, 14, 15,
                16, 17, 18
        }, 1, 2, 3, 3);
        GradTensor weight = GradTensor.of(new float[] {
                1, 0,
                0, 1,
                0, 1,
                1, 0
        }, 2, 1, 2, 2);
        GradTensor bias = GradTensor.of(new float[] {0.5f, -1.0f}, 2);

        GradTensor output = backend.depthwiseConv2d(input, weight, bias, 1, 0);

        assertArrayEquals(new long[] {1, 2, 2, 2}, output.shape());
        assertArrayEquals(new float[] {
                6.5f, 8.5f,
                12.5f, 14.5f,
                23.0f, 25.0f,
                29.0f, 31.0f
        }, output.data(), 1e-6f);
    }

    @Test
    void imageOpsResizeCropAndNormalize() {
        GradTensor image = GradTensor.of(new float[] {
                1, 2,
                3, 4,
                10, 20,
                30, 40
        }, 1, 2, 2, 2);

        GradTensor resized = backend.resize(image, 1, 1, "bilinear");
        assertArrayEquals(new long[] {1, 2, 1, 1}, resized.shape());
        assertArrayEquals(new float[] {2.5f, 25.0f}, resized.data(), 1e-6f);

        GradTensor cropped = backend.crop(image, 0, 1, 2, 1);
        assertArrayEquals(new long[] {1, 2, 2, 1}, cropped.shape());
        assertArrayEquals(new float[] {2.0f, 4.0f, 20.0f, 40.0f}, cropped.data(), 1e-6f);

        GradTensor normalized = backend.normalize(image, new float[] {1.0f, 10.0f}, new float[] {1.0f, 10.0f});
        assertArrayEquals(new float[] {
                0, 1,
                2, 3,
                0, 1,
                2, 3
        }, normalized.data(), 1e-6f);
    }

    @Test
    void attentionAppliesMaskAndStableSoftmax() {
        GradTensor query = GradTensor.of(new float[] {
                1, 0,
                0, 1
        }, 1, 2, 2);
        GradTensor key = GradTensor.of(new float[] {
                1, 0,
                0, 1
        }, 1, 2, 2);
        GradTensor value = GradTensor.of(new float[] {
                10, 0,
                0, 20
        }, 1, 2, 2);
        GradTensor mask = GradTensor.of(new float[] {
                1, 0,
                1, 1
        }, 2, 2);

        GradTensor output = backend.attention(query, key, value, mask, 1.0f);

        float w0 = (float) (1.0 / (1.0 + Math.E));
        float w1 = (float) (Math.E / (1.0 + Math.E));
        assertArrayEquals(new long[] {1, 2, 2}, output.shape());
        assertArrayEquals(new float[] {
                10.0f, 0.0f,
                10.0f * w0, 20.0f * w1
        }, output.data(), 1e-5f);
    }

    @Test
    void multiHeadAttentionSplitsHeadsAndBroadcastsBatchMask() {
        GradTensor query = GradTensor.of(new float[] {
                1, 0, 1, 1,
                0, 1, 1, -1
        }, 1, 2, 4);
        GradTensor key = GradTensor.of(new float[] {
                1, 0, 1, 1,
                0, 1, 1, -1
        }, 1, 2, 4);
        GradTensor value = GradTensor.of(new float[] {
                10, 0, 5, 1,
                0, 20, 1, 5
        }, 1, 2, 4);
        GradTensor mask = GradTensor.of(new float[] {
                1, 0,
                1, 1
        }, 1, 2, 2);

        GradTensor output = backend.multiHeadAttention(query, key, value, 2, mask);

        float scale = (float) (1.0 / Math.sqrt(2.0));
        float[] head0Weights = softmax(0.0f, scale);
        float[] head1Weights = softmax(0.0f, 2.0f * scale);
        assertArrayEquals(new long[] {1, 2, 4}, output.shape());
        assertArrayEquals(new float[] {
                10.0f, 0.0f, 5.0f, 1.0f,
                10.0f * head0Weights[0], 20.0f * head0Weights[1],
                5.0f * head1Weights[0] + 1.0f * head1Weights[1],
                1.0f * head1Weights[0] + 5.0f * head1Weights[1]
        }, output.data(), 1e-5f);
    }

    @Test
    void backendOpsRejectInvalidInputsInsteadOfReturningNull() {
        GradTensor image = GradTensor.zeros(1, 1, 2, 2);
        assertThrows(IllegalArgumentException.class,
                () -> backend.depthwiseConv2d(image, GradTensor.zeros(2, 1, 1, 1), null, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> backend.resize(image, 2, 2, "lanczos"));
        assertThrows(IllegalArgumentException.class,
                () -> backend.crop(image, 1, 1, 2, 2));
        assertThrows(IllegalArgumentException.class,
                () -> backend.normalize(image, new float[] {0.0f}, new float[] {0.0f}));
        assertThrows(IllegalArgumentException.class,
                () -> backend.multiHeadAttention(
                        GradTensor.zeros(1, 2, 3),
                        GradTensor.zeros(1, 2, 3),
                        GradTensor.zeros(1, 2, 3),
                        2,
                        null));

        assertEquals("cpu", backend.getBackendId());
    }

    private static float[] softmax(float a, float b) {
        float max = Math.max(a, b);
        float expA = (float) Math.exp(a - max);
        float expB = (float) Math.exp(b - max);
        float sum = expA + expB;
        return new float[] {expA / sum, expB / sum};
    }
}
