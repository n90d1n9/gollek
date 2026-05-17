package tech.kayys.gollek.ml.autograd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GradTensorAutogradTest {

    @Test
    void compositeMseStyleExpressionBackpropagatesThroughPowAndMean() {
        GradTensor predictions = GradTensor.of(new float[] {2f, -1f, 0.5f}, 3).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {1f, 1f, -0.5f}, 3);

        GradTensor loss = predictions.sub(targets).pow(2f).mean();

        loss.backward();

        assertArrayEquals(new float[] {2f / 3f, -4f / 3f, 2f / 3f}, predictions.grad().data(), 1e-6f);
    }

    @Test
    void reluAndSumBackpropagateElementwiseMask() {
        GradTensor input = GradTensor.of(new float[] {-1f, 0f, 2f}, 3).requiresGrad(true);

        input.relu().sum().backward();

        assertArrayEquals(new float[] {0f, 0f, 1f}, input.grad().data(), 1e-6f);
    }

    @Test
    void sigmoidLogChainBackpropagatesAnalyticGradient() {
        GradTensor input = GradTensor.of(new float[] {0.2f, -0.4f}, 2).requiresGrad(true);

        input.sigmoid().log().sum().backward();

        float sigmoid0 = sigmoid(0.2f);
        float sigmoid1 = sigmoid(-0.4f);
        assertArrayEquals(new float[] {1f - sigmoid0, 1f - sigmoid1}, input.grad().data(), 1e-6f);
    }

    @Test
    void softmaxBackpropagatesJacobianVectorProductByLastDimension() {
        GradTensor logits = GradTensor.of(new float[] {
                1f, 2f, 3f,
                -1f, 0.5f, 2f
        }, 2, 3).requiresGrad(true);
        GradTensor softmax = logits.softmax();
        float[] upstream = new float[] {
                1f, 0f, 0f,
                0f, 2f, -1f
        };

        softmax.backward(GradTensor.of(upstream, 2, 3));

        assertArrayEquals(
                expectedSoftmaxGradient(softmax.data(), upstream, 3),
                logits.grad().data(),
                1e-6f);
    }

    @Test
    void reshapeAndDimReductionPreserveGradientFlow() {
        GradTensor input = GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3)
                .requiresGrad(true);

        input.reshape(3, 2).mean(1).sum().backward();

        assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, input.grad().data(), 1e-6f);
    }

    @Test
    void viewHelpersBackpropagateWithoutDroppingGradients() {
        GradTensor input = GradTensor.of(new float[] {1f, 2f, 3f}, 1, 3)
                .requiresGrad(true);

        input.unsqueeze(0).squeeze().flatten().sum().backward();

        assertArrayEquals(new float[] {1f, 1f, 1f}, input.grad().data(), 1e-6f);
    }

    @Test
    void catSplitsGradientBackToInputsAlongSelectedDimension() {
        GradTensor left = GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).requiresGrad(true);
        GradTensor right = GradTensor.of(new float[] {5f, 6f}, 2, 1).requiresGrad(true);

        GradTensor.cat(1, left, right)
                .backward(GradTensor.of(new float[] {
                        10f, 20f, 30f,
                        40f, 50f, 60f
                }, 2, 3));

        assertArrayEquals(new float[] {10f, 20f, 40f, 50f}, left.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {30f, 60f}, right.grad().data(), 1e-6f);
    }

    @Test
    void stackBackpropagatesThroughInsertedDimension() {
        GradTensor first = GradTensor.of(new float[] {1f, 2f}, 2).requiresGrad(true);
        GradTensor second = GradTensor.of(new float[] {3f, 4f}, 2).requiresGrad(true);

        GradTensor.stack(0, first, second)
                .backward(GradTensor.of(new float[] {
                        1f, 2f,
                        3f, 4f
                }, 2, 2));

        assertArrayEquals(new float[] {1f, 2f}, first.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {3f, 4f}, second.grad().data(), 1e-6f);
    }

    @Test
    void whereRoutesBroadcastGradientOnlyToSelectedInput() {
        GradTensor condition = GradTensor.of(new float[] {1f, 0f, 1f, 0f}, 2, 2);
        GradTensor x = GradTensor.of(new float[] {2f, 4f}, 1, 2).requiresGrad(true);
        GradTensor y = GradTensor.of(new float[] {10f, 20f, 30f, 40f}, 2, 2).requiresGrad(true);

        GradTensor.where(condition, x, y)
                .backward(GradTensor.of(new float[] {
                        1f, 2f,
                        3f, 4f
                }, 2, 2));

        assertArrayEquals(new float[] {4f, 0f}, x.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {0f, 2f, 0f, 4f}, y.grad().data(), 1e-6f);
    }

    @Test
    void scalarCatBackpropagatesToScalarInputs() {
        GradTensor first = GradTensor.scalar(2f).requiresGrad(true);
        GradTensor second = GradTensor.scalar(-3f).requiresGrad(true);

        GradTensor.cat(first, second).backward(GradTensor.of(new float[] {5f, -7f}, 2));

        assertArrayEquals(new float[] {5f}, first.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {-7f}, second.grad().data(), 1e-6f);
    }

    @Test
    void attentionScoreEinsumBackpropagatesToQueryAndKey() {
        GradTensor query = GradTensor.of(new float[] {
                1f, 2f,
                3f, 4f
        }, 1, 1, 2, 2).requiresGrad(true);
        GradTensor key = GradTensor.of(new float[] {
                5f, 6f,
                7f, 8f
        }, 1, 1, 2, 2).requiresGrad(true);

        query.einsum("bhid,bhjd->bhij", key)
                .backward(GradTensor.of(new float[] {
                        1f, 2f,
                        3f, 4f
                }, 1, 1, 2, 2));

        assertArrayEquals(new float[] {
                19f, 22f,
                43f, 50f
        }, query.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {
                10f, 14f,
                14f, 20f
        }, key.grad().data(), 1e-6f);
    }

    @Test
    void attentionApplyEinsumBackpropagatesToWeightsAndValue() {
        GradTensor weights = GradTensor.of(new float[] {
                1f, 2f,
                3f, 4f
        }, 1, 1, 2, 2).requiresGrad(true);
        GradTensor value = GradTensor.of(new float[] {
                5f, 6f,
                7f, 8f
        }, 1, 1, 2, 2).requiresGrad(true);

        weights.einsum("bhij,bhjd->bhid", value)
                .backward(GradTensor.of(new float[] {
                        1f, 2f,
                        3f, 4f
                }, 1, 1, 2, 2));

        assertArrayEquals(new float[] {
                17f, 23f,
                39f, 53f
        }, weights.grad().data(), 1e-6f);
        assertArrayEquals(new float[] {
                10f, 14f,
                14f, 20f
        }, value.grad().data(), 1e-6f);
    }

    private static float sigmoid(float value) {
        return 1f / (1f + (float) Math.exp(-value));
    }

    private static float[] expectedSoftmaxGradient(float[] softmax, float[] upstream, int width) {
        float[] grad = new float[softmax.length];
        int rows = softmax.length / width;
        for (int row = 0; row < rows; row++) {
            int base = row * width;
            float dot = 0f;
            for (int i = 0; i < width; i++) {
                dot += upstream[base + i] * softmax[base + i];
            }
            for (int i = 0; i < width; i++) {
                grad[base + i] = softmax[base + i] * (upstream[base + i] - dot);
            }
        }
        return grad;
    }
}
