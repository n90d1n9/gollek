package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

import java.util.Arrays;

/**
 * Cosine Embedding Loss for similarity learning.
 * <p>
 * Encourages embeddings of similar pairs to be close and dissimilar pairs to be far apart
 * using cosine distance as the metric.
 * <p>
 * For positive pairs (y=1): {@code loss = 1 - cos(x1, x2)}
 * For negative pairs (y=-1): {@code loss = max(0, cos(x1, x2) - margin)}
 * <p>
 * Equivalent to {@code torch.nn.CosineEmbeddingLoss}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new CosineEmbeddingLoss(0.5f);
 * var x1 = GradTensor.randn(32, 128);
 * var x2 = GradTensor.randn(32, 128);
 * var targets = GradTensor.of(new float[]{1, -1, 1, ...}, 32);
 * var lossTensor = loss.compute(x1, x2, targets);
 * }</pre>
 */
public class CosineEmbeddingLoss {

    private final float margin;

    /**
     * Create a CosineEmbeddingLoss with default margin of 0.0.
     */
    public CosineEmbeddingLoss() {
        this(0.0f);
    }

    /**
     * Create a CosineEmbeddingLoss with specified margin.
     *
     * @param margin margin for dissimilar pairs (default: 0.0). For negative pairs,
     *               loss is max(0, cos(x1, x2) - margin)
     */
    public CosineEmbeddingLoss(float margin) {
        if (!Float.isFinite(margin) || margin < -1.0f || margin > 1.0f) {
            throw new IllegalArgumentException("margin must be finite and within [-1, 1], got: " + margin);
        }
        this.margin = margin;
    }

    /**
     * Compute cosine embedding loss.
     *
     * @param x1     first embedding tensor [batch, dim] with requiresGrad=true
     * @param x2     second embedding tensor [batch, dim] with requiresGrad=true
     * @param target labels tensor [batch]: 1 for similar pairs, -1 for dissimilar pairs
     * @return scalar loss tensor with gradient support for backpropagation
     *
     * @throws IllegalArgumentException if x1 and x2 shapes do not match
     * @throws IllegalArgumentException if target batch size does not match x1 batch size
     */
    public GradTensor compute(GradTensor x1, GradTensor x2, GradTensor target) {
        long[] s1 = x1.shape();
        long[] s2 = x2.shape();
        long[] targetShape = target.shape();
        requireShapes(s1, s2, targetShape);

        float[] d1 = x1.data(), d2 = x2.data();
        float[] targetData = target.data();
        int batch = (int) s1[0];
        int dim = (int) s1[1];

        float[] losses = new float[batch];
        float[] denominators = new float[batch];
        float[] norm1Sqrts = new float[batch];
        float[] norm2Sqrts = new float[batch];
        float[] sanitizedTargets = new float[batch];
        float totalLoss = 0;

        // Forward pass: compute loss for each sample in batch
        for (int b = 0; b < batch; b++) {
            int off = b * dim;
            float label = requireCosineTarget(targetData[b], b);
            float dot = 0, norm1 = 0, norm2 = 0;

            // Compute dot product and norms
            for (int i = 0; i < dim; i++) {
                float v1 = d1[off + i];
                float v2 = d2[off + i];
                dot += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }

            // Compute cosine similarity: cos(x1, x2) = dot / (||x1|| * ||x2||)
            float norm1Sqrt = (float) Math.sqrt(norm1);
            float norm2Sqrt = (float) Math.sqrt(norm2);
            float denominator = norm1Sqrt * norm2Sqrt + 1e-8f;
            float cosine = dot / denominator;
            sanitizedTargets[b] = label;
            denominators[b] = denominator;
            norm1Sqrts[b] = norm1Sqrt;
            norm2Sqrts[b] = norm2Sqrt;

            // Compute loss: 1 - cos for positive pairs, max(0, cos - margin) for negative
            if (label == 1.0f) {
                losses[b] = 1 - cosine;
            } else {
                losses[b] = Math.max(0, cosine - margin);
            }
            totalLoss += losses[b];
        }

        float avgLoss = totalLoss / batch;

        // Create output tensor with automatic differentiation support
        GradTensor out = GradTensor.scalar(avgLoss);

        // Register backward function for gradient computation
        if (x1.requiresGrad() || x2.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("CosineEmbeddingLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / batch;
                    float[] grad1 = new float[d1.length];
                    float[] grad2 = new float[d2.length];

                    // Compute gradients for each sample
                    for (int b = 0; b < batch; b++) {
                        int off = b * dim;
                        float dot = 0;
                        for (int i = 0; i < dim; i++) {
                            float v1 = d1[off + i];
                            float v2 = d2[off + i];
                            dot += v1 * v2;
                        }

                        // Compute gradient only if loss is non-zero
                        boolean isPositive = sanitizedTargets[b] == 1.0f;
                        boolean isActive = isPositive || (losses[b] > 0);

                        if (isActive) {
                            float denominator = denominators[b];
                            float denominatorSquared = denominator * denominator;
                            float norm1Sqrt = norm1Sqrts[b];
                            float norm2Sqrt = norm2Sqrts[b];
                            float lossSign = isPositive ? -1.0f : 1.0f;

                            for (int i = 0; i < dim; i++) {
                                float v1 = d1[off + i];
                                float v2 = d2[off + i];
                                float dDenomDx1 = norm1Sqrt == 0.0f ? 0.0f : norm2Sqrt * v1 / norm1Sqrt;
                                float dDenomDx2 = norm2Sqrt == 0.0f ? 0.0f : norm1Sqrt * v2 / norm2Sqrt;

                                float dCosineDx1 = (v2 * denominator - dot * dDenomDx1) / denominatorSquared;
                                float dCosineDx2 = (v1 * denominator - dot * dDenomDx2) / denominatorSquared;
                                grad1[off + i] = scale * lossSign * dCosineDx1;
                                grad2[off + i] = scale * lossSign * dCosineDx2;
                            }
                        }
                    }

                    if (x1.requiresGrad()) {
                        x1.backward(GradTensor.of(grad1, x1.shape()));
                    }
                    if (x2.requiresGrad()) {
                        x2.backward(GradTensor.of(grad2, x2.shape()));
                    }
                }
            });
        }

        return out;
    }

    private static void requireShapes(long[] x1Shape, long[] x2Shape, long[] targetShape) {
        if (x1Shape.length != 2 || x2Shape.length != 2) {
            throw new IllegalArgumentException(
                    "x1 and x2 must be 2D tensors [batch, dim], got: "
                            + Arrays.toString(x1Shape) + " and " + Arrays.toString(x2Shape));
        }
        if (!Arrays.equals(x1Shape, x2Shape)) {
            throw new IllegalArgumentException(
                    "x1 and x2 shapes must match, got: "
                            + Arrays.toString(x1Shape) + " vs " + Arrays.toString(x2Shape));
        }
        if (x1Shape[0] <= 0 || x1Shape[1] <= 0) {
            throw new IllegalArgumentException(
                    "x1 and x2 must have positive batch and dim, got: " + Arrays.toString(x1Shape));
        }
        if (targetShape.length != 1 || targetShape[0] != x1Shape[0]) {
            throw new IllegalArgumentException(
                    "target must be a 1D tensor [batch] matching embeddings, got: "
                            + Arrays.toString(targetShape) + " for embeddings " + Arrays.toString(x1Shape));
        }
    }

    private static float requireCosineTarget(float target, int index) {
        if (Math.abs(target - 1.0f) <= 1e-6f) {
            return 1.0f;
        }
        if (Math.abs(target + 1.0f) <= 1e-6f) {
            return -1.0f;
        }
        throw new IllegalArgumentException(
                "target must contain only 1.0 or -1.0, got " + target + " at index " + index);
    }

    /**
     * Get the margin value.
     *
     * @return the margin for dissimilar pairs
     */
    public float getMargin() {
        return margin;
    }

    @Override
    public String toString() {
        return "CosineEmbeddingLoss(margin=" + margin + ")";
    }
}
