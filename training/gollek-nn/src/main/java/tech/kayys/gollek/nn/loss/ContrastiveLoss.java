package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.Arrays;

/**
 * Contrastive Loss — pulls together positive pairs and pushes apart negative
 * pairs.
 *
 * <p>
 * Used in self-supervised learning (SimCLR, MoCo, CLIP) and metric learning.
 *
 * <p>
 * Formula:
 * 
 * <pre>
 *   L = y · d² + (1-y) · max(0, margin - d)²
 *   where d = ||f(x₁) - f(x₂)||₂
 * </pre>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new ContrastiveLoss(margin = 1.0f);
 * // y=1 → same class (pull together), y=0 → different class (push apart)
 * GradTensor l = loss.forward(emb1, emb2, labels);
 * }</pre>
 */
public final class ContrastiveLoss {

    private final float margin;

    /** Creates a contrastive loss with default margin of 1.0. */
    public ContrastiveLoss() {
        this(1.0f);
    }

    /**
     * Creates a contrastive loss with custom margin.
     *
     * @param margin minimum distance for negative pairs (default 1.0)
     */
    public ContrastiveLoss(float margin) {
        if (!Float.isFinite(margin) || margin < 0.0f) {
            throw new IllegalArgumentException("margin must be finite and non-negative, got: " + margin);
        }
        this.margin = margin;
    }

    /**
     * Computes the contrastive loss over a batch of pairs.
     *
     * @param x1     first embeddings {@code [N, D]}
     * @param x2     second embeddings {@code [N, D]}
     * @param labels similarity labels {@code [N]}: 1=same class, 0=different class
     * @return scalar mean contrastive loss
     */
    public GradTensor forward(GradTensor x1, GradTensor x2, GradTensor labels) {
        long[] x1Shape = x1.shape();
        long[] x2Shape = x2.shape();
        long[] labelShape = labels.shape();
        requireShapes(x1Shape, x2Shape, labelShape);

        int N = (int) x1Shape[0], D = (int) x1Shape[1];
        float[] a = x1.data(), b = x2.data(), y = labels.data();
        float[] losses = new float[N];
        float[] distances = new float[N];
        boolean[] activeNegative = new boolean[N];
        float[] sanitizedLabels = new float[N];

        for (int n = 0; n < N; n++) {
            float distSq = 0f;
            for (int d = 0; d < D; d++) {
                float diff = a[n * D + d] - b[n * D + d];
                distSq += diff * diff;
            }
            float dist = (float) Math.sqrt(distSq);
            distances[n] = dist;
            float label = requireBinaryLabel(y[n], n);
            sanitizedLabels[n] = label;
            if (label == 1.0f) {
                losses[n] = distSq;
            } else {
                float marginViolation = Math.max(0f, margin - dist);
                losses[n] = marginViolation * marginViolation;
                activeNegative[n] = marginViolation > 0.0f && dist > 0.0f;
            }
        }
        GradTensor out = GradTensor.scalar(VectorOps.sum(losses) / N);
        if (x1.requiresGrad() || x2.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("ContrastiveLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / N;
                    float[] gradX1 = new float[a.length];
                    float[] gradX2 = new float[b.length];

                    for (int n = 0; n < N; n++) {
                        int offset = n * D;
                        float label = sanitizedLabels[n];
                        float distance = distances[n];
                        for (int d = 0; d < D; d++) {
                            int index = offset + d;
                            float diff = a[index] - b[index];
                            float grad;
                            if (label == 1.0f) {
                                grad = 2f * diff;
                            } else if (activeNegative[n]) {
                                grad = -2f * (margin - distance) * diff / distance;
                            } else {
                                grad = 0.0f;
                            }
                            gradX1[index] = grad * scale;
                            gradX2[index] = -grad * scale;
                        }
                    }

                    if (x1.requiresGrad()) {
                        x1.backward(GradTensor.of(gradX1, x1.shape()));
                    }
                    if (x2.requiresGrad()) {
                        x2.backward(GradTensor.of(gradX2, x2.shape()));
                    }
                }
            });
        }
        return out;
    }

    private static void requireShapes(long[] x1Shape, long[] x2Shape, long[] labelShape) {
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
        if (labelShape.length != 1 || labelShape[0] != x1Shape[0]) {
            throw new IllegalArgumentException(
                    "labels must be a 1D tensor [batch] matching embeddings, got: "
                            + Arrays.toString(labelShape) + " for embeddings " + Arrays.toString(x1Shape));
        }
    }

    private static float requireBinaryLabel(float label, int index) {
        if (Math.abs(label) <= 1e-6f) {
            return 0.0f;
        }
        if (Math.abs(label - 1.0f) <= 1e-6f) {
            return 1.0f;
        }
        throw new IllegalArgumentException(
                "labels must contain only 0.0 or 1.0, got " + label + " at index " + index);
    }
}
