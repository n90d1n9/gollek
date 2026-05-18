package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.Arrays;

/**
 * Triplet Loss — trains embeddings so that an anchor is closer to a positive
 * sample than to a negative sample by at least a margin.
 *
 * <p>
 * Based on <em>"FaceNet: A Unified Embedding for Face Recognition"</em>
 * (Schroff et al., 2015).
 *
 * <p>
 * Loss formula:
 * 
 * <pre>
 *   L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
 * </pre>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new TripletLoss(margin = 0.2f);
 * GradTensor l = loss.forward(anchor, positive, negative);
 * }</pre>
 */
public final class TripletLoss {

    private final float margin;

    /**
     * Creates a triplet loss with the given margin.
     *
     * @param margin minimum distance gap between positive and negative pairs
     *               (default 0.2)
     */
    public TripletLoss(float margin) {
        if (!Float.isFinite(margin) || margin < 0.0f) {
            throw new IllegalArgumentException("margin must be finite and non-negative, got: " + margin);
        }
        this.margin = margin;
    }

    /** Creates a triplet loss with default margin of 0.2. */
    public TripletLoss() {
        this(0.2f);
    }

    /**
     * Computes the triplet loss over a batch of anchor/positive/negative
     * embeddings.
     *
     * @param anchor   anchor embeddings {@code [N, D]}
     * @param positive positive embeddings {@code [N, D]} (same class as anchor)
     * @param negative negative embeddings {@code [N, D]} (different class)
     * @return scalar mean triplet loss
     */
    public GradTensor forward(GradTensor anchor, GradTensor positive, GradTensor negative) {
        long[] anchorShape = anchor.shape();
        long[] positiveShape = positive.shape();
        long[] negativeShape = negative.shape();
        requireEmbeddingShapes(anchorShape, positiveShape, negativeShape);

        int N = (int) anchorShape[0];
        int D = (int) anchorShape[1];
        float[] a = anchor.data(), p = positive.data(), n = negative.data();

        float[] losses = new float[N];
        boolean[] active = new boolean[N];
        for (int i = 0; i < N; i++) {
            float distAP = 0f, distAN = 0f;
            for (int d = 0; d < D; d++) {
                float dap = a[i * D + d] - p[i * D + d];
                float dan = a[i * D + d] - n[i * D + d];
                distAP += dap * dap;
                distAN += dan * dan;
            }
            float rawLoss = distAP - distAN + margin;
            losses[i] = Math.max(0f, rawLoss);
            active[i] = rawLoss > 0.0f;
        }
        GradTensor out = GradTensor.scalar(VectorOps.sum(losses) / N);
        if (anchor.requiresGrad() || positive.requiresGrad() || negative.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("TripletLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / N;
                    float[] gradAnchor = new float[a.length];
                    float[] gradPositive = new float[p.length];
                    float[] gradNegative = new float[n.length];

                    for (int i = 0; i < N; i++) {
                        if (!active[i]) {
                            continue;
                        }
                        int offset = i * D;
                        for (int d = 0; d < D; d++) {
                            int index = offset + d;
                            float anchorValue = a[index];
                            float positiveValue = p[index];
                            float negativeValue = n[index];

                            gradAnchor[index] = 2f * (negativeValue - positiveValue) * scale;
                            gradPositive[index] = 2f * (positiveValue - anchorValue) * scale;
                            gradNegative[index] = 2f * (anchorValue - negativeValue) * scale;
                        }
                    }

                    if (anchor.requiresGrad()) {
                        anchor.backward(GradTensor.of(gradAnchor, anchor.shape()));
                    }
                    if (positive.requiresGrad()) {
                        positive.backward(GradTensor.of(gradPositive, positive.shape()));
                    }
                    if (negative.requiresGrad()) {
                        negative.backward(GradTensor.of(gradNegative, negative.shape()));
                    }
                }
            });
        }
        return out;
    }

    private static void requireEmbeddingShapes(long[] anchorShape, long[] positiveShape, long[] negativeShape) {
        if (anchorShape.length != 2 || positiveShape.length != 2 || negativeShape.length != 2) {
            throw new IllegalArgumentException(
                    "anchor, positive, and negative must be 2D tensors [batch, dim], got: "
                            + Arrays.toString(anchorShape) + ", "
                            + Arrays.toString(positiveShape) + ", "
                            + Arrays.toString(negativeShape));
        }
        if (!Arrays.equals(anchorShape, positiveShape) || !Arrays.equals(anchorShape, negativeShape)) {
            throw new IllegalArgumentException(
                    "anchor, positive, and negative shapes must match, got: "
                            + Arrays.toString(anchorShape) + ", "
                            + Arrays.toString(positiveShape) + ", "
                            + Arrays.toString(negativeShape));
        }
        if (anchorShape[0] <= 0 || anchorShape[1] <= 0) {
            throw new IllegalArgumentException(
                    "anchor, positive, and negative must have positive batch and dim, got: "
                            + Arrays.toString(anchorShape));
        }
    }
}
