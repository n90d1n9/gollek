package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.Arrays;

/**
 * Dice Loss — overlap-based loss for image segmentation tasks.
 *
 * <p>
 * Optimizes the Dice coefficient (F1 score over pixels), which handles
 * class imbalance better than cross-entropy for segmentation.
 *
 * <p>
 * Formula:
 * 
 * <pre>
 *   Dice = 2 * |A ∩ B| / (|A| + |B|)
 *   L    = 1 - Dice
 * </pre>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new DiceLoss();
 * GradTensor l = loss.forward(predicted, target); // both [N, H, W] in [0,1]
 * }</pre>
 */
public final class DiceLoss {

    private final float smooth; // smoothing to avoid division by zero

    /**
     * Creates a Dice loss with default smoothing of 1.0.
     */
    public DiceLoss() {
        this(1.0f);
    }

    /**
     * Creates a Dice loss with custom smoothing.
     *
     * @param smooth Laplace smoothing constant (default 1.0)
     */
    public DiceLoss(float smooth) {
        if (!Float.isFinite(smooth) || smooth < 0.0f) {
            throw new IllegalArgumentException("smooth must be finite and non-negative, got: " + smooth);
        }
        this.smooth = smooth;
    }

    /**
     * Computes the Dice loss between predicted probabilities and binary targets.
     *
     * @param pred   predicted probabilities {@code [N, ...]} in range [0, 1]
     * @param target binary ground-truth mask {@code [N, ...]} with values 0 or 1
     * @return scalar Dice loss in range [0, 1]
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        long[] predShape = pred.shape();
        long[] targetShape = target.shape();
        if (!Arrays.equals(predShape, targetShape)) {
            throw new IllegalArgumentException(
                    "pred and target shapes must match, got: "
                            + Arrays.toString(predShape) + " vs " + Arrays.toString(targetShape));
        }
        float[] p = pred.data(), t = target.data();
        if (p.length == 0) {
            throw new IllegalArgumentException("pred must contain at least one element");
        }
        float[] targetData = new float[t.length];
        for (int i = 0; i < p.length; i++) {
            requireProbability(p[i], "pred", i);
            targetData[i] = requireBinaryTarget(t[i], i);
        }

        // intersection = sum(p * t), union = sum(p) + sum(t)
        float[] pt = new float[p.length];
        VectorOps.mul(p, targetData, pt);
        float intersection = VectorOps.sum(pt);
        float sumP = VectorOps.sum(p);
        float sumT = VectorOps.sum(targetData);

        float numerator = 2f * intersection + smooth;
        float denominator = sumP + sumT + smooth;
        if (denominator == 0.0f) {
            throw new IllegalArgumentException("Dice denominator is zero; use positive smoothing or non-empty masks");
        }
        float dice = numerator / denominator;
        GradTensor out = GradTensor.scalar(1f - dice);
        if (pred.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("DiceLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float upstreamScale = upstream.item();
                    float denomSquared = denominator * denominator;
                    float[] grad = new float[p.length];
                    for (int i = 0; i < p.length; i++) {
                        grad[i] = upstreamScale * (numerator - 2f * targetData[i] * denominator) / denomSquared;
                    }
                    pred.backward(GradTensor.of(grad, pred.shape()));
                }
            });
        }
        return out;
    }

    private static void requireProbability(float value, String name, int index) {
        if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
            throw new IllegalArgumentException(
                    name + " values must be finite probabilities in [0, 1], got "
                            + value + " at index " + index);
        }
    }

    private static float requireBinaryTarget(float target, int index) {
        if (Math.abs(target) <= 1e-6f) {
            return 0.0f;
        }
        if (Math.abs(target - 1.0f) <= 1e-6f) {
            return 1.0f;
        }
        throw new IllegalArgumentException(
                "target values must contain only 0.0 or 1.0, got " + target + " at index " + index);
    }
}
