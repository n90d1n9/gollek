package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Binary focal loss with logits for imbalanced binary and multi-label tasks.
 *
 * <p>
 * This combines sigmoid, binary cross-entropy, and focal weighting in one
 * numerically stable operation:
 *
 * <pre>
 * FL = -alpha_t * positive_weight_t * (1 - p_t)^gamma * log(p_t)
 * </pre>
 *
 * <p>
 * Logits and targets must have identical shapes. Targets must be binary
 * values. Per-label positive weights use the last tensor dimension, matching
 * {@link BCEWithLogitsLoss}.
 */
public class BinaryFocalWithLogitsLoss {
    private final float gamma;
    private final float alpha;
    private final float[] positiveWeights;

    public BinaryFocalWithLogitsLoss() {
        this(2.0f, 0.25f);
    }

    public BinaryFocalWithLogitsLoss(float gamma, float alpha) {
        this.gamma = validateGamma(gamma);
        this.alpha = validateAlpha(alpha);
        this.positiveWeights = null;
    }

    public BinaryFocalWithLogitsLoss(float gamma, float alpha, float positiveWeight) {
        this(gamma, alpha, new float[] { positiveWeight });
    }

    public BinaryFocalWithLogitsLoss(float gamma, float alpha, float[] positiveWeights) {
        this.gamma = validateGamma(gamma);
        this.alpha = validateAlpha(alpha);
        this.positiveWeights = validatePositiveWeights(positiveWeights);
    }

    public GradTensor forward(GradTensor logits, GradTensor targets) {
        return compute(logits, targets);
    }

    public GradTensor compute(GradTensor logits, GradTensor targets) {
        long[] lShape = logits.shape();
        long[] tShape = targets.shape();
        if (!Arrays.equals(lShape, tShape)) {
            throw new IllegalArgumentException(
                    "logits and targets shapes must match, got: "
                            + Arrays.toString(lShape) + " vs " + Arrays.toString(tShape));
        }
        requirePositiveWeightShape(lShape);

        float[] logitsData = logits.data();
        float[] targetsData = targets.data();
        int n = logitsData.length;
        if (n == 0) {
            throw new IllegalArgumentException("logits must contain at least one element");
        }

        float totalLoss = 0.0f;
        float[] probabilities = new float[n];
        float[] targetProbabilities = new float[n];
        float[] elementScales = new float[n];
        float[] targetSigns = new float[n];

        for (int i = 0; i < n; i++) {
            float y = requireBinaryTarget(targetsData[i]);
            float probability = sigmoid(logitsData[i]);
            boolean positive = y > 0.5f;
            float pt = clampProbability(positive ? probability : 1.0f - probability);
            float alphaT = positive ? alpha : 1.0f - alpha;
            float positiveWeight = positive ? positiveWeightFor(i, lShape) : 1.0f;
            float scale = alphaT * positiveWeight;

            probabilities[i] = probability;
            targetProbabilities[i] = pt;
            elementScales[i] = scale;
            targetSigns[i] = positive ? 1.0f : -1.0f;

            totalLoss -= scale * focalFactor(pt) * (float) Math.log(pt);
        }

        GradTensor out = GradTensor.scalar(totalLoss / n);
        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("BinaryFocalWithLogitsLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float upstreamScale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        float p = probabilities[i];
                        float pt = targetProbabilities[i];
                        float dLossDpt = focalDerivative(pt) * elementScales[i];
                        float dPtDLogit = targetSigns[i] * p * (1.0f - p);
                        grad[i] = dLossDpt * dPtDLogit * upstreamScale;
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }

    private float positiveWeightFor(int elementIndex, long[] logitsShape) {
        if (positiveWeights == null) {
            return 1.0f;
        }
        if (positiveWeights.length == 1) {
            return positiveWeights[0];
        }
        return positiveWeights[positiveWeightLabelIndex(elementIndex, logitsShape)];
    }

    private void requirePositiveWeightShape(long[] logitsShape) {
        if (positiveWeights == null || positiveWeights.length == 1) {
            return;
        }
        int labelDimension = logitsShape.length == 0 ? 1 : (int) logitsShape[logitsShape.length - 1];
        if (positiveWeights.length != labelDimension) {
            throw new IllegalArgumentException(
                    "positiveWeights length " + positiveWeights.length
                            + " must be 1 or match logits label dimension " + labelDimension);
        }
    }

    private float focalFactor(float pt) {
        return (float) Math.pow(1.0f - pt, gamma);
    }

    private float focalDerivative(float pt) {
        if (gamma == 0.0f) {
            return -1.0f / pt;
        }
        float oneMinusPt = 1.0f - pt;
        float focusing = (float) Math.pow(oneMinusPt, gamma);
        float focusingDerivative = gamma
                * (float) Math.pow(oneMinusPt, gamma - 1.0f)
                * (float) Math.log(pt);
        return focusingDerivative - (focusing / pt);
    }

    private static int positiveWeightLabelIndex(int elementIndex, long[] shape) {
        if (shape.length == 0) {
            return 0;
        }
        int lastDimension = (int) shape[shape.length - 1];
        if (lastDimension <= 0) {
            return 0;
        }
        return elementIndex % lastDimension;
    }

    private static float sigmoid(float logit) {
        if (logit >= 0.0f) {
            float expNeg = (float) Math.exp(-logit);
            return 1.0f / (1.0f + expNeg);
        }
        float exp = (float) Math.exp(logit);
        return exp / (1.0f + exp);
    }

    private static float clampProbability(float probability) {
        return Math.min(Math.max(probability, 1e-7f), 1.0f - 1e-7f);
    }

    private static float requireBinaryTarget(float target) {
        if (Math.abs(target) <= 1e-6f) {
            return 0.0f;
        }
        if (Math.abs(target - 1.0f) <= 1e-6f) {
            return 1.0f;
        }
        throw new IllegalArgumentException("targets must contain only 0.0 or 1.0, got: " + target);
    }

    private static float validateGamma(float gamma) {
        if (!Float.isFinite(gamma) || gamma < 0.0f) {
            throw new IllegalArgumentException("gamma must be finite and non-negative, got: " + gamma);
        }
        return gamma;
    }

    private static float validateAlpha(float alpha) {
        if (!Float.isFinite(alpha) || alpha <= 0.0f || alpha >= 1.0f) {
            throw new IllegalArgumentException("alpha must be finite and in the open interval (0, 1), got: " + alpha);
        }
        return alpha;
    }

    private static float[] validatePositiveWeights(float[] weights) {
        Objects.requireNonNull(weights, "positiveWeights must not be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("positiveWeights must contain at least one value");
        }
        float[] copy = weights.clone();
        for (float weight : copy) {
            if (!Float.isFinite(weight) || weight <= 0.0f) {
                throw new IllegalArgumentException("positiveWeights must be finite and positive, got: " + weight);
            }
        }
        return copy;
    }

    @Override
    public String toString() {
        if (positiveWeights != null) {
            return "BinaryFocalWithLogitsLoss(gamma=" + gamma
                    + ", alpha=" + alpha
                    + ", positiveWeights=" + Arrays.toString(positiveWeights)
                    + ")";
        }
        return "BinaryFocalWithLogitsLoss(gamma=" + gamma + ", alpha=" + alpha + ")";
    }
}
