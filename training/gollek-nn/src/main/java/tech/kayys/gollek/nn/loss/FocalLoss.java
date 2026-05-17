package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

import java.util.Arrays;
import java.util.Objects;

/**
 * Focal Loss — down-weights easy examples to focus training on hard ones.
 * Designed for class-imbalanced datasets (e.g. object detection).
 *
 * <p>
 * FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 *
 * <p>
 * This implementation expects multi-class logits shaped {@code [batch, classes]}
 * and class-index targets shaped {@code [batch]}. It supports either a scalar
 * alpha or per-class alpha weights.
 */
public class FocalLoss {

    private final float gamma;
    private final float alpha;
    private final float[] classWeights;

    public FocalLoss() {
        this(2.0f, 0.25f);
    }

    public FocalLoss(float gamma) {
        this(gamma, 1.0f);
    }

    public FocalLoss(float gamma, float alpha) {
        this.gamma = validateGamma(gamma);
        this.alpha = validateAlpha(alpha, "alpha");
        this.classWeights = null;
    }

    public FocalLoss(float gamma, float[] classWeights) {
        this.gamma = validateGamma(gamma);
        this.alpha = 1.0f;
        this.classWeights = validateClassWeights(classWeights);
    }

    /**
     * @param logits  raw model output [N, C] (before softmax)
     * @param targets class indices [N]
     */
    public GradTensor forward(GradTensor logits, GradTensor targets) {
        return compute(logits, targets);
    }

    /**
     * Compute focal loss for multi-class classification.
     *
     * @param logits raw model output [N, C] before softmax
     * @param targets class indices [N]
     * @return scalar loss tensor
     */
    public GradTensor compute(GradTensor logits, GradTensor targets) {
        long[] shape = logits.shape();
        if (shape.length != 2) {
            throw new IllegalArgumentException(
                    "logits must be 2D [batch, classes], got shape: " + Arrays.toString(shape));
        }
        int batch = (int) shape[0];
        int classes = (int) shape[1];
        requireClassWeightShape(classes);

        float[] logitsData = logits.data();
        float[] targetsData = targets.data();
        if (targetsData.length != batch) {
            throw new IllegalArgumentException(
                    "targets batch size must match logits batch size, got: "
                            + targetsData.length + " vs " + batch);
        }

        // Softmax per sample
        float[] probs = new float[batch * classes];
        for (int n = 0; n < batch; n++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                max = Math.max(max, logitsData[n * classes + c]);
            }
            float sum = 0;
            for (int c = 0; c < classes; c++) {
                int index = n * classes + c;
                probs[index] = (float) Math.exp(logitsData[index] - max);
                sum += probs[index];
            }
            for (int c = 0; c < classes; c++) {
                probs[n * classes + c] /= sum;
            }
        }

        // Focal loss per sample
        float totalLoss = 0.0f;
        int[] targetClasses = new int[batch];
        float[] targetProbabilities = new float[batch];
        float[] sampleAlphas = new float[batch];
        for (int n = 0; n < batch; n++) {
            int cls = ClassIndexTargets.require(targetsData[n], classes, n);
            float pt = clampProbability(probs[n * classes + cls]);
            float sampleAlpha = alphaFor(cls);
            targetClasses[n] = cls;
            targetProbabilities[n] = pt;
            sampleAlphas[n] = sampleAlpha;
            totalLoss -= sampleAlpha * focalFactor(pt) * (float) Math.log(pt);
        }

        GradTensor out = GradTensor.scalar(totalLoss / batch);
        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("FocalLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / batch;
                    float[] grad = new float[logitsData.length];

                    for (int n = 0; n < batch; n++) {
                        int target = targetClasses[n];
                        float pt = targetProbabilities[n];
                        float coeff = focalGradientCoefficient(pt, sampleAlphas[n]) * scale;
                        int offset = n * classes;
                        for (int c = 0; c < classes; c++) {
                            float indicatorMinusProbability = (c == target ? 1.0f : 0.0f) - probs[offset + c];
                            grad[offset + c] = coeff * indicatorMinusProbability;
                        }
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }

    private float focalFactor(float pt) {
        return (float) Math.pow(1.0f - pt, gamma);
    }

    private float focalGradientCoefficient(float pt, float sampleAlpha) {
        float oneMinusPt = 1.0f - pt;
        if (gamma == 0.0f) {
            return -sampleAlpha;
        }
        float focusing = (float) Math.pow(oneMinusPt, gamma);
        float focusingDerivative = gamma * pt * (float) Math.pow(oneMinusPt, gamma - 1.0f) * (float) Math.log(pt);
        return sampleAlpha * (focusingDerivative - focusing);
    }

    private float alphaFor(int targetClass) {
        return classWeights == null ? alpha : classWeights[targetClass];
    }

    private void requireClassWeightShape(int classes) {
        if (classWeights == null) {
            return;
        }
        if (classWeights.length != classes) {
            throw new IllegalArgumentException(
                    "classWeights length " + classWeights.length + " must match logits classes " + classes);
        }
    }

    private static float clampProbability(float probability) {
        return Math.min(Math.max(probability, 1e-7f), 1.0f - 1e-7f);
    }

    private static float validateGamma(float gamma) {
        if (!Float.isFinite(gamma) || gamma < 0.0f) {
            throw new IllegalArgumentException("gamma must be finite and non-negative, got: " + gamma);
        }
        return gamma;
    }

    private static float validateAlpha(float alpha, String name) {
        if (!Float.isFinite(alpha) || alpha <= 0.0f) {
            throw new IllegalArgumentException(name + " must be finite and positive, got: " + alpha);
        }
        return alpha;
    }

    private static float[] validateClassWeights(float[] weights) {
        Objects.requireNonNull(weights, "classWeights must not be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("classWeights must contain at least one value");
        }
        float[] copy = weights.clone();
        for (float weight : copy) {
            validateAlpha(weight, "classWeights");
        }
        return copy;
    }

    @Override
    public String toString() {
        if (classWeights != null) {
            return "FocalLoss(gamma=" + gamma + ", classWeights=" + Arrays.toString(classWeights) + ")";
        }
        return "FocalLoss(gamma=" + gamma + ", alpha=" + alpha + ")";
    }
}
