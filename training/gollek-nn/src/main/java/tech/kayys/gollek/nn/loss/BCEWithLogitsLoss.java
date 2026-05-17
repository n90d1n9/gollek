package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

import java.util.Arrays;
import java.util.Objects;

/**
 * Binary Cross-Entropy (BCE) Loss with Logits for binary classification.
 * <p>
 * BCEWithLogitsLoss combines a sigmoid activation and binary cross-entropy loss in one operation
 * for numerical stability. It expects raw logits (not sigmoid probabilities) as input.
 * <p>
 * Optimal for binary classification tasks (two classes: 0 or 1).
 * <p>
 * {@code BCEWithLogits = sigmoid(logit) ⊕ binary_ce(sigmoid(logit), target)}
 * <p>
 * Equivalent to {@code torch.nn.BCEWithLogitsLoss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * σ(x) = 1 / (1 + exp(-x))
 *
 * Loss = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Logits:</b> arbitrary shape (NOT sigmoid probabilities!)</li>
 *   <li><b>Targets:</b> same shape as logits, values in {0, 1}</li>
 *   <li><b>Output:</b> scalar loss value</li>
 * </ul>
 *
 * <h3>Example: Binary Classification</h3>
 * <pre>{@code
 * var loss = new BCEWithLogitsLoss();
 *
 * // Raw logits (not sigmoid!)
 * var logits = GradTensor.of(new float[]{2.0f, -1.0f, 0.5f, -2.0f}, 4);
 * var targets = GradTensor.of(new float[]{1.0f, 0.0f, 1.0f, 0.0f}, 4);
 *
 * var lossValue = loss.compute(logits, targets);
 * lossValue.backward();
 * }</pre>
 *
 * <h3>Example: Multi-label Classification</h3>
 * <pre>{@code
 * var loss = new BCEWithLogitsLoss();
 *
 * // Multiple independent binary classifications
 * // shape: [batch=2, num_classes=5]
 * var logits = GradTensor.randn(2, 5);
 * var targets = GradTensor.of(new float[][]{
 *     {1, 0, 1, 0, 1},   // sample 1: has classes 0, 2, 4
 *     {0, 1, 1, 1, 0}    // sample 2: has classes 1, 2, 3
 * }, 2, 5);
 *
 * var lossValue = loss.compute(logits, targets);
 * }</pre>
 *
 * <h3>Numerical Stability Features</h3>
 * <ul>
 *   <li>Sigmoid and BCE combined to prevent overflow</li>
 *   <li>Uses max() trick internally to avoid exp() overflow</li>
 *   <li>Stable computation for both small and large logit values</li>
 * </ul>
 *
 * <h3>Key Characteristics</h3>
 * <ul>
 *   <li>For binary classification (2 classes)</li>
 *   <li>Can also handle multi-label classification (multiple classes per sample)</li>
 *   <li>Expects logits, not probabilities</li>
 *   <li>Symmetric: treats false positives and false negatives equally</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Binary classification tasks (positive/negative, yes/no)</li>
 *   <li>Multi-label classification (image can have multiple tags)</li>
 *   <li>Anomaly detection (normal vs anomalous)</li>
 *   <li>Preference instead of CrossEntropyLoss for 2-class problems</li>
 * </ul>
 *
 * <h3>Comparison</h3>
 * <ul>
 *   <li>vs CrossEntropyLoss: CE for multi-class, BCE for binary/multi-label</li>
 *   <li>vs BCELoss: WithLogits is numerically more stable</li>
 *   <li>vs FocalLoss: FocalLoss handles class imbalance better</li>
 * </ul>
 *
 * <h3>Important Notes</h3>
 * <ul>
 *   <li><b>Input format:</b> Raw logits, NOT sigmoid(logits)</li>
 *   <li><b>Target format:</b> Float values 0.0 or 1.0 (not {0, 1} ints)</li>
 *   <li><b>Shape matching:</b> Logits and targets must have identical shapes</li>
 * </ul>
 */
public class BCEWithLogitsLoss {
    private final float[] positiveWeights;

    public BCEWithLogitsLoss() {
        this.positiveWeights = null;
    }

    public BCEWithLogitsLoss(float positiveWeight) {
        this(new float[] { positiveWeight });
    }

    public BCEWithLogitsLoss(float[] positiveWeights) {
        this.positiveWeights = validatePositiveWeights(positiveWeights);
    }

    /**
     * Compute Binary Cross-Entropy loss with logits.
     *
     * @param logits  raw model scores (NOT sigmoid probabilities), arbitrary shape
     * @param targets ground truth binary values {0, 1}, same shape as logits
     * @return scalar loss tensor
     *
     * @throws IllegalArgumentException if shapes do not match
     * @throws IllegalArgumentException if target values are not in [0, 1]
     */
    public GradTensor compute(GradTensor logits, GradTensor targets) {
        long[] lShape = logits.shape();
        long[] tShape = targets.shape();

        if (!java.util.Arrays.equals(lShape, tShape)) {
            throw new IllegalArgumentException(
                "logits and targets shapes must match, got: " + java.util.Arrays.toString(lShape) +
                " vs " + java.util.Arrays.toString(tShape));
        }

        float[] lData = logits.data();
        float[] tData = targets.data();
        int n = lData.length;
        if (n == 0) {
            throw new IllegalArgumentException("logits must contain at least one element");
        }
        requirePositiveWeightShape(lShape);

        // Validate target values
        float[] sanitizedTargets = new float[n];
        for (int i = 0; i < n; i++) {
            sanitizedTargets[i] = requireBinaryTarget(tData[i]);
        }

        float totalLoss = 0;
        float[] sigmoidData = new float[n];
        float[] elementWeights = new float[n];

        // Compute sigmoid and BCE
        for (int i = 0; i < n; i++) {
            float logit = lData[i];
            // Numerically stable sigmoid: σ(x) = 1/(1 + exp(-x))
            // Using: σ(x) = exp(x)/(1 + exp(x)) for x > 0, else 1/(1 + exp(-x))
            float sigmoid;
            if (logit >= 0) {
                float expX = (float) Math.exp(-logit);
                sigmoid = 1.0f / (1.0f + expX);
            } else {
                float expX = (float) Math.exp(logit);
                sigmoid = expX / (1.0f + expX);
            }
            sigmoidData[i] = sigmoid;

            float y = sanitizedTargets[i];
            float positiveWeight = positiveWeightFor(i, lShape);
            elementWeights[i] = y > 0.5f ? positiveWeight : 1.0f;
            totalLoss += y > 0.5f
                    ? positiveWeight * softplus(-logit)
                    : softplus(logit);
        }

        float meanLoss = totalLoss / n;
        GradTensor out = GradTensor.scalar(meanLoss);

        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("BCEWithLogitsLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];

                    // Gradient: (σ(x) - y)
                    for (int i = 0; i < n; i++) {
                        grad[i] = (sigmoidData[i] - sanitizedTargets[i]) * elementWeights[i] * scale;
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

    private static float softplus(float value) {
        if (value > 20.0f) {
            return value;
        }
        if (value < -20.0f) {
            return (float) Math.exp(value);
        }
        return (float) Math.log1p(Math.exp(value));
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
        return positiveWeights == null
                ? "BCEWithLogitsLoss()"
                : "BCEWithLogitsLoss(positiveWeights=" + Arrays.toString(positiveWeights) + ")";
    }
}
