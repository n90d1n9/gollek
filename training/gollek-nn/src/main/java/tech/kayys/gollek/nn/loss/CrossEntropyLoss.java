package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

import java.util.Arrays;
import java.util.Objects;

/**
 * Cross-entropy loss for multi-class classification tasks.
 * <p>
 * Cross-entropy is the de facto standard loss function for classification problems.
 * It combines log-softmax normalization and negative log-likelihood in a single
 * numerically stable computation. It penalizes confident incorrect predictions heavily.
 * <p>
 * The loss expects raw logits (not probabilities) as input and handles softmax
 * internally for numerical stability using the log-sum-exp trick.
 * <p>
 * Equivalent to {@code torch.nn.CrossEntropyLoss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * For each sample:
 *   LogSoftmax(x_c) = x_c - log(Σ_i exp(x_i))
 *   NLL = -LogSoftmax(x_target)
 *
 * Loss = (1/batch) Σ NLL
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Logits:</b> [batch, numClasses] - raw model outputs (NOT probabilities)</li>
 *   <li><b>Targets:</b> [batch] - class indices (0 to numClasses-1) as float values</li>
 *   <li><b>Output:</b> scalar loss value</li>
 * </ul>
 *
 * <h3>Example: Binary Classification</h3>
 * <pre>{@code
 * var loss = new CrossEntropyLoss();
 *
 * // Binary classification: 2 classes
 * var logits = GradTensor.of(new float[][]{
 *     {2.0f, -1.0f},    // sample 1: high score for class 0
 *     {-1.0f, 2.0f},    // sample 2: high score for class 1
 * }, 2, 2);
 *
 * var targets = GradTensor.of(new float[]{0, 1}, 2);  // true classes
 * var lossValue = loss.compute(logits, targets);      // ~0.126 (low loss = good)
 * }</pre>
 *
 * <h3>Example: Multi-class (MNIST-like)</h3>
 * <pre>{@code
 * var loss = new CrossEntropyLoss();
 *
 * // 10-class classification (MNIST)
 * var model = ...;  // neural network returning [batch, 10] logits
 * var predictions = model.forward(images);  // [32, 10] logits
 * var targets = GradTensor.of(new float[]{3, 7, 2, ...}, 32);  // 32 class labels
 *
 * var lossValue = loss.compute(predictions, targets);
 * lossValue.backward();
 * optimizer.step();
 * }</pre>
 *
 * <h3>Example: ImageNet Classification</h3>
 * <pre>{@code
 * // 1000-class classification
 * var logits = model.forward(images);  // [batch, 1000]
 * var loss = new CrossEntropyLoss();
 * var lossValue = loss.compute(logits, labels);
 * }</pre>
 *
 * <h3>Numerical Stability Features</h3>
 * <ul>
 *   <li><b>Log-sum-exp trick:</b> Subtracts max logit before exp to prevent overflow</li>
 *   <li><b>Epsilon:</b> Adds 1e-8 to log argument to prevent log(0)</li>
 *   <li><b>Stable backward:</b> Gradients computed from softmax outputs, not raw logits</li>
 * </ul>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Optimal for categorical probability distributions</li>
 *   <li>Penalizes confident incorrect predictions heavily</li>
 *   <li>Expected class probability targets: one class has prob 1.0, others 0.0</li>
 *   <li>Well-defined gradients everywhere</li>
 * </ul>
 *
 * <h3>Important Notes</h3>
 * <ul>
 *   <li><b>Input:</b> Use raw logits, NOT softmax probabilities</li>
 *   <li><b>Target format:</b> Integer class indices (as float), not one-hot vectors</li>
 *   <li><b>Batch dimension:</b> Always required, even for single samples</li>
 * </ul>
 *
 * <h3>Gradient Interpretation</h3>
 * The gradient w.r.t logits equals: softmax(logits) - one_hot(target)
 * This means the model learns by moving the softmax distribution towards one-hot.
 *
 * <h3>Alternatives</h3>
 * <ul>
 *   <li><b>BCEWithLogitsLoss:</b> For binary classification (2 classes)</li>
 *   <li><b>FocalLoss:</b> For imbalanced datasets</li>
 *   <li><b>SoftmaxCrossEntropyLoss:</b> For soft labels/label smoothing</li>
 * </ul>
 */
public class CrossEntropyLoss {
    private final float[] classWeights;

    public CrossEntropyLoss() {
        this.classWeights = null;
    }

    public CrossEntropyLoss(float[] classWeights) {
        this.classWeights = validateClassWeights(classWeights);
    }

    /**
     * Compute cross-entropy loss for classification.
     *
     * @param logits  raw model scores [batch, numClasses] (NOT probabilities)
     * @param targets class indices [batch] (integer values as floats, range: 0 to numClasses-1)
     * @return scalar loss tensor
     *
     * @throws IllegalArgumentException if logits shape is not 2D
     * @throws IllegalArgumentException if targets batch size doesn't match logits batch size
     * @throws IndexOutOfBoundsException if any target index is out of range [0, numClasses)
     */
    public GradTensor compute(GradTensor logits, GradTensor targets) {
        long[] s = logits.shape();
        if (s.length != 2) {
            throw new IllegalArgumentException(
                "logits must be 2D [batch, numClasses], got shape: " + Arrays.toString(s));
        }

        int batch = (int) s[0];
        int numClasses = (int) s[1];
        if (batch <= 0 || numClasses <= 0) {
            throw new IllegalArgumentException(
                    "logits must have positive batch and class dimensions, got shape: " + Arrays.toString(s));
        }
        requireClassWeightShape(numClasses);
        float[] logitsData = logits.data();
        float[] targetsData = ClassIndexTargets.requireVectorData(targets, batch, "targets");

        // Compute log-softmax and NLL
        float totalLoss = 0;
        float[] softmaxData = new float[logitsData.length];
        float[] sampleWeights = new float[batch];
        int[] targetClasses = new int[batch];
        float totalSampleWeight = 0.0f;

        // Process each sample in batch
        for (int b = 0; b < batch; b++) {
            int off = b * numClasses;

            // Log-sum-exp trick: subtract max for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < numClasses; c++) {
                max = Math.max(max, requireFiniteLogit(logitsData[off + c], off + c));
            }

            // Compute softmax with numerical stability
            float sumExp = 0;
            for (int c = 0; c < numClasses; c++) {
                softmaxData[off + c] = (float) Math.exp(logitsData[off + c] - max);
                sumExp += softmaxData[off + c];
            }
            for (int c = 0; c < numClasses; c++) {
                softmaxData[off + c] /= sumExp;
            }

            // Get target class and compute NLL
            int target = ClassIndexTargets.require(targetsData[b], numClasses, b);
            targetClasses[b] = target;
            float sampleWeight = classWeightFor(target);
            sampleWeights[b] = sampleWeight;
            totalSampleWeight += sampleWeight;
            // NLL = -log(softmax[target])
            totalLoss -= sampleWeight * (float) Math.log(softmaxData[off + target] + 1e-8f);
        }

        final float finalTotalSampleWeight = totalSampleWeight;
        float meanLoss = totalLoss / finalTotalSampleWeight;
        GradTensor out = GradTensor.scalar(meanLoss);

        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("CrossEntropyLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / finalTotalSampleWeight;
                    float[] grad = new float[logitsData.length];

                    // Gradient = softmax - one_hot(target)
                    for (int b = 0; b < batch; b++) {
                        int off = b * numClasses;
                        int target = targetClasses[b];
                        float sampleScale = sampleWeights[b] * scale;
                        for (int c = 0; c < numClasses; c++) {
                            grad[off + c] = softmaxData[off + c] * sampleScale;
                        }
                        // Subtract 1 for target class (one_hot encoding)
                        grad[off + target] -= sampleScale;
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }

    private float classWeightFor(int targetClass) {
        if (classWeights == null) {
            return 1.0f;
        }
        return classWeights[targetClass];
    }

    private void requireClassWeightShape(int numClasses) {
        if (classWeights == null) {
            return;
        }
        if (classWeights.length != numClasses) {
            throw new IllegalArgumentException(
                    "classWeights length " + classWeights.length + " must match logits numClasses " + numClasses);
        }
    }

    private static float[] validateClassWeights(float[] weights) {
        Objects.requireNonNull(weights, "classWeights must not be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("classWeights must contain at least one value");
        }
        float[] copy = weights.clone();
        for (float weight : copy) {
            if (!Float.isFinite(weight) || weight <= 0.0f) {
                throw new IllegalArgumentException("classWeights must be finite and positive, got: " + weight);
            }
        }
        return copy;
    }

    private static float requireFiniteLogit(float logit, int index) {
        if (!Float.isFinite(logit)) {
            throw new IllegalArgumentException("logits must be finite, got " + logit + " at index " + index);
        }
        return logit;
    }

    @Override
    public String toString() {
        return classWeights == null
                ? "CrossEntropyLoss()"
                : "CrossEntropyLoss(classWeights=" + Arrays.toString(classWeights) + ")";
    }
}
