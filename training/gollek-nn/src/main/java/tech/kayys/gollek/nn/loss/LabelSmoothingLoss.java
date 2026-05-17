package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;

/**
 * Label Smoothing Cross-Entropy Loss — regularizes classification by
 * softening hard one-hot targets toward a uniform distribution.
 *
 * <p>
 * Prevents overconfident predictions and improves calibration.
 * Used in image classification (ViT, EfficientNet) and NLP (BERT fine-tuning).
 *
 * <p>
 * Smoothed target: {@code y_smooth = (1-ε)·y_hard + ε/C}
 * where {@code ε} is the smoothing factor and {@code C} is the number of
 * classes.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new LabelSmoothingLoss(smoothing = 0.1f);
 * GradTensor l = loss.forward(logits, labels); // logits [N,C], labels [N]
 * }</pre>
 */
public final class LabelSmoothingLoss {

    private final float smoothing;

    /**
     * Creates a label smoothing loss.
     *
     * @param smoothing smoothing factor ε in [0, 1) (default 0.1)
     */
    public LabelSmoothingLoss(float smoothing) {
        if (!Float.isFinite(smoothing) || smoothing < 0.0f || smoothing >= 1.0f) {
            throw new IllegalArgumentException(
                    "smoothing must be finite and in [0, 1), got: " + smoothing);
        }
        this.smoothing = smoothing;
    }

    /** Creates a label smoothing loss with default ε=0.1. */
    public LabelSmoothingLoss() {
        this(0.1f);
    }

    /**
     * Computes the label-smoothed cross-entropy loss.
     *
     * @param logits raw model outputs {@code [N, C]}
     * @param labels integer class indices {@code [N]}
     * @return scalar mean loss
     */
    public GradTensor forward(GradTensor logits, GradTensor labels) {
        long[] s = logits.shape();
        if (s.length != 2) {
            throw new IllegalArgumentException(
                    "logits must be 2D [batch, classes], got shape: " + Arrays.toString(s));
        }
        int N = (int) s[0], C = (int) s[1];
        if (N <= 0 || C <= 0) {
            throw new IllegalArgumentException(
                    "logits must have positive batch and class dimensions, got shape: " + Arrays.toString(s));
        }
        float[] lg = logits.data(), lb = labels.data();
        if (lb.length != N) {
            throw new IllegalArgumentException(
                    "labels batch size must match logits batch size, got: " + lb.length + " vs " + N);
        }
        float[] softmaxData = new float[lg.length];
        int[] targetClasses = new int[N];
        float totalLoss = 0.0f;

        for (int n = 0; n < N; n++) {
            // Log-softmax for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++)
                max = Math.max(max, lg[n * C + c]);
            float sumExp = 0f;
            for (int c = 0; c < C; c++) {
                int index = n * C + c;
                softmaxData[index] = (float) Math.exp(lg[index] - max);
                sumExp += softmaxData[index];
            }
            for (int c = 0; c < C; c++)
                softmaxData[n * C + c] /= sumExp;
            float logSumExp = max + (float) Math.log(sumExp);

            int cls = ClassIndexTargets.require(lb[n], C, n);
            targetClasses[n] = cls;
            float hardLoss = logSumExp - lg[n * C + cls]; // CE for true class

            // Uniform loss over all classes (smoothing term)
            float uniformLoss = 0f;
            for (int c = 0; c < C; c++)
                uniformLoss += logSumExp - lg[n * C + c];
            uniformLoss /= C;

            totalLoss += (1f - smoothing) * hardLoss + smoothing * uniformLoss;
        }
        GradTensor out = GradTensor.scalar(totalLoss / N);
        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("LabelSmoothingLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / N;
                    float uniformTarget = smoothing / C;
                    float hardTarget = 1.0f - smoothing + uniformTarget;
                    float[] grad = new float[lg.length];

                    for (int n = 0; n < N; n++) {
                        int target = targetClasses[n];
                        int offset = n * C;
                        for (int c = 0; c < C; c++) {
                            float smoothedTarget = c == target ? hardTarget : uniformTarget;
                            grad[offset + c] = (softmaxData[offset + c] - smoothedTarget) * scale;
                        }
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }
}
