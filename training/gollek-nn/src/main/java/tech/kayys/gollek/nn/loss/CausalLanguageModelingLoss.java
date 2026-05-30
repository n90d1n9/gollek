package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;
import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Next-token cross-entropy loss for causal language-model training.
 *
 * <p>Logits are expected as {@code [batch, sequence, vocab]} and labels as
 * {@code [batch, sequence]}. Labels equal to {@link #DEFAULT_IGNORE_INDEX} are
 * skipped for both loss and gradients, matching the padding convention used by
 * language-model data collators.</p>
 */
public final class CausalLanguageModelingLoss {
    public static final float DEFAULT_IGNORE_INDEX = -100.0f;

    private final float ignoreIndex;

    public CausalLanguageModelingLoss() {
        this(DEFAULT_IGNORE_INDEX);
    }

    public CausalLanguageModelingLoss(float ignoreIndex) {
        if (!Float.isFinite(ignoreIndex)) {
            throw new IllegalArgumentException("ignoreIndex must be finite, got: " + ignoreIndex);
        }
        this.ignoreIndex = ignoreIndex;
    }

    public float ignoreIndex() {
        return ignoreIndex;
    }

    public GradTensor forward(GradTensor logits, GradTensor labels) {
        return compute(logits, labels);
    }

    public GradTensor compute(GradTensor logits, GradTensor labels) {
        ShapeSpec spec = requireShapes(logits, labels);
        float[] logitsData = logits.data();
        float[] labelsData = labels.data();
        float[] softmaxData = new float[logitsData.length];
        int[] targetClasses = new int[spec.batch() * spec.sequence()];
        Arrays.fill(targetClasses, -1);

        float totalLoss = 0.0f;
        int validTokens = 0;
        for (int b = 0; b < spec.batch(); b++) {
            for (int t = 0; t < spec.sequence(); t++) {
                int tokenIndex = b * spec.sequence() + t;
                float label = labelsData[tokenIndex];
                if (isIgnored(label)) {
                    continue;
                }
                int target = ClassIndexTargets.require(label, spec.vocab(), "token " + tokenIndex);
                int offset = tokenIndex * spec.vocab();
                softmax(logitsData, softmaxData, offset, spec.vocab());
                totalLoss -= (float) Math.log(softmaxData[offset + target] + 1e-8f);
                targetClasses[tokenIndex] = target;
                validTokens++;
            }
        }

        if (validTokens == 0) {
            throw new IllegalArgumentException("labels contain no valid causal language-modeling targets");
        }

        float meanLoss = totalLoss / validTokens;
        GradTensor out = GradTensor.scalar(meanLoss);
        if (logits.requiresGrad()) {
            out.requiresGrad(true);
            final int finalValidTokens = validTokens;
            out.setGradFn(new Function.Context("CausalLanguageModelingLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / finalValidTokens;
                    float[] grad = new float[logitsData.length];
                    for (int tokenIndex = 0; tokenIndex < targetClasses.length; tokenIndex++) {
                        int target = targetClasses[tokenIndex];
                        if (target < 0) {
                            continue;
                        }
                        int offset = tokenIndex * spec.vocab();
                        for (int c = 0; c < spec.vocab(); c++) {
                            grad[offset + c] = softmaxData[offset + c] * scale;
                        }
                        grad[offset + target] -= scale;
                    }
                    logits.backward(GradTensor.of(grad, logits.shape()));
                }
            });
        }
        return out;
    }

    private boolean isIgnored(float label) {
        return Float.compare(label, ignoreIndex) == 0;
    }

    private static void softmax(float[] logitsData, float[] softmaxData, int offset, int vocab) {
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < vocab; c++) {
            max = Math.max(max, requireFiniteLogit(logitsData[offset + c], offset + c));
        }

        float sumExp = 0.0f;
        for (int c = 0; c < vocab; c++) {
            float value = (float) Math.exp(logitsData[offset + c] - max);
            softmaxData[offset + c] = value;
            sumExp += value;
        }
        for (int c = 0; c < vocab; c++) {
            softmaxData[offset + c] /= sumExp;
        }
    }

    private static ShapeSpec requireShapes(GradTensor logits, GradTensor labels) {
        long[] logitsShape = logits.shape();
        if (logitsShape.length != 3) {
            throw new IllegalArgumentException(
                    "logits must be 3D [batch, sequence, vocab], got shape: " + Arrays.toString(logitsShape));
        }
        int batch = Math.toIntExact(logitsShape[0]);
        int sequence = Math.toIntExact(logitsShape[1]);
        int vocab = Math.toIntExact(logitsShape[2]);
        if (batch <= 0 || sequence <= 0 || vocab <= 0) {
            throw new IllegalArgumentException(
                    "logits must have positive batch, sequence, and vocab dimensions, got shape: "
                            + Arrays.toString(logitsShape));
        }

        long[] labelShape = labels.shape();
        if (labelShape.length != 2 || labelShape[0] != batch || labelShape[1] != sequence) {
            throw new IllegalArgumentException(
                    "labels must be 2D [batch, sequence] matching logits, got shape: "
                            + Arrays.toString(labelShape));
        }
        return new ShapeSpec(batch, sequence, vocab);
    }

    private static float requireFiniteLogit(float logit, int index) {
        if (!Float.isFinite(logit)) {
            throw new IllegalArgumentException("logits must be finite, got " + logit + " at index " + index);
        }
        return logit;
    }

    @Override
    public String toString() {
        return "CausalLanguageModelingLoss(ignoreIndex=" + ignoreIndex + ")";
    }

    private record ShapeSpec(int batch, int sequence, int vocab) {
    }
}
