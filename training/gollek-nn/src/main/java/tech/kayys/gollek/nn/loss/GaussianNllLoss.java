package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Gaussian negative log-likelihood for heteroscedastic regression.
 *
 * <p>
 * Predictions must end with a two-value distribution dimension:
 * {@code [mean, logVariance]}. Targets match the prediction shape without that
 * final distribution dimension. The loss is useful when a model should learn
 * both point estimates and data-dependent uncertainty.
 */
public final class GaussianNllLoss {

    private static final float LOG_TWO_PI = (float) Math.log(2.0 * Math.PI);

    private final boolean includeConstant;
    private final float minLogVariance;
    private final float maxLogVariance;

    /** Creates Gaussian NLL without the additive {@code 0.5 * log(2*pi)} constant. */
    public GaussianNllLoss() {
        this(false);
    }

    public GaussianNllLoss(boolean includeConstant) {
        this(includeConstant, -30.0f, 20.0f);
    }

    public GaussianNllLoss(boolean includeConstant, float minLogVariance, float maxLogVariance) {
        if (!Float.isFinite(minLogVariance) || !Float.isFinite(maxLogVariance) || minLogVariance >= maxLogVariance) {
            throw new IllegalArgumentException(
                    "minLogVariance must be finite and less than maxLogVariance, got: "
                            + minLogVariance + " >= " + maxLogVariance);
        }
        this.includeConstant = includeConstant;
        this.minLogVariance = minLogVariance;
        this.maxLogVariance = maxLogVariance;
    }

    public GradTensor forward(GradTensor predictions, GradTensor targets) {
        return compute(predictions, targets);
    }

    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        ShapeSpec spec = requireCompatibleFiniteNonEmpty(predictions, targets);
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();

        float total = 0.0f;
        for (int i = 0; i < spec.targetCount; i++) {
            float mean = predictionData[2 * i];
            float logVariance = clippedLogVariance(predictionData[2 * i + 1]);
            float diff = targetData[i] - mean;
            float inverseVariance = (float) Math.exp(-logVariance);
            total += 0.5f * (logVariance + diff * diff * inverseVariance);
            if (includeConstant) {
                total += 0.5f * LOG_TWO_PI;
            }
        }

        GradTensor out = GradTensor.scalar(total / spec.targetCount);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("GaussianNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.targetCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int i = 0; i < spec.targetCount; i++) {
                        float rawLogVariance = predictionData[2 * i + 1];
                        float logVariance = clippedLogVariance(rawLogVariance);
                        float mean = predictionData[2 * i];
                        float diff = targetData[i] - mean;
                        float inverseVariance = (float) Math.exp(-logVariance);
                        grad[2 * i] = (mean - targetData[i]) * inverseVariance * scale;
                        grad[2 * i + 1] = logVarianceGradient(rawLogVariance, diff, inverseVariance) * scale;
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public boolean includeConstant() {
        return includeConstant;
    }

    public float minLogVariance() {
        return minLogVariance;
    }

    public float maxLogVariance() {
        return maxLogVariance;
    }

    @Override
    public String toString() {
        return "GaussianNllLoss(includeConstant=" + includeConstant
                + ", minLogVariance=" + minLogVariance
                + ", maxLogVariance=" + maxLogVariance + ")";
    }

    private float clippedLogVariance(float logVariance) {
        if (logVariance < minLogVariance) {
            return minLogVariance;
        }
        if (logVariance > maxLogVariance) {
            return maxLogVariance;
        }
        return logVariance;
    }

    private float logVarianceGradient(float rawLogVariance, float diff, float inverseVariance) {
        if (rawLogVariance < minLogVariance || rawLogVariance > maxLogVariance) {
            return 0.0f;
        }
        return 0.5f * (1.0f - diff * diff * inverseVariance);
    }

    private ShapeSpec requireCompatibleFiniteNonEmpty(GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 2) {
            throw new IllegalArgumentException(
                    "GaussianNllLoss predictions must end with mean/logVariance dimension 2, got: "
                            + Arrays.toString(predictionShape));
        }
        long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
        if (!Arrays.equals(expectedTargetShape, targetShape)) {
            throw new IllegalArgumentException(
                    "GaussianNllLoss targets must match predictions without the final distribution dimension, got: "
                            + Arrays.toString(targetShape)
                            + " expected: " + Arrays.toString(expectedTargetShape));
        }

        int predictionCount = predictions.data().length;
        int targetCount = targets.data().length;
        if (targetCount == 0) {
            throw new IllegalArgumentException("GaussianNllLoss requires at least one target");
        }
        for (int i = 0; i < predictionCount; i++) {
            if (!Float.isFinite(predictions.data()[i])) {
                throw new IllegalArgumentException(
                        "GaussianNllLoss predictions must be finite, got "
                                + predictions.data()[i] + " at index " + i);
            }
        }
        for (int i = 0; i < targetCount; i++) {
            if (!Float.isFinite(targets.data()[i])) {
                throw new IllegalArgumentException(
                        "GaussianNllLoss targets must be finite, got " + targets.data()[i] + " at index " + i);
            }
        }
        return new ShapeSpec(predictionCount, targetCount);
    }

    private record ShapeSpec(int predictionCount, int targetCount) {
    }
}
