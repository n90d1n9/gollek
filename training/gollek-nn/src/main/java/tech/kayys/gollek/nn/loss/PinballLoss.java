package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Pinball loss for quantile regression.
 *
 * <p>
 * A quantile {@code q} penalizes under-prediction by {@code q} and
 * over-prediction by {@code 1 - q}. This lets a model learn medians,
 * prediction bands, or any requested conditional quantile instead of only a
 * conditional mean.
 */
public final class PinballLoss {

    private final float[] quantiles;

    /** Creates median quantile regression loss, equivalent to {@code q=0.5}. */
    public PinballLoss() {
        this(0.5f);
    }

    public PinballLoss(float quantile) {
        this(new float[] { quantile });
    }

    public PinballLoss(double quantile) {
        this((float) quantile);
    }

    public PinballLoss(float... quantiles) {
        this.quantiles = normalizeQuantiles(quantiles);
    }

    public PinballLoss(double... quantiles) {
        this.quantiles = normalizeQuantiles(toFloatArray(quantiles));
    }

    public static PinballLoss interval(float lowerQuantile, float upperQuantile) {
        if (lowerQuantile >= upperQuantile) {
            throw new IllegalArgumentException(
                    "lowerQuantile must be less than upperQuantile, got: "
                            + lowerQuantile + " >= " + upperQuantile);
        }
        return new PinballLoss(lowerQuantile, upperQuantile);
    }

    public static PinballLoss interval(double lowerQuantile, double upperQuantile) {
        return interval((float) lowerQuantile, (float) upperQuantile);
    }

    public GradTensor forward(GradTensor predictions, GradTensor targets) {
        return compute(predictions, targets);
    }

    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        ShapeSpec spec = requireCompatibleFiniteNonEmpty(predictions, targets);
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();

        float total = 0.0f;
        for (int targetIndex = 0; targetIndex < spec.targetCount; targetIndex++) {
            for (int quantileIndex = 0; quantileIndex < quantiles.length; quantileIndex++) {
                int predictionIndex = spec.predictionIndex(targetIndex, quantileIndex);
                total += pinball(
                        targetData[spec.targetIndex(targetIndex)] - predictionData[predictionIndex],
                        quantiles[quantileIndex]);
            }
        }

        GradTensor out = GradTensor.scalar(total / spec.predictionCount);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("PinballLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.predictionCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int targetIndex = 0; targetIndex < spec.targetCount; targetIndex++) {
                        for (int quantileIndex = 0; quantileIndex < quantiles.length; quantileIndex++) {
                            int predictionIndex = spec.predictionIndex(targetIndex, quantileIndex);
                            float error = targetData[spec.targetIndex(targetIndex)]
                                    - predictionData[predictionIndex];
                            grad[predictionIndex] = pinballGradient(error, quantiles[quantileIndex]) * scale;
                        }
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public float[] quantiles() {
        return quantiles.clone();
    }

    @Override
    public String toString() {
        return "PinballLoss(quantiles=" + Arrays.toString(quantiles) + ")";
    }

    private ShapeSpec requireCompatibleFiniteNonEmpty(GradTensor predictions, GradTensor targets) {
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();
        if (predictionData.length == 0) {
            throw new IllegalArgumentException("PinballLoss requires at least one prediction");
        }

        ShapeSpec spec = quantiles.length == 1
                ? ShapeSpec.singleQuantile(predictions, targets)
                : ShapeSpec.multiQuantile(predictions, targets, quantiles.length);

        for (int i = 0; i < predictionData.length; i++) {
            if (!Float.isFinite(predictionData[i])) {
                throw new IllegalArgumentException(
                        "PinballLoss predictions must be finite, got " + predictionData[i] + " at index " + i);
            }
        }
        for (int i = 0; i < targetData.length; i++) {
            if (!Float.isFinite(targetData[i])) {
                throw new IllegalArgumentException(
                        "PinballLoss targets must be finite, got " + targetData[i] + " at index " + i);
            }
        }
        return spec;
    }

    private static float pinball(float error, float quantile) {
        if (error > 0.0f) {
            return quantile * error;
        }
        if (error < 0.0f) {
            return (quantile - 1.0f) * error;
        }
        return 0.0f;
    }

    private static float pinballGradient(float error, float quantile) {
        if (error > 0.0f) {
            return -quantile;
        }
        if (error < 0.0f) {
            return 1.0f - quantile;
        }
        return 0.0f;
    }

    private static float[] normalizeQuantiles(float[] quantiles) {
        if (quantiles == null || quantiles.length == 0) {
            throw new IllegalArgumentException("PinballLoss requires at least one quantile");
        }
        float[] copy = quantiles.clone();
        for (float quantile : copy) {
            if (!Float.isFinite(quantile) || quantile <= 0.0f || quantile >= 1.0f) {
                throw new IllegalArgumentException(
                        "PinballLoss quantiles must be finite values in (0, 1), got: " + quantile);
            }
        }
        return copy;
    }

    private static float[] toFloatArray(double[] values) {
        if (values == null) {
            return null;
        }
        float[] copy = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            copy[i] = (float) values[i];
        }
        return copy;
    }

    private record ShapeSpec(
            int predictionCount,
            int targetCount,
            boolean repeatedTargets) {

        static ShapeSpec singleQuantile(GradTensor predictions, GradTensor targets) {
            int n = RegressionLosses.requireSameFiniteNonEmpty(predictions, targets, "PinballLoss");
            return new ShapeSpec(n, n, false);
        }

        static ShapeSpec multiQuantile(GradTensor predictions, GradTensor targets, int quantileCount) {
            long[] predictionShape = predictions.shape();
            long[] targetShape = targets.shape();
            if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != quantileCount) {
                throw new IllegalArgumentException(
                        "PinballLoss multi-quantile predictions must end with quantile dimension "
                                + quantileCount + ", got: " + Arrays.toString(predictionShape));
            }
            long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
            if (!Arrays.equals(expectedTargetShape, targetShape)) {
                throw new IllegalArgumentException(
                        "PinballLoss multi-quantile targets must match predictions without the final "
                                + "quantile dimension, got: " + Arrays.toString(targetShape)
                                + " expected: " + Arrays.toString(expectedTargetShape));
            }
            int predictionCount = predictions.data().length;
            int targetCount = targets.data().length;
            if (targetCount == 0) {
                throw new IllegalArgumentException("PinballLoss requires at least one target");
            }
            return new ShapeSpec(predictionCount, targetCount, true);
        }

        int predictionIndex(int targetIndex, int quantileIndex) {
            return repeatedTargets ? targetIndex * quantileCount() + quantileIndex : targetIndex;
        }

        int targetIndex(int targetIndex) {
            return targetIndex;
        }

        private int quantileCount() {
            return predictionCount / targetCount;
        }
    }
}
