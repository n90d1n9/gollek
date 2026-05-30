package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Proper scoring loss for prediction intervals.
 *
 * <p>
 * Uses the interval score, also known as the Winkler score:
 * {@code width + 2/alpha * missDistance}. Lower is better. Narrow intervals are
 * rewarded only when they still contain the target; misses receive a linear
 * penalty controlled by {@code alpha}. Crossed intervals are penalized directly
 * so models can recover from invalid lower/upper ordering during training.
 */
public final class PredictionIntervalLoss {

    private final float alpha;
    private final float missPenalty;
    private final float crossingPenalty;

    /** Creates a 90% prediction interval loss ({@code alpha=0.1}). */
    public PredictionIntervalLoss() {
        this(0.1f);
    }

    public PredictionIntervalLoss(float alpha) {
        this(alpha, defaultCrossingPenalty(alpha));
    }

    public PredictionIntervalLoss(double alpha) {
        this((float) alpha);
    }

    public PredictionIntervalLoss(float alpha, float crossingPenalty) {
        this.alpha = normalizeAlpha(alpha);
        this.missPenalty = 2.0f / this.alpha;
        this.crossingPenalty = normalizeCrossingPenalty(crossingPenalty);
    }

    public PredictionIntervalLoss(double alpha, double crossingPenalty) {
        this((float) alpha, (float) crossingPenalty);
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
            float lower = predictionData[2 * i];
            float upper = predictionData[2 * i + 1];
            float target = targetData[i];
            total += score(lower, upper, target);
        }

        GradTensor out = GradTensor.scalar(total / spec.targetCount);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("PredictionIntervalLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.targetCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int i = 0; i < spec.targetCount; i++) {
                        float lower = predictionData[2 * i];
                        float upper = predictionData[2 * i + 1];
                        float target = targetData[i];
                        grad[2 * i] = lowerGradient(lower, upper, target) * scale;
                        grad[2 * i + 1] = upperGradient(lower, upper, target) * scale;
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public float alpha() {
        return alpha;
    }

    public float crossingPenalty() {
        return crossingPenalty;
    }

    @Override
    public String toString() {
        return "PredictionIntervalLoss(alpha=" + alpha + ", crossingPenalty=" + crossingPenalty + ")";
    }

    private float score(float lower, float upper, float target) {
        float width = upper - lower;
        float lowerMiss = target < lower ? lower - target : 0.0f;
        float upperMiss = target > upper ? target - upper : 0.0f;
        float crossing = lower > upper ? lower - upper : 0.0f;
        return width + missPenalty * (lowerMiss + upperMiss) + crossingPenalty * crossing;
    }

    private float lowerGradient(float lower, float upper, float target) {
        float grad = -1.0f;
        if (target < lower) {
            grad += missPenalty;
        }
        if (lower > upper) {
            grad += crossingPenalty;
        }
        return grad;
    }

    private float upperGradient(float lower, float upper, float target) {
        float grad = 1.0f;
        if (target > upper) {
            grad -= missPenalty;
        }
        if (lower > upper) {
            grad -= crossingPenalty;
        }
        return grad;
    }

    private ShapeSpec requireCompatibleFiniteNonEmpty(GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 2) {
            throw new IllegalArgumentException(
                    "PredictionIntervalLoss predictions must end with lower/upper dimension 2, got: "
                            + Arrays.toString(predictionShape));
        }
        long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
        if (!Arrays.equals(expectedTargetShape, targetShape)) {
            throw new IllegalArgumentException(
                    "PredictionIntervalLoss targets must match predictions without the final lower/upper dimension, "
                            + "got: " + Arrays.toString(targetShape)
                            + " expected: " + Arrays.toString(expectedTargetShape));
        }

        int predictionCount = predictions.data().length;
        int targetCount = targets.data().length;
        if (targetCount == 0) {
            throw new IllegalArgumentException("PredictionIntervalLoss requires at least one target");
        }
        for (int i = 0; i < predictionCount; i++) {
            if (!Float.isFinite(predictions.data()[i])) {
                throw new IllegalArgumentException(
                        "PredictionIntervalLoss predictions must be finite, got "
                                + predictions.data()[i] + " at index " + i);
            }
        }
        for (int i = 0; i < targetCount; i++) {
            if (!Float.isFinite(targets.data()[i])) {
                throw new IllegalArgumentException(
                        "PredictionIntervalLoss targets must be finite, got " + targets.data()[i] + " at index " + i);
            }
        }
        return new ShapeSpec(predictionCount, targetCount);
    }

    private static float normalizeAlpha(float alpha) {
        if (!Float.isFinite(alpha) || alpha <= 0.0f || alpha >= 1.0f) {
            throw new IllegalArgumentException("alpha must be finite and in (0, 1), got: " + alpha);
        }
        return alpha;
    }

    private static float normalizeCrossingPenalty(float crossingPenalty) {
        if (!Float.isFinite(crossingPenalty) || crossingPenalty <= 1.0f) {
            throw new IllegalArgumentException(
                    "crossingPenalty must be finite and greater than 1, got: " + crossingPenalty);
        }
        return crossingPenalty;
    }

    private static float defaultCrossingPenalty(float alpha) {
        float normalizedAlpha = normalizeAlpha(alpha);
        return 1.0f + 2.0f / normalizedAlpha;
    }

    private record ShapeSpec(int predictionCount, int targetCount) {
    }
}
