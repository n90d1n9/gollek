package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Negative binomial negative log-likelihood for overdispersed count regression.
 *
 * <p>
 * Predictions end with two values per target: {@code [mean, inverseDispersion]}.
 * By default both values are interpreted in log-space, so models can emit
 * unconstrained real values. The inverse-dispersion parameter is the usual
 * negative-binomial shape/total-count parameter: larger values approach a
 * Poisson-like variance, smaller values allow stronger overdispersion.
 */
public final class NegativeBinomialNllLoss {

    private static final double[] LANCZOS = {
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
    };
    private static final double HALF_LOG_TWO_PI = 0.9189385332046727;
    private static final float DEFAULT_EPS = 1.0e-8f;

    private final boolean logInput;
    private final boolean includeConstant;
    private final float eps;

    /** Creates negative-binomial NLL for {@code [logMean, logInverseDispersion]} predictions. */
    public NegativeBinomialNllLoss() {
        this(true);
    }

    public NegativeBinomialNllLoss(boolean logInput) {
        this(logInput, false);
    }

    public NegativeBinomialNllLoss(boolean logInput, boolean includeConstant) {
        this(logInput, includeConstant, DEFAULT_EPS);
    }

    public NegativeBinomialNllLoss(boolean logInput, boolean includeConstant, double eps) {
        if (!Double.isFinite(eps) || eps <= 0.0) {
            throw new IllegalArgumentException("eps must be finite and positive, got: " + eps);
        }
        this.logInput = logInput;
        this.includeConstant = includeConstant;
        this.eps = (float) eps;
    }

    public GradTensor forward(GradTensor predictions, GradTensor targets) {
        return compute(predictions, targets);
    }

    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        ShapeSpec spec = requireCompatibleCountData(predictions, targets);
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();

        double total = 0.0;
        for (int i = 0; i < spec.targetCount; i++) {
            double mean = positiveParameter(predictionData[2 * i]);
            double inverseDispersion = positiveParameter(predictionData[2 * i + 1]);
            total += elementLoss(mean, inverseDispersion, targetData[i]);
        }

        GradTensor out = GradTensor.scalar((float) (total / spec.targetCount));
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("NegativeBinomialNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.targetCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int i = 0; i < spec.targetCount; i++) {
                        double mean = positiveParameter(predictionData[2 * i]);
                        double inverseDispersion = positiveParameter(predictionData[2 * i + 1]);
                        double target = targetData[i];
                        double meanGradient = meanGradient(mean, inverseDispersion, target);
                        double dispersionGradient = inverseDispersionGradient(mean, inverseDispersion, target);
                        grad[2 * i] = (float) (scale * chainGradient(meanGradient, mean));
                        grad[2 * i + 1] = (float) (scale * chainGradient(dispersionGradient, inverseDispersion));
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public boolean logInput() {
        return logInput;
    }

    public boolean includeConstant() {
        return includeConstant;
    }

    public float eps() {
        return eps;
    }

    @Override
    public String toString() {
        return "NegativeBinomialNllLoss(logInput=" + logInput
                + ", includeConstant=" + includeConstant
                + ", eps=" + eps + ")";
    }

    private double elementLoss(double mean, double inverseDispersion, double target) {
        double total = logGamma(inverseDispersion) - logGamma(target + inverseDispersion)
                - inverseDispersion * Math.log(inverseDispersion)
                + (target + inverseDispersion) * Math.log(target + inverseDispersion > 0.0
                        ? mean + inverseDispersion
                        : mean)
                - target * Math.log(mean);
        if (includeConstant) {
            total += logGamma(target + 1.0);
        }
        return total;
    }

    private double meanGradient(double mean, double inverseDispersion, double target) {
        return (target + inverseDispersion) / (mean + inverseDispersion) - target / mean;
    }

    private double inverseDispersionGradient(double mean, double inverseDispersion, double target) {
        return digamma(inverseDispersion)
                - digamma(target + inverseDispersion)
                - Math.log(inverseDispersion)
                - 1.0
                + Math.log(mean + inverseDispersion)
                + (target + inverseDispersion) / (mean + inverseDispersion);
    }

    private double chainGradient(double parameterGradient, double parameterValue) {
        return logInput ? parameterGradient * parameterValue : parameterGradient;
    }

    private double positiveParameter(float raw) {
        return logInput ? Math.exp(raw) : raw + eps;
    }

    private ShapeSpec requireCompatibleCountData(GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 2) {
            throw new IllegalArgumentException(
                    "NegativeBinomialNllLoss predictions must end with mean/inverseDispersion dimension 2, got: "
                            + Arrays.toString(predictionShape));
        }
        long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
        if (!Arrays.equals(expectedTargetShape, targetShape)) {
            throw new IllegalArgumentException(
                    "NegativeBinomialNllLoss targets must match predictions without the final distribution "
                            + "dimension, got: " + Arrays.toString(targetShape)
                            + " expected: " + Arrays.toString(expectedTargetShape));
        }

        int predictionCount = predictions.data().length;
        int targetCount = targets.data().length;
        if (targetCount == 0) {
            throw new IllegalArgumentException("NegativeBinomialNllLoss requires at least one target");
        }
        for (int i = 0; i < predictionCount; i++) {
            float value = predictions.data()[i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "NegativeBinomialNllLoss predictions must be finite, got " + value + " at index " + i);
            }
            if (!logInput && value < 0.0f) {
                throw new IllegalArgumentException(
                        "NegativeBinomialNllLoss raw predictions must be non-negative when logInput=false, got "
                                + value + " at index " + i);
            }
        }
        for (int i = 0; i < targetCount; i++) {
            float target = targets.data()[i];
            if (!Float.isFinite(target) || target < 0.0f) {
                throw new IllegalArgumentException(
                        "NegativeBinomialNllLoss targets must be finite non-negative counts, got "
                                + target + " at index " + i);
            }
        }
        return new ShapeSpec(predictionCount, targetCount);
    }

    private static double logGamma(double z) {
        if (z < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1.0 - z);
        }
        double x = 0.99999999999980993;
        double shifted = z - 1.0;
        for (int i = 0; i < LANCZOS.length; i++) {
            x += LANCZOS[i] / (shifted + i + 1.0);
        }
        double t = shifted + LANCZOS.length - 0.5;
        return HALF_LOG_TWO_PI + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
    }

    private static double digamma(double x) {
        double result = 0.0;
        while (x < 8.0) {
            result -= 1.0 / x;
            x += 1.0;
        }
        double inverse = 1.0 / x;
        double inverseSquared = inverse * inverse;
        return result
                + Math.log(x)
                - 0.5 * inverse
                - inverseSquared * (1.0 / 12.0
                        - inverseSquared * (1.0 / 120.0
                                - inverseSquared * (1.0 / 252.0
                                        - inverseSquared * (1.0 / 240.0))));
    }

    private record ShapeSpec(int predictionCount, int targetCount) {
    }
}
