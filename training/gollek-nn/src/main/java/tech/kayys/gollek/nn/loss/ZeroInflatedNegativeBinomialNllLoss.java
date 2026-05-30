package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Zero-inflated negative-binomial negative log-likelihood for overdispersed
 * count data with excess zeros.
 *
 * <p>
 * Predictions end with {@code [mean, inverseDispersion, zeroInflationLogit]}.
 * By default the mean and inverse-dispersion values are interpreted in
 * log-space, while the zero-inflation logit remains unconstrained.
 */
public final class ZeroInflatedNegativeBinomialNllLoss {

    private static final float DEFAULT_EPS = 1.0e-8f;
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

    private final boolean logInput;
    private final boolean includeConstant;
    private final float eps;

    /**
     * Creates ZINB NLL for
     * {@code [logMean, logInverseDispersion, zeroInflationLogit]} predictions.
     */
    public ZeroInflatedNegativeBinomialNllLoss() {
        this(true);
    }

    public ZeroInflatedNegativeBinomialNllLoss(boolean logInput) {
        this(logInput, false);
    }

    public ZeroInflatedNegativeBinomialNllLoss(boolean logInput, boolean includeConstant) {
        this(logInput, includeConstant, DEFAULT_EPS);
    }

    public ZeroInflatedNegativeBinomialNllLoss(boolean logInput, boolean includeConstant, double eps) {
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
            int offset = 3 * i;
            double mean = positiveParameter(predictionData[offset]);
            double inverseDispersion = positiveParameter(predictionData[offset + 1]);
            double zeroInflationLogit = predictionData[offset + 2];
            double target = targetData[i];
            total += target == 0.0
                    ? zeroLoss(mean, inverseDispersion, zeroInflationLogit)
                    : positiveLoss(mean, inverseDispersion, zeroInflationLogit, target);
        }

        GradTensor out = GradTensor.scalar((float) (total / spec.targetCount));
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("ZeroInflatedNegativeBinomialNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.targetCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int i = 0; i < spec.targetCount; i++) {
                        int offset = 3 * i;
                        double mean = positiveParameter(predictionData[offset]);
                        double inverseDispersion = positiveParameter(predictionData[offset + 1]);
                        double zeroInflationLogit = predictionData[offset + 2];
                        double target = targetData[i];

                        double meanGradient = target == 0.0
                                ? zeroMeanGradient(mean, inverseDispersion, zeroInflationLogit)
                                : meanGradient(mean, inverseDispersion, target);
                        double dispersionGradient = target == 0.0
                                ? zeroInverseDispersionGradient(mean, inverseDispersion, zeroInflationLogit)
                                : inverseDispersionGradient(mean, inverseDispersion, target);
                        double logitGradient = target == 0.0
                                ? zeroLogitGradient(mean, inverseDispersion, zeroInflationLogit)
                                : sigmoid(zeroInflationLogit);

                        grad[offset] = (float) (scale * chainGradient(meanGradient, mean));
                        grad[offset + 1] = (float) (scale * chainGradient(dispersionGradient, inverseDispersion));
                        grad[offset + 2] = (float) (scale * logitGradient);
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
        return "ZeroInflatedNegativeBinomialNllLoss(logInput=" + logInput
                + ", includeConstant=" + includeConstant
                + ", eps=" + eps + ")";
    }

    private double zeroLoss(double mean, double inverseDispersion, double zeroInflationLogit) {
        return -zeroLogProbability(mean, inverseDispersion, zeroInflationLogit);
    }

    private double positiveLoss(
            double mean,
            double inverseDispersion,
            double zeroInflationLogit,
            double target) {
        return -logSigmoid(-zeroInflationLogit) + negativeBinomialLoss(mean, inverseDispersion, target);
    }

    private double negativeBinomialLoss(double mean, double inverseDispersion, double target) {
        double loss = logGamma(inverseDispersion) - logGamma(target + inverseDispersion)
                - inverseDispersion * Math.log(inverseDispersion)
                + (target + inverseDispersion) * Math.log(mean + inverseDispersion)
                - target * Math.log(mean);
        if (includeConstant) {
            loss += logGamma(target + 1.0);
        }
        return loss;
    }

    private double zeroMeanGradient(double mean, double inverseDispersion, double zeroInflationLogit) {
        double posteriorNegativeBinomialZero = posteriorNegativeBinomialZero(
                mean,
                inverseDispersion,
                zeroInflationLogit);
        return posteriorNegativeBinomialZero * inverseDispersion / (mean + inverseDispersion);
    }

    private double zeroInverseDispersionGradient(
            double mean,
            double inverseDispersion,
            double zeroInflationLogit) {
        double posteriorNegativeBinomialZero = posteriorNegativeBinomialZero(
                mean,
                inverseDispersion,
                zeroInflationLogit);
        double logZeroDerivative = Math.log(inverseDispersion / (mean + inverseDispersion))
                + 1.0
                - inverseDispersion / (mean + inverseDispersion);
        return -posteriorNegativeBinomialZero * logZeroDerivative;
    }

    private double zeroLogitGradient(double mean, double inverseDispersion, double zeroInflationLogit) {
        double logNegativeBinomialZero = negativeBinomialZeroLogProbability(mean, inverseDispersion);
        double oneMinusNegativeBinomialZero = -Math.expm1(logNegativeBinomialZero);
        if (oneMinusNegativeBinomialZero <= 0.0) {
            return 0.0;
        }
        double logProb = zeroLogProbability(mean, inverseDispersion, zeroInflationLogit);
        return -oneMinusNegativeBinomialZero
                * Math.exp(logSigmoid(zeroInflationLogit) + logSigmoid(-zeroInflationLogit) - logProb);
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

    private double posteriorNegativeBinomialZero(
            double mean,
            double inverseDispersion,
            double zeroInflationLogit) {
        double logNegativeBinomialZero = negativeBinomialZeroLogProbability(mean, inverseDispersion);
        double logProb = zeroLogProbability(mean, inverseDispersion, zeroInflationLogit);
        return Math.exp(logSigmoid(-zeroInflationLogit) + logNegativeBinomialZero - logProb);
    }

    private double zeroLogProbability(double mean, double inverseDispersion, double zeroInflationLogit) {
        return logAddExp(
                logSigmoid(zeroInflationLogit),
                logSigmoid(-zeroInflationLogit)
                        + negativeBinomialZeroLogProbability(mean, inverseDispersion));
    }

    private double negativeBinomialZeroLogProbability(double mean, double inverseDispersion) {
        return inverseDispersion * (Math.log(inverseDispersion) - Math.log(mean + inverseDispersion));
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
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 3) {
            throw new IllegalArgumentException(
                    "ZeroInflatedNegativeBinomialNllLoss predictions must end with "
                            + "mean/inverseDispersion/zeroInflationLogit dimension 3, got: "
                            + Arrays.toString(predictionShape));
        }
        long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
        if (!Arrays.equals(expectedTargetShape, targetShape)) {
            throw new IllegalArgumentException(
                    "ZeroInflatedNegativeBinomialNllLoss targets must match predictions without the final "
                            + "distribution dimension, got: " + Arrays.toString(targetShape)
                            + " expected: " + Arrays.toString(expectedTargetShape));
        }

        int predictionCount = predictions.data().length;
        int targetCount = targets.data().length;
        if (targetCount == 0) {
            throw new IllegalArgumentException("ZeroInflatedNegativeBinomialNllLoss requires at least one target");
        }
        for (int i = 0; i < predictionCount; i++) {
            float value = predictions.data()[i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "ZeroInflatedNegativeBinomialNllLoss predictions must be finite, got "
                                + value + " at index " + i);
            }
            if (!logInput && i % 3 != 2 && value < 0.0f) {
                throw new IllegalArgumentException(
                        "ZeroInflatedNegativeBinomialNllLoss raw mean and inverse-dispersion predictions must be "
                                + "non-negative when logInput=false, got " + value + " at index " + i);
            }
        }
        for (int i = 0; i < targetCount; i++) {
            float target = targets.data()[i];
            if (!Float.isFinite(target) || target < 0.0f) {
                throw new IllegalArgumentException(
                        "ZeroInflatedNegativeBinomialNllLoss targets must be finite non-negative counts, got "
                                + target + " at index " + i);
            }
        }
        return new ShapeSpec(predictionCount, targetCount);
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            double expNeg = Math.exp(-value);
            return 1.0 / (1.0 + expNeg);
        }
        double exp = Math.exp(value);
        return exp / (1.0 + exp);
    }

    private static double logSigmoid(double value) {
        if (value >= 0.0) {
            return -Math.log1p(Math.exp(-value));
        }
        return value - Math.log1p(Math.exp(value));
    }

    private static double logAddExp(double a, double b) {
        double max = Math.max(a, b);
        if (Double.isInfinite(max)) {
            return max;
        }
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
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
