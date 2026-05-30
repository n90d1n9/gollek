package tech.kayys.gollek.ml.nn.loss;

import java.util.Arrays;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Zero-inflated Poisson negative log-likelihood for count data with excess zeros.
 *
 * <p>
 * Predictions end with {@code [rate, zeroInflationLogit]}. By default the rate
 * is interpreted in log-space so models can emit unconstrained real values,
 * while the zero-inflation logit is always unconstrained.
 */
public final class ZeroInflatedPoissonNllLoss {

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

    private final boolean logRateInput;
    private final boolean includeConstant;
    private final float eps;

    /** Creates zero-inflated Poisson NLL for {@code [logRate, zeroInflationLogit]}. */
    public ZeroInflatedPoissonNllLoss() {
        this(true);
    }

    public ZeroInflatedPoissonNllLoss(boolean logRateInput) {
        this(logRateInput, false);
    }

    public ZeroInflatedPoissonNllLoss(boolean logRateInput, boolean includeConstant) {
        this(logRateInput, includeConstant, DEFAULT_EPS);
    }

    public ZeroInflatedPoissonNllLoss(boolean logRateInput, boolean includeConstant, double eps) {
        if (!Double.isFinite(eps) || eps <= 0.0) {
            throw new IllegalArgumentException("eps must be finite and positive, got: " + eps);
        }
        this.logRateInput = logRateInput;
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
            double rate = positiveRate(predictionData[2 * i]);
            double zeroInflationLogit = predictionData[2 * i + 1];
            double target = targetData[i];
            total += target == 0.0 ? zeroLoss(rate, zeroInflationLogit) : positiveLoss(rate, zeroInflationLogit, target);
        }

        GradTensor out = GradTensor.scalar((float) (total / spec.targetCount));
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("ZeroInflatedPoissonNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / spec.targetCount;
                    float[] grad = new float[spec.predictionCount];
                    for (int i = 0; i < spec.targetCount; i++) {
                        double rate = positiveRate(predictionData[2 * i]);
                        double zeroInflationLogit = predictionData[2 * i + 1];
                        double target = targetData[i];
                        double rateGradient = target == 0.0
                                ? zeroRateGradient(rate, zeroInflationLogit)
                                : positiveRateGradient(rate, target);
                        double logitGradient = target == 0.0
                                ? zeroLogitGradient(rate, zeroInflationLogit)
                                : sigmoid(zeroInflationLogit);
                        grad[2 * i] = (float) (scale * chainRateGradient(rateGradient, rate));
                        grad[2 * i + 1] = (float) (scale * logitGradient);
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public boolean logRateInput() {
        return logRateInput;
    }

    public boolean includeConstant() {
        return includeConstant;
    }

    public float eps() {
        return eps;
    }

    @Override
    public String toString() {
        return "ZeroInflatedPoissonNllLoss(logRateInput=" + logRateInput
                + ", includeConstant=" + includeConstant
                + ", eps=" + eps + ")";
    }

    private double zeroLoss(double rate, double zeroInflationLogit) {
        return -zeroLogProbability(rate, zeroInflationLogit);
    }

    private double positiveLoss(double rate, double zeroInflationLogit, double target) {
        double loss = -logSigmoid(-zeroInflationLogit) - target * Math.log(rate) + rate;
        if (includeConstant) {
            loss += logGamma(target + 1.0);
        }
        return loss;
    }

    private double zeroRateGradient(double rate, double zeroInflationLogit) {
        double logProb = zeroLogProbability(rate, zeroInflationLogit);
        return Math.exp(logSigmoid(-zeroInflationLogit) - rate - logProb);
    }

    private double zeroLogitGradient(double rate, double zeroInflationLogit) {
        double oneMinusExpNegRate = -Math.expm1(-rate);
        if (oneMinusExpNegRate <= 0.0) {
            return 0.0;
        }
        double logProb = zeroLogProbability(rate, zeroInflationLogit);
        return -Math.exp(logSigmoid(zeroInflationLogit)
                + logSigmoid(-zeroInflationLogit)
                + Math.log(oneMinusExpNegRate)
                - logProb);
    }

    private double positiveRateGradient(double rate, double target) {
        return 1.0 - target / rate;
    }

    private double chainRateGradient(double rateGradient, double rate) {
        return logRateInput ? rateGradient * rate : rateGradient;
    }

    private double zeroLogProbability(double rate, double zeroInflationLogit) {
        return logAddExp(
                logSigmoid(zeroInflationLogit),
                logSigmoid(-zeroInflationLogit) - rate);
    }

    private double positiveRate(float rawRate) {
        return logRateInput ? Math.exp(rawRate) : rawRate + eps;
    }

    private ShapeSpec requireCompatibleCountData(GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 2) {
            throw new IllegalArgumentException(
                    "ZeroInflatedPoissonNllLoss predictions must end with rate/zeroInflationLogit dimension 2, got: "
                            + Arrays.toString(predictionShape));
        }
        long[] expectedTargetShape = Arrays.copyOf(predictionShape, predictionShape.length - 1);
        if (!Arrays.equals(expectedTargetShape, targetShape)) {
            throw new IllegalArgumentException(
                    "ZeroInflatedPoissonNllLoss targets must match predictions without the final distribution "
                            + "dimension, got: " + Arrays.toString(targetShape)
                            + " expected: " + Arrays.toString(expectedTargetShape));
        }

        int predictionCount = predictions.data().length;
        int targetCount = targets.data().length;
        if (targetCount == 0) {
            throw new IllegalArgumentException("ZeroInflatedPoissonNllLoss requires at least one target");
        }
        for (int i = 0; i < predictionCount; i++) {
            float value = predictions.data()[i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "ZeroInflatedPoissonNllLoss predictions must be finite, got " + value + " at index " + i);
            }
            if (!logRateInput && i % 2 == 0 && value < 0.0f) {
                throw new IllegalArgumentException(
                        "ZeroInflatedPoissonNllLoss raw rate predictions must be non-negative when logRateInput=false,"
                                + " got " + value + " at index " + i);
            }
        }
        for (int i = 0; i < targetCount; i++) {
            float target = targets.data()[i];
            if (!Float.isFinite(target) || target < 0.0f) {
                throw new IllegalArgumentException(
                        "ZeroInflatedPoissonNllLoss targets must be finite non-negative counts, got "
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

    private record ShapeSpec(int predictionCount, int targetCount) {
    }
}
