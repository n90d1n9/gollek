package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Tweedie negative log-likelihood for count-heavy and positive regression.
 *
 * <p>
 * The supported variance powers are {@code 1 <= power <= 2}: Poisson
 * ({@code power=1}), compound Poisson-Gamma ({@code 1 < power < 2}), and Gamma
 * ({@code power=2}). By default predictions are interpreted as log-means so a
 * model can emit unconstrained real values.
 */
public final class TweedieNllLoss {

    public static final float DEFAULT_POWER = 1.5f;
    private static final float DEFAULT_EPS = 1.0e-8f;
    private static final float POWER_TOLERANCE = 1.0e-6f;

    private final float power;
    private final boolean logInput;
    private final float eps;

    public TweedieNllLoss() {
        this(DEFAULT_POWER);
    }

    public TweedieNllLoss(double power) {
        this(power, true);
    }

    public TweedieNllLoss(double power, boolean logInput) {
        this(power, logInput, DEFAULT_EPS);
    }

    public TweedieNllLoss(double power, boolean logInput, double eps) {
        if (!Double.isFinite(power) || power < 1.0 || power > 2.0) {
            throw new IllegalArgumentException("power must be finite and in [1, 2], got: " + power);
        }
        if (!Double.isFinite(eps) || eps <= 0.0) {
            throw new IllegalArgumentException("eps must be finite and positive, got: " + eps);
        }
        this.power = (float) power;
        this.logInput = logInput;
        this.eps = (float) eps;
    }

    public GradTensor forward(GradTensor predictions, GradTensor targets) {
        return compute(predictions, targets);
    }

    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        int n = requireCompatibleTweedieData(predictions, targets);
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();

        double total = 0.0;
        for (int i = 0; i < n; i++) {
            total += elementLoss(predictionData[i], targetData[i]);
        }

        GradTensor out = GradTensor.scalar((float) (total / n));
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("TweedieNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        grad[i] = (float) (elementGradient(predictionData[i], targetData[i]) * scale);
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    public float power() {
        return power;
    }

    public boolean logInput() {
        return logInput;
    }

    public float eps() {
        return eps;
    }

    @Override
    public String toString() {
        return "TweedieNllLoss(power=" + power + ", logInput=" + logInput + ", eps=" + eps + ")";
    }

    private double elementLoss(float prediction, float target) {
        double logMean = logMean(prediction);
        if (isPoissonPower()) {
            return Math.exp(logMean) - target * logMean;
        }
        if (isGammaPower()) {
            return logMean + target * Math.exp(-logMean);
        }
        double oneMinusPower = 1.0 - power;
        double twoMinusPower = 2.0 - power;
        return Math.exp(twoMinusPower * logMean) / twoMinusPower
                - target * Math.exp(oneMinusPower * logMean) / oneMinusPower;
    }

    private double elementGradient(float prediction, float target) {
        double logMean = logMean(prediction);
        double gradLogMean;
        if (isPoissonPower()) {
            gradLogMean = Math.exp(logMean) - target;
        } else if (isGammaPower()) {
            gradLogMean = 1.0 - target * Math.exp(-logMean);
        } else {
            gradLogMean = Math.exp((2.0 - power) * logMean)
                    - target * Math.exp((1.0 - power) * logMean);
        }
        return logInput ? gradLogMean : gradLogMean / Math.exp(logMean);
    }

    private double logMean(float prediction) {
        return logInput ? prediction : Math.log(prediction + eps);
    }

    private boolean isPoissonPower() {
        return Math.abs(power - 1.0f) <= POWER_TOLERANCE;
    }

    private boolean isGammaPower() {
        return Math.abs(power - 2.0f) <= POWER_TOLERANCE;
    }

    private int requireCompatibleTweedieData(GradTensor predictions, GradTensor targets) {
        int n = RegressionLosses.requireSameFiniteNonEmpty(predictions, targets, "TweedieNllLoss");
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();
        for (int i = 0; i < n; i++) {
            if (targetData[i] < 0.0f) {
                throw new IllegalArgumentException(
                        "TweedieNllLoss targets must be non-negative, got " + targetData[i] + " at index " + i);
            }
            if (!logInput && predictionData[i] < 0.0f) {
                throw new IllegalArgumentException(
                        "TweedieNllLoss mean predictions must be non-negative when logInput=false, got "
                                + predictionData[i] + " at index " + i);
            }
        }
        return n;
    }
}
