package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Poisson negative log-likelihood for count regression.
 *
 * <p>
 * By default predictions are interpreted as log-rates, so the model can emit
 * unconstrained real values and the loss uses {@code exp(prediction)} as the
 * Poisson rate. Set {@code logInput=false} when predictions are already
 * non-negative rates.
 */
public final class PoissonNllLoss {

    private static final float DEFAULT_EPS = 1.0e-8f;
    private static final float TWO_PI = (float) (2.0 * Math.PI);

    private final boolean logInput;
    private final boolean full;
    private final float eps;

    /** Creates Poisson NLL for log-rate predictions without Stirling target term. */
    public PoissonNllLoss() {
        this(true);
    }

    public PoissonNllLoss(boolean logInput) {
        this(logInput, false);
    }

    public PoissonNllLoss(boolean logInput, boolean full) {
        this(logInput, full, DEFAULT_EPS);
    }

    public PoissonNllLoss(boolean logInput, boolean full, float eps) {
        if (!Float.isFinite(eps) || eps <= 0.0f) {
            throw new IllegalArgumentException("eps must be finite and positive, got: " + eps);
        }
        this.logInput = logInput;
        this.full = full;
        this.eps = eps;
    }

    public GradTensor forward(GradTensor predictions, GradTensor targets) {
        return compute(predictions, targets);
    }

    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        int n = requireCompatibleCountData(predictions, targets);
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();

        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            total += elementLoss(predictionData[i], targetData[i]);
        }

        GradTensor out = GradTensor.scalar(total / n);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("PoissonNllLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        grad[i] = elementGradient(predictionData[i], targetData[i]) * scale;
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

    public boolean full() {
        return full;
    }

    public float eps() {
        return eps;
    }

    @Override
    public String toString() {
        return "PoissonNllLoss(logInput=" + logInput + ", full=" + full + ", eps=" + eps + ")";
    }

    private float elementLoss(float prediction, float target) {
        float loss = logInput
                ? (float) Math.exp(prediction) - target * prediction
                : prediction - target * (float) Math.log(prediction + eps);
        if (full && target > 1.0f) {
            loss += target * (float) Math.log(target)
                    - target
                    + 0.5f * (float) Math.log(TWO_PI * target);
        }
        return loss;
    }

    private float elementGradient(float prediction, float target) {
        if (logInput) {
            return (float) Math.exp(prediction) - target;
        }
        return 1.0f - target / (prediction + eps);
    }

    private int requireCompatibleCountData(GradTensor predictions, GradTensor targets) {
        int n = RegressionLosses.requireSameFiniteNonEmpty(predictions, targets, "PoissonNllLoss");
        float[] predictionData = predictions.data();
        float[] targetData = targets.data();
        for (int i = 0; i < n; i++) {
            if (targetData[i] < 0.0f) {
                throw new IllegalArgumentException(
                        "PoissonNllLoss targets must be non-negative counts, got "
                                + targetData[i] + " at index " + i);
            }
            if (!logInput && predictionData[i] < 0.0f) {
                throw new IllegalArgumentException(
                        "PoissonNllLoss rate predictions must be non-negative when logInput=false, got "
                                + predictionData[i] + " at index " + i);
            }
        }
        return n;
    }
}
