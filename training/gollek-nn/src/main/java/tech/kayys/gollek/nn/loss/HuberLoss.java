package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Huber Loss (Smooth L1) — quadratic for small errors, linear for large errors.
 *
 * <p>
 * Less sensitive to outliers than MSE while being differentiable everywhere.
 * Used in DQN, object detection (Fast R-CNN), and robust regression.
 *
 * <p>
 * Formula:
 * 
 * <pre>
 *   L(y, ŷ) = 0.5·(y-ŷ)²          if |y-ŷ| ≤ δ
 *            = δ·(|y-ŷ| - 0.5·δ)   otherwise
 * </pre>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new HuberLoss(delta = 1.0f);
 * GradTensor l = loss.forward(pred, target);
 * }</pre>
 */
public final class HuberLoss {

    private final float delta;

    /** Creates a Huber loss with default δ=1.0. */
    public HuberLoss() {
        this(1.0f);
    }

    /**
     * Creates a Huber loss with custom δ.
     *
     * @param delta threshold between quadratic and linear regions (default 1.0)
     */
    public HuberLoss(float delta) {
        if (!Float.isFinite(delta) || delta <= 0.0f) {
            throw new IllegalArgumentException("delta must be finite and positive, got: " + delta);
        }
        this.delta = delta;
    }

    /**
     * Computes the mean Huber loss.
     *
     * @param pred   predictions (any shape)
     * @param target ground-truth values (same shape)
     * @return scalar mean Huber loss
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        return compute(pred, target);
    }

    /**
     * Computes the mean Huber loss and attaches the gradient to predictions.
     *
     * @param pred   predictions (any shape)
     * @param target ground-truth values (same shape)
     * @return scalar mean Huber loss
     */
    public GradTensor compute(GradTensor pred, GradTensor target) {
        int n = RegressionLosses.requireSameFiniteNonEmpty(pred, target, "HuberLoss");
        float[] p = pred.data(), t = target.data();
        float[] losses = new float[n];
        for (int i = 0; i < n; i++) {
            float diff = Math.abs(p[i] - t[i]);
            losses[i] = diff <= delta
                    ? 0.5f * diff * diff
                    : delta * (diff - 0.5f * delta);
        }
        float total = 0.0f;
        for (float loss : losses) {
            total += loss;
        }

        GradTensor out = GradTensor.scalar(total / n);
        if (pred.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("HuberLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        float diff = p[i] - t[i];
                        float absDiff = Math.abs(diff);
                        if (absDiff <= delta) {
                            grad[i] = diff * scale;
                        } else {
                            grad[i] = (diff > 0 ? delta : -delta) * scale;
                        }
                    }
                    pred.backward(GradTensor.of(grad, pred.shape()));
                }
            });
        }
        return out;
    }

    public float delta() {
        return delta;
    }
}
