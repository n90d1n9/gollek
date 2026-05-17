package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;
import java.util.Objects;

/**
 * Gradient scaler for mixed-precision (FP16/BF16) training.
 *
 * <p>Scales the loss before backward to prevent gradient underflow in
 * lower-precision formats, then unscales before the optimizer step.
 * Dynamically adjusts the scale factor based on overflow detection.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * var scaler = GradScaler.builder()
 *         .initScale(65536.0)
 *         .growthInterval(2000)
 *         .build();
 *
 * for (var batch : loader) {
 *     var loss = model.forward(batch.inputs());
 *     var scaledLoss = scaler.scale(loss);
 *     scaledLoss.backward();
 *     boolean overflow = scaler.unscaleAndCheck(optimizer);
 *     if (!overflow) { scaler.step(optimizer); }
 *     scaler.update();
 *     optimizer.zeroGrad();
 * }
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class GradScaler {

    private static final double MIN_REPRESENTABLE_SCALE = Float.MIN_NORMAL;
    private static final double MAX_REPRESENTABLE_SCALE = Float.MAX_VALUE;

    private double scale;
    private final double growthFactor;
    private final double backoffFactor;
    private final int growthInterval;
    private int stepsWithoutOverflow;
    private boolean overflowDetected;

    private GradScaler(Builder builder) {
        requireRepresentableScale(builder.initScale, "initScale");
        if (!Double.isFinite(builder.growthFactor) || builder.growthFactor <= 1.0) {
            throw new IllegalArgumentException("growthFactor must be finite and > 1.0");
        }
        if (!Double.isFinite(builder.backoffFactor) || builder.backoffFactor <= 0.0 || builder.backoffFactor >= 1.0) {
            throw new IllegalArgumentException("backoffFactor must be finite and in (0.0, 1.0)");
        }
        if (builder.growthInterval <= 0) {
            throw new IllegalArgumentException("growthInterval must be positive");
        }
        this.scale = builder.initScale;
        this.growthFactor = builder.growthFactor;
        this.backoffFactor = builder.backoffFactor;
        this.growthInterval = builder.growthInterval;
        this.stepsWithoutOverflow = 0;
        this.overflowDetected = false;
    }

    /**
     * Create a new builder for constructing GradScaler instances.
     *
     * @return new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Scale the loss tensor for mixed-precision backward pass.
     * This returns a differentiable tensor, so callers must backpropagate the
     * returned value.
     *
     * @param loss the loss tensor to scale
     * @return differentiable scaled loss tensor
     */
    public GradTensor scale(GradTensor loss) {
        Objects.requireNonNull(loss, "loss");
        return loss.mul(checkedScaleAsFloat());
    }

    /**
     * Unscale gradients owned by an optimizer and check for overflow/NaN.
     *
     * @param optimizer optimizer whose parameter gradients should be unscaled
     * @return true if overflow was detected (gradients are invalid)
     */
    public boolean unscaleAndCheck(Optimizer optimizer) {
        Objects.requireNonNull(optimizer, "optimizer");
        return unscaleAndCheckParameters(optimizer.parameters());
    }

    /**
     * Unscale parameter gradients and check for overflow/NaN.
     *
     * @param parameters list of parameters whose gradients to unscale
     * @return true if overflow was detected (gradients are invalid)
     */
    public boolean unscaleAndCheckParameters(List<Parameter> parameters) {
        Objects.requireNonNull(parameters, "parameters");
        overflowDetected = false;
        double invScale = inverseScale();

        for (Parameter parameter : parameters) {
            if (parameter != null && parameter.grad() != null
                    && unscaleGradient(parameter.grad().data(), invScale)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Unscale gradients and check for overflow/NaN.
     *
     * @param parameters list of parameters whose gradients to unscale
     * @return true if overflow was detected (gradients are invalid)
     */
    public boolean unscaleAndCheck(List<GradTensor> parameters) {
        Objects.requireNonNull(parameters, "parameters");
        overflowDetected = false;
        double invScale = inverseScale();

        for (GradTensor param : parameters) {
            if (param != null && param.grad() != null
                    && unscaleGradient(param.grad().data(), invScale)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Perform the optimizer step if no overflow was detected.
     *
     * @param optimizer the optimizer to step
     */
    public void step(Optimizer optimizer) {
        if (!overflowDetected) {
            optimizer.step();
        }
    }

    /**
     * Update the scale factor based on overflow history.
     * Grows the scale if no overflows occurred for {@code growthInterval} steps.
     * Shrinks it immediately on overflow.
     */
    public void update() {
        if (overflowDetected) {
            scale = Math.max(scale * backoffFactor, MIN_REPRESENTABLE_SCALE);
            stepsWithoutOverflow = 0;
        } else {
            stepsWithoutOverflow++;
            if (stepsWithoutOverflow >= growthInterval) {
                scale = Math.min(scale * growthFactor, MAX_REPRESENTABLE_SCALE);
                stepsWithoutOverflow = 0;
            }
        }
    }

    /**
     * Get the current loss scale factor.
     *
     * @return current scale
     */
    public double getScale() {
        return scale;
    }

    /**
     * Returns whether the last unscale pass found invalid gradients.
     *
     * @return true when the last optimizer step should be skipped
     */
    public boolean overflowDetected() {
        return overflowDetected;
    }

    private boolean unscaleGradient(float[] grad, double invScale) {
        for (int i = 0; i < grad.length; i++) {
            grad[i] *= (float) invScale;
            if (!Float.isFinite(grad[i])) {
                overflowDetected = true;
                return true;
            }
        }
        return false;
    }

    private double inverseScale() {
        requireRepresentableScale(scale, "scale");
        return 1.0 / scale;
    }

    private float checkedScaleAsFloat() {
        requireRepresentableScale(scale, "scale");
        return (float) scale;
    }

    private static void requireRepresentableScale(double value, String name) {
        if (!Double.isFinite(value)
                || value < MIN_REPRESENTABLE_SCALE
                || value > MAX_REPRESENTABLE_SCALE) {
            throw new IllegalArgumentException(
                    name + " must be finite and representable as a positive float scale");
        }
    }

    /**
     * Builder for GradScaler.
     */
    public static class Builder {
        private double initScale = 65536.0;
        private double growthFactor = 2.0;
        private double backoffFactor = 0.5;
        private int growthInterval = 2000;

        private Builder() {}

        /** Set the initial scale factor (default: 65536.0). */
        public Builder initScale(double scale) {
            requireRepresentableScale(scale, "initScale");
            this.initScale = scale;
            return this;
        }

        /** Set the growth factor (default: 2.0). */
        public Builder growthFactor(double factor) {
            if (!Double.isFinite(factor) || factor <= 1.0) {
                throw new IllegalArgumentException("growthFactor must be finite and > 1.0");
            }
            this.growthFactor = factor;
            return this;
        }

        /** Set the backoff factor on overflow (default: 0.5). */
        public Builder backoffFactor(double factor) {
            if (!Double.isFinite(factor) || factor <= 0.0 || factor >= 1.0) {
                throw new IllegalArgumentException("backoffFactor must be finite and in (0.0, 1.0)");
            }
            this.backoffFactor = factor;
            return this;
        }

        /** Set the number of successful steps before growing (default: 2000). */
        public Builder growthInterval(int interval) {
            if (interval <= 0) {
                throw new IllegalArgumentException("growthInterval must be positive");
            }
            this.growthInterval = interval;
            return this;
        }

        /** Build the GradScaler instance. */
        public GradScaler build() { return new GradScaler(this); }
    }
}
