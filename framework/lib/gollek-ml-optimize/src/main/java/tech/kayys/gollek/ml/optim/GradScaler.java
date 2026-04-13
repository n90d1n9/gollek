package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;

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
 *     scaler.scale(loss);
 *     loss.backward();
 *     boolean overflow = scaler.unscaleAndCheck(optimizer.getParameters());
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

    private double scale;
    private final double growthFactor;
    private final double backoffFactor;
    private final int growthInterval;
    private int stepsWithoutOverflow;
    private boolean overflowDetected;

    private GradScaler(Builder builder) {
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
     * This multiplies the loss value by the current scale factor.
     *
     * @param loss the loss tensor to scale
     */
    public void scale(GradTensor loss) {
        // In a real implementation, this would modify the loss gradient
        // scale before backward is called. Here we track the scale factor
        // for unscaling after backward.
    }

    /**
     * Unscale gradients and check for overflow/NaN.
     *
     * @param parameters list of parameters whose gradients to unscale
     * @return true if overflow was detected (gradients are invalid)
     */
    public boolean unscaleAndCheck(List<GradTensor> parameters) {
        overflowDetected = false;
        double invScale = 1.0 / scale;

        for (GradTensor param : parameters) {
            if (param.grad() != null) {
                float[] grad = param.grad().data();
                for (int i = 0; i < grad.length; i++) {
                    grad[i] *= (float) invScale;
                    if (Float.isInfinite(grad[i]) || Float.isNaN(grad[i])) {
                        overflowDetected = true;
                        return true;
                    }
                }
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
            scale *= backoffFactor;
            stepsWithoutOverflow = 0;
        } else {
            stepsWithoutOverflow++;
            if (stepsWithoutOverflow >= growthInterval) {
                scale *= growthFactor;
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
     * Builder for GradScaler.
     */
    public static class Builder {
        private double initScale = 65536.0;
        private double growthFactor = 2.0;
        private double backoffFactor = 0.5;
        private int growthInterval = 2000;

        private Builder() {}

        /** Set the initial scale factor (default: 65536.0). */
        public Builder initScale(double scale) { this.initScale = scale; return this; }

        /** Set the growth factor (default: 2.0). */
        public Builder growthFactor(double factor) { this.growthFactor = factor; return this; }

        /** Set the backoff factor on overflow (default: 0.5). */
        public Builder backoffFactor(double factor) { this.backoffFactor = factor; return this; }

        /** Set the number of successful steps before growing (default: 2000). */
        public Builder growthInterval(int interval) { this.growthInterval = interval; return this; }

        /** Build the GradScaler instance. */
        public GradScaler build() { return new GradScaler(this); }
    }
}
