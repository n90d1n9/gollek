package tech.kayys.gollek.ml.optim;

import java.util.Map;

/**
 * Learning rate scheduler base class.
 * <p>
 * Learning rate schedulers adjust the learning rate during training to improve convergence.
 * They are typically used with optimizers to modify the learning rate at specific intervals
 * (e.g., per epoch, per batch, or based on validation metrics).
 * <p>
 * Common scheduling strategies include step decay, exponential decay, and cosine annealing.
 *
 * <h3>Usage Pattern</h3>
 * <pre>{@code
 * var optimizer = new AdamW(model.parameters(), 0.001f);
 * var scheduler = new StepLR(optimizer, 10, 0.1f);  // decay every 10 epochs
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Training loop
 *     for (var batch : dataloader) {
 *         // Forward, backward, step...
 *     }
 *     scheduler.step();  // Update learning rate
 * }
 * }</pre>
 */
public abstract class LRScheduler {

    protected final Optimizer optimizer;

    /**
     * Create a learning rate scheduler.
     *
     * @param optimizer the optimizer whose learning rate to schedule
     *
     * @throws IllegalArgumentException if optimizer is null
     */
    public LRScheduler(Optimizer optimizer) {
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        this.optimizer = optimizer;
    }

    /**
     * Perform a learning rate schedule step.
     * <p>
     * Typically called once per epoch or per batch, depending on the scheduler type.
     * Updates the optimizer's learning rate according to the schedule.
     */
    public abstract void step();

    /**
     * Perform a learning rate schedule step using a monitored validation value.
     *
     * <p>Schedulers that react to validation loss or metrics should override
     * this method. Step-based schedulers keep their existing behavior by
     * delegating to {@link #step()}.</p>
     *
     * @param metric monitored value for this step, such as validation loss
     */
    public void step(double metric) {
        step();
    }

    /**
     * Returns the current learning rate managed by this scheduler.
     *
     * @return current learning rate
     */
    public abstract float getLr();

    /**
     * Get current learning rate from optimizer.
     *
     * @return current learning rate
     */
    protected float getLearningRate() {
        return optimizer.learningRate();
    }

    /**
     * Set learning rate on optimizer.
     *
     * @param lr new learning rate
     */
    protected void setLearningRate(float lr) {
        optimizer.setLearningRate(lr);
    }

    /**
     * Indicates whether this scheduler can export/import progress state.
     *
     * <p>Schedulers used with checkpoint resume should override this and
     * return {@code true}; custom schedulers can still be used without
     * checkpoint persistence.</p>
     */
    public boolean supportsStateDict() {
        return false;
    }

    /**
     * Export scheduler progress into a serializable map.
     *
     * <p>The default implementation is intentionally empty for custom
     * schedulers that do not expose checkpoint state yet.</p>
     */
    public Map<String, Object> stateDict() {
        return Map.of();
    }

    /**
     * Restore scheduler progress from a previously exported state map.
     *
     * <p>The default implementation is a no-op.</p>
     */
    public void loadStateDict(Map<String, Object> state) {
        // no-op by default
    }
}
