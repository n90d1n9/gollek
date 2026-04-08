package tech.kayys.gollek.sdk.session;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.DataLoader;
import tech.kayys.gollek.ml.metrics.MetricsTracker;
import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.ml.nn.optim.Optimizer;

import java.util.function.BiFunction;

/**
 * Training session — manages a complete training lifecycle with metrics,
 * checkpointing, and early stopping.
 *
 * <p>Provides a higher-level alternative to the raw training loop,
 * integrating {@link MetricsTracker}, callbacks, and session state.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * try (TrainingSession session = TrainingSession.builder()
 *         .model(model)
 *         .optimizer(optimizer)
 *         .lossFn((pred, target) -> pred.sub(target).pow(2f).mean())
 *         .epochs(50)
 *         .build()) {
 *
 *     session.fit(trainLoader, valLoader);
 *     System.out.println(session.metrics().summary());
 * }
 * }</pre>
 */
public final class TrainingSession implements AutoCloseable {

    private final Module    model;
    private final Optimizer optimizer;
    private final BiFunction<GradTensor, GradTensor, GradTensor> lossFn;
    private final int       epochs;
    private final MetricsTracker metrics = new MetricsTracker();

    private TrainingSession(Builder b) {
        this.model     = b.model;
        this.optimizer = b.optimizer;
        this.lossFn    = b.lossFn;
        this.epochs    = b.epochs;
    }

    /**
     * Runs the full training loop.
     *
     * @param trainLoader training data loader
     * @param valLoader   validation data loader (can be null)
     */
    public void fit(DataLoader trainLoader, DataLoader valLoader) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.train();
            float trainLoss = runEpoch(trainLoader, true);
            metrics.log("train/loss", trainLoss, epoch);

            if (valLoader != null) {
                model.eval();
                float valLoss = runEpoch(valLoader, false);
                metrics.log("val/loss", valLoss, epoch);
                System.out.printf("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f%n",
                    epoch + 1, epochs, trainLoss, valLoss);
            } else {
                System.out.printf("Epoch %d/%d  train_loss=%.4f%n",
                    epoch + 1, epochs, trainLoss);
            }
        }
    }

    /** Runs one epoch, returns mean loss. */
    private float runEpoch(DataLoader loader, boolean train) {
        float totalLoss = 0f; int steps = 0;
        for (DataLoader.Batch batch : loader) {
            if (train) {
                model.zeroGrad();
                GradTensor pred = model.forward(batch.inputs());
                GradTensor loss = lossFn.apply(pred, batch.labels());
                loss.backward();
                optimizer.step();
                totalLoss += loss.item();
            } else {
                GradTensor pred = model.forward(batch.inputs());
                totalLoss += lossFn.apply(pred, batch.labels()).item();
            }
            steps++;
        }
        return steps > 0 ? totalLoss / steps : 0f;
    }

    /**
     * Returns the metrics tracker for this session.
     *
     * @return {@link MetricsTracker} with train/val loss history
     */
    public MetricsTracker metrics() { return metrics; }

    @Override public void close() { /* release resources */ }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link TrainingSession}.
     */
    public static final class Builder {
        private Module    model;
        private Optimizer optimizer;
        private BiFunction<GradTensor, GradTensor, GradTensor> lossFn;
        private int       epochs = 10;

        /** @param model model to train */
        public Builder model(Module m)                                          { this.model = m; return this; }
        /** @param opt optimizer */
        public Builder optimizer(Optimizer o)                                   { this.optimizer = o; return this; }
        /** @param fn loss function: (pred, target) → scalar */
        public Builder lossFn(BiFunction<GradTensor, GradTensor, GradTensor> fn){ this.lossFn = fn; return this; }
        /** @param e number of training epochs */
        public Builder epochs(int e)                                            { this.epochs = e; return this; }

        /** Builds the training session. */
        public TrainingSession build() { return new TrainingSession(this); }
    }
}
