package tech.kayys.gollek.ml.bytelatent;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Lightweight executable trainer session for byte-latent data flows.
 */
public final class ByteLatentTrainer {
    private ByteLatentTrainer() {
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private TextByteSequenceDataset dataset;
        private ByteLatentTrainerConfig config;
        private ByteLatentBatchLossEvaluator lossEvaluator;
        private ByteLatentModel model;
        private List<ByteLatentTrainingListener> listeners = List.of();

        private Builder() {
        }

        public Builder dataset(TextByteSequenceDataset dataset) {
            this.dataset = dataset;
            return this;
        }

        public Builder config(ByteLatentTrainerConfig config) {
            this.config = config;
            return this;
        }

        public Builder lossEvaluator(ByteLatentBatchLossEvaluator lossEvaluator) {
            this.lossEvaluator = lossEvaluator;
            return this;
        }

        public Builder model(ByteLatentModel model) {
            this.model = model;
            return this;
        }

        public Builder listeners(List<ByteLatentTrainingListener> listeners) {
            this.listeners = List.copyOf(Objects.requireNonNull(listeners, "listeners must not be null"));
            return this;
        }

        public ByteLatentTrainerSession build() {
            TextByteSequenceDataset resolvedDataset =
                    Objects.requireNonNull(dataset, "dataset must not be null");
            ByteLatentTrainerConfig resolvedConfig =
                    Objects.requireNonNull(config, "config must not be null");
            ByteLatentModel resolvedModel =
                    model != null ? model : new ReferenceByteLatentModel(resolvedConfig.modelSpec());
            ByteLatentBatchLossEvaluator resolvedEvaluator =
                    lossEvaluator != null
                            ? lossEvaluator
                            : (batch, ignoredConfig, epoch, batchIndex) -> resolvedModel.forward(batch).meanLoss();
            return new DefaultSession(resolvedDataset, resolvedConfig, resolvedEvaluator, resolvedModel, listeners);
        }
    }

    private static final class DefaultSession implements ByteLatentTrainerSession {
        private final TextByteSequenceDataset dataset;
        private final ByteLatentTrainerConfig config;
        private final ByteLatentBatchLossEvaluator lossEvaluator;
        private final ByteLatentModel model;
        private final List<ByteLatentTrainingListener> listeners;
        private final ByteLatentCheckpointIO.ResumeState resumeState;
        private final java.util.List<ByteLatentHistoryRow> historyRows;
        private volatile boolean stopped;
        private volatile int currentEpoch;
        private volatile int globalStep;
        private volatile TrainingSummary summary;

        private DefaultSession(
                TextByteSequenceDataset dataset,
                ByteLatentTrainerConfig config,
                ByteLatentBatchLossEvaluator lossEvaluator,
                ByteLatentModel model,
                List<ByteLatentTrainingListener> listeners) {
            this.dataset = dataset;
            this.config = config;
            this.lossEvaluator = lossEvaluator;
            this.model = model;
            this.listeners = List.copyOf(listeners);
            this.resumeState = config.resumeFromCheckpoint()
                    ? ByteLatentCheckpointIO.loadResumeState(config.checkpointDir())
                    : ByteLatentCheckpointIO.ResumeState.notRequested();
            this.historyRows = new java.util.ArrayList<>(resumeState.historyRows());
            if (resumeState.loaded()) {
                this.summary = resumeState.summary();
                this.currentEpoch = resumeState.summary().epochCount();
                this.globalStep = resumeState.globalStep();
            } else {
                this.summary = new TrainingSummary(0, Double.NaN, 0, null, null, 0L, Map.of());
            }
        }

        @Override
        public ByteLatentTrainerConfig config() {
            return config;
        }

        @Override
        public int currentEpoch() {
            return currentEpoch;
        }

        @Override
        public int globalStep() {
            return globalStep;
        }

        @Override
        public TrainingSummary fit() {
            long startedAt = System.nanoTime();
            double bestLoss = resumeState.loaded()
                    ? summary.bestValidationLoss()
                    : Double.POSITIVE_INFINITY;
            int bestEpoch = resumeState.loaded()
                    ? summary.bestValidationEpoch()
                    : 0;
            Double latestTrainLoss = resumeState.loaded()
                    ? summary.latestTrainLoss()
                    : null;
            int startEpoch = resumeState.loaded()
                    ? Math.min(config.epochs(), summary.epochCount()) + 1
                    : 1;
            notifyListenersSafely(listener -> listener.onTrainingStart(this));
            try {
                for (int epoch = startEpoch; epoch <= config.epochs() && !stopped; epoch++) {
                    currentEpoch = epoch;
                    int epochNumber = epoch;
                    notifyListenersSafely(listener -> listener.onEpochStart(this, epochNumber));
                    ByteLatentTrainingPlan plan = ByteLatentTrainerSupport.plan(dataset, config);
                    double totalLoss = 0.0d;
                    int seenBatches = 0;
                    int batchIndex = 0;
                    for (ByteSequenceWindowBatch batch : plan.batches()) {
                        if (stopped) {
                            break;
                        }
                        int step = globalStep;
                        notifyListenersSafely(listener -> listener.onBatchStart(this, step));
                        double loss = lossEvaluator.evaluate(batch, config, epochNumber, batchIndex);
                        totalLoss += loss;
                        seenBatches++;
                        batchIndex++;
                        globalStep++;
                        notifyListenersSafely(listener -> listener.onBatchEnd(this, globalStep, loss));
                    }
                    latestTrainLoss = seenBatches == 0 ? 0.0d : totalLoss / seenBatches;
                    if (latestTrainLoss < bestLoss) {
                        bestLoss = latestTrainLoss;
                        bestEpoch = epochNumber;
                    }
                    historyRows.add(new ByteLatentHistoryRow(epochNumber, globalStep, seenBatches, latestTrainLoss));
                    double epochLoss = latestTrainLoss;
                    notifyListenersSafely(listener -> listener.onEpochEnd(this, epochNumber, epochLoss));
                    summary = buildSummary(startedAt, bestLoss, bestEpoch, latestTrainLoss);
                    persistCheckpointSafely(summary);
                }
            } catch (Exception error) {
                notifyListenersSafely(listener -> listener.onTrainingError(this, error));
                throw error;
            }
            if (currentEpoch == 0) {
                summary = buildSummary(startedAt, Double.NaN, 0, null);
            }
            notifyListenersSafely(listener -> listener.onTrainingEnd(this, summary));
            return summary;
        }

        @Override
        public TrainingSummary summary() {
            return summary;
        }

        @Override
        public boolean isStopped() {
            return stopped;
        }

        @Override
        public void stop() {
            stopped = true;
        }

        private void persistCheckpointSafely(TrainingSummary summary) {
            if (config.checkpointDir() == null) {
                return;
            }
            try {
                ByteLatentCheckpointIO.persistSummary(config.checkpointDir(), summary, historyRows);
            } catch (Exception error) {
                Map<String, Object> metadata = new LinkedHashMap<>(summary.metadata());
                metadata.put("checkpointSaveFailed", true);
                metadata.put("checkpointError", error.getMessage());
                this.summary = new TrainingSummary(
                        summary.epochCount(),
                        summary.bestValidationLoss(),
                        summary.bestValidationEpoch(),
                        summary.latestTrainLoss(),
                        summary.latestValidationLoss(),
                        summary.durationMs(),
                        Map.copyOf(metadata));
            }
        }

        private TrainingSummary buildSummary(
                long startedAt,
                double bestLoss,
                int bestEpoch,
                Double latestTrainLoss) {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("familyId", ByteLatentModelFamily.FAMILY_ID);
            metadata.put("datasetSize", dataset.size());
            metadata.put("batchSize", config.batchSize());
            metadata.put("windowLength", config.windowLength());
            metadata.put("shuffle", config.shuffle());
            metadata.put("seed", config.seed());
            metadata.put("globalStep", globalStep);
            metadata.put("stopped", stopped);
            metadata.put("lossMode", model != null ? "model-forward" : "batch-evaluator");
            metadata.put("listenerCount", listeners.size());
            if (model != null) {
                metadata.put("modelClass", model.getClass().getSimpleName());
                metadata.put("modelFamilyId", model.familyId());
                metadata.put("modelVocabularySize", model.spec().byteVocabularySize());
            }
            metadata.put("checkpointEnabled", config.checkpointDir() != null);
            metadata.put("historyRowCount", historyRows.size());
            metadata.put("resumeRequested", config.resumeFromCheckpoint());
            metadata.put("resumeLoaded", resumeState.loaded());
            metadata.put("resumeMissing", resumeState.missing());
            metadata.put("resumeFailed", resumeState.failed());
            metadata.put("historyLoaded", resumeState.loaded() && !resumeState.historyRows().isEmpty());
            if (resumeState.detail() != null) {
                metadata.put("resumeDetail", resumeState.detail());
            }
            if (config.checkpointDir() != null) {
                metadata.put("checkpointDir", config.checkpointDir().toString());
                metadata.put("summaryFile", config.checkpointDir().resolve(ByteLatentCheckpointIO.SUMMARY_FILE_NAME).toString());
                metadata.put("checkpointManifestFile", config.checkpointDir().resolve(ByteLatentCheckpointIO.MANIFEST_FILE_NAME).toString());
                metadata.put("historyFile", config.checkpointDir().resolve(ByteLatentCheckpointIO.HISTORY_FILE_NAME).toString());
                metadata.put("reportFile", config.checkpointDir().resolve(ByteLatentCheckpointIO.REPORT_FILE_NAME).toString());
            }
            return new TrainingSummary(
                    currentEpoch,
                    bestLoss,
                    bestEpoch,
                    latestTrainLoss,
                    null,
                    (System.nanoTime() - startedAt) / 1_000_000L,
                    Map.copyOf(metadata));
        }

        private void notifyListenersSafely(ListenerConsumer consumer) {
            for (ByteLatentTrainingListener listener : listeners) {
                try {
                    consumer.accept(listener);
                } catch (Exception ignored) {
                    // Listener failures are non-fatal in this lightweight runtime.
                }
            }
        }
    }

    @FunctionalInterface
    private interface ListenerConsumer {
        void accept(ByteLatentTrainingListener listener) throws Exception;
    }
}
