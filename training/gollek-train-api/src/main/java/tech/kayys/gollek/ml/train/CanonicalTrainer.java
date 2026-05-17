package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.optim.GradientClipper;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.CanonicalTrainerRuntime;
import tech.kayys.gollek.trainer.Trainers;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Typed bridge that executes real forward/loss/backward steps using the
 * canonical trainer runtime.
 */
public final class CanonicalTrainer implements AutoCloseable {

    private static final String MODEL_CHECKPOINT_FILE_NAME = "canonical-model.safetensors";
    private static final String BEST_MODEL_CHECKPOINT_FILE_NAME = "canonical-best-model.safetensors";
    private static final String OPTIMIZER_CHECKPOINT_FILE_NAME = "canonical-optimizer.state";
    private static final String SCHEDULER_CHECKPOINT_FILE_NAME = "canonical-scheduler.state";
    private static final String HISTORY_CHECKPOINT_FILE_NAME = "canonical-history.csv";
    private static final String REPORT_CHECKPOINT_FILE_NAME = "canonical-report.json";

    private final NNModule model;
    private final Optimizer optimizer;
    private final LRScheduler learningRateScheduler;
    private final SchedulerStepUnit schedulerStepUnit;
    private final String schedulerMonitorMetric;
    private final LossFunction lossFunction;
    private final List<Metric> trainMetrics;
    private final List<Metric> validationMetrics;
    private final double gradientClip;
    private final int earlyStoppingPatience;
    private final double earlyStoppingMinDelta;
    private final String earlyStoppingMonitorMetric;
    private final BestModelMonitorMode earlyStoppingMonitorMode;
    private final int gradientAccumulationSteps;
    private final boolean resumeFromCheckpoint;
    private final boolean failOnCheckpointLoadError;
    private final boolean saveBestModelCheckpoint;
    private final boolean restoreBestModelAtEnd;
    private final String bestModelMonitorMetric;
    private final BestModelMonitorMode bestModelMonitorMode;
    private final String preferredDevice;
    private final Path modelCheckpointFile;
    private final Path bestModelCheckpointFile;
    private final Path optimizerCheckpointFile;
    private final Path schedulerCheckpointFile;
    private final Path historyFile;
    private final Path reportFile;
    private final AtomicBoolean modelCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean optimizerCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean schedulerCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean historyLoadAttempted = new AtomicBoolean(false);
    private volatile boolean modelCheckpointLoaded;
    private volatile boolean modelCheckpointSaved;
    private volatile boolean bestModelCheckpointSaved;
    private volatile boolean bestModelCheckpointRestored;
    private volatile boolean optimizerCheckpointLoaded;
    private volatile boolean optimizerCheckpointSaved;
    private volatile boolean schedulerCheckpointLoaded;
    private volatile boolean schedulerCheckpointSaved;
    private volatile boolean historyLoaded;
    private volatile boolean historySaved;
    private volatile boolean reportSaved;
    private volatile String modelCheckpointLoadError;
    private volatile String modelCheckpointSaveError;
    private volatile String bestModelCheckpointSaveError;
    private volatile String bestModelCheckpointLoadError;
    private volatile String optimizerCheckpointLoadError;
    private volatile String optimizerCheckpointSaveError;
    private volatile String schedulerCheckpointLoadError;
    private volatile String schedulerCheckpointSaveError;
    private volatile String historyLoadError;
    private volatile String historySaveError;
    private volatile String reportSaveError;
    private volatile int schedulerStepCount;
    private volatile int pendingGradientAccumulationBatches;
    private volatile int optimizerStepCount;
    private volatile int bestModelCheckpointEpoch = -1;
    private volatile double bestModelCheckpointValidationLoss = Double.NaN;
    private volatile double bestModelCheckpointMonitorValue = Double.NaN;
    private volatile int earlyStoppingMonitorBestEpoch = -1;
    private volatile int earlyStoppingMonitorEpochsWithoutImprovement;
    private volatile int earlyStoppingMonitorStopEpoch = -1;
    private volatile double earlyStoppingMonitorBestValue = Double.NaN;
    private volatile double earlyStoppingMonitorLatestValue = Double.NaN;
    private volatile boolean earlyStoppingMonitorTriggered;
    private volatile Acceleration.BackendStatus accelerationStatusAtStart;
    private volatile Map<String, Double> latestTrainMetrics = Map.of();
    private volatile Map<String, Double> latestValidationMetrics = Map.of();
    private volatile Map<String, Object> latestTrainMetricDetails = Map.of();
    private volatile Map<String, Object> latestValidationMetricDetails = Map.of();
    private volatile double latestGradientL2NormBeforeClip = 0.0;
    private volatile double latestGradientL2Norm = 0.0;
    private volatile double latestGradientMaxAbsBeforeClip = 0.0;
    private volatile double latestGradientMaxAbs = 0.0;
    private volatile int latestGradientParameterCount;
    private volatile long latestGradientValueCount;
    private volatile boolean latestGradientClipped;
    private volatile double latestParameterL2Norm = 0.0;
    private volatile double latestParameterMaxAbs = 0.0;
    private volatile int latestParameterCount;
    private volatile long latestParameterValueCount;
    private volatile long trainBatchCount;
    private volatile long validationBatchCount;
    private volatile long trainSampleCount;
    private volatile long validationSampleCount;
    private volatile long trainInputElementCount;
    private volatile long validationInputElementCount;
    private volatile long trainLabelElementCount;
    private volatile long validationLabelElementCount;
    private volatile long trainComputeNanos;
    private volatile long validationComputeNanos;
    private volatile long epochTrainBatchCount;
    private volatile long epochValidationBatchCount;
    private volatile long epochTrainSampleCount;
    private volatile long epochValidationSampleCount;
    private volatile long epochTrainInputElementCount;
    private volatile long epochValidationInputElementCount;
    private volatile long epochTrainLabelElementCount;
    private volatile long epochValidationLabelElementCount;
    private volatile long epochTrainComputeNanos;
    private volatile long epochValidationComputeNanos;
    private volatile boolean nonFiniteDetected;
    private volatile String nonFinitePhase;
    private volatile String nonFiniteKind;
    private volatile double nonFiniteValue = Double.NaN;
    private volatile String nonFiniteMessage;
    private volatile boolean nonFiniteOptimizerStepSkipped;
    private final List<Map<String, Object>> epochHistory = new ArrayList<>();
    private final CanonicalTrainerRuntime runtime;
    private volatile TrainingSummary latestSummary;

    private CanonicalTrainer(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model must not be null");
        this.optimizer = Objects.requireNonNull(builder.optimizer, "optimizer must not be null");
        this.learningRateScheduler = builder.learningRateScheduler;
        this.schedulerStepUnit = builder.schedulerStepUnit;
        this.schedulerMonitorMetric = normalizeBestModelMonitorMetric(builder.schedulerMonitorMetric);
        this.lossFunction = Objects.requireNonNull(builder.lossFunction, "loss function must not be null");
        this.trainMetrics = instantiateMetrics(builder.metricFactories, "train");
        this.validationMetrics = instantiateMetrics(builder.metricFactories, "validation");
        this.gradientClip = Math.max(0.0, builder.gradientClip);
        this.earlyStoppingPatience = Math.max(0, builder.earlyStoppingPatience);
        this.earlyStoppingMinDelta = Math.max(0.0, builder.earlyStoppingMinDelta);
        this.earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(builder.earlyStoppingMonitorMetric);
        this.earlyStoppingMonitorMode = builder.earlyStoppingMonitorMode == null
                ? (this.earlyStoppingMonitorMetric == null ? BestModelMonitorMode.MIN : BestModelMonitorMode.MAX)
                : builder.earlyStoppingMonitorMode;
        this.gradientAccumulationSteps = Math.max(1, builder.gradientAccumulationSteps);
        this.resumeFromCheckpoint = builder.resumeFromCheckpoint;
        this.failOnCheckpointLoadError = builder.failOnCheckpointLoadError;
        this.saveBestModelCheckpoint = builder.saveBestModelCheckpoint;
        this.restoreBestModelAtEnd = builder.restoreBestModelAtEnd;
        this.bestModelMonitorMetric = normalizeBestModelMonitorMetric(builder.bestModelMonitorMetric);
        this.bestModelMonitorMode = builder.bestModelMonitorMode == null
                ? (this.bestModelMonitorMetric == null ? BestModelMonitorMode.MIN : BestModelMonitorMode.MAX)
                : builder.bestModelMonitorMode;
        requireBestModelMonitorMetricPresent(this.bestModelMonitorMetric, this.validationMetrics);
        requireBestModelMonitorMetricPresent(this.earlyStoppingMonitorMetric, this.validationMetrics);
        requireBestModelMonitorMetricPresent(this.schedulerMonitorMetric, this.validationMetrics);
        this.preferredDevice = normalizeDevice(builder.preferredDevice);
        this.modelCheckpointFile = resolveModelCheckpointFile(builder.checkpointDir);
        this.bestModelCheckpointFile = resolveBestModelCheckpointFile(builder.checkpointDir);
        this.optimizerCheckpointFile = resolveOptimizerCheckpointFile(builder.checkpointDir);
        this.schedulerCheckpointFile = resolveSchedulerCheckpointFile(builder.checkpointDir);
        this.historyFile = resolveHistoryFile(builder.checkpointDir);
        this.reportFile = resolveReportFile(builder.checkpointDir);

        int runtimeEarlyStoppingPatience = this.earlyStoppingMonitorMetric == null
                ? this.earlyStoppingPatience
                : 0;
        CanonicalTrainerRuntime.Builder runtimeBuilder = Trainers.canonicalBuilder()
                .model(model)
                .optimizer(optimizer)
                .epochs(builder.epochs)
                .gradientClip(this.gradientClip)
                .mixedPrecision(builder.mixedPrecision)
                .earlyStopping(runtimeEarlyStoppingPatience, this.earlyStoppingMinDelta)
                .resumeFromCheckpoint(builder.resumeFromCheckpoint)
                .failOnCheckpointLoadError(builder.failOnCheckpointLoadError)
                .trainBatchLoss((session, epoch, step, batch) -> runTrainingBatch(batch))
                .validationBatchLoss((session, epoch, step, batch) -> runValidationBatch(batch))
                .listener(new TrainingListener() {
                    @Override
                    public void onEpochStart(TrainerSession session, int epoch) {
                        resetMetrics();
                        resetEpochThroughputCounters();
                    }

                    @Override
                    public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                        flushGradientAccumulation(session);
                        latestTrainMetrics = snapshotMetrics(trainMetrics);
                        latestTrainMetricDetails = snapshotMetricDetails(trainMetrics);
                        stepScheduler(SchedulerStepUnit.EPOCH);
                        recordEpochHistory(epoch, trainLoss);
                        persistCheckpointsSafely();
                        persistReportSafely(runtime.summary());
                    }

                    @Override
                    public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                        latestValidationMetrics = snapshotMetrics(validationMetrics);
                        latestValidationMetricDetails = snapshotMetricDetails(validationMetrics);
                        persistBestModelCheckpointIfImproved(epoch, valLoss);
                        updateCustomEarlyStoppingMonitor(session, epoch, valLoss);
                        stepScheduler(SchedulerStepUnit.VALIDATION, schedulerMonitorValue(valLoss));
                        updateEpochHistoryValidation(epoch, valLoss);
                        persistHistorySafely();
                        persistReportSafely(runtime.summary());
                    }

                    @Override
                    public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
                        if (!nonFiniteDetected) {
                            restoreBestModelCheckpointAtEndSafely();
                            persistCheckpointsSafely();
                        }
                        persistReportSafely(summary);
                    }
                });

        if (builder.checkpointDir != null) {
            runtimeBuilder.checkpointDir(builder.checkpointDir);
        }
        if (!builder.listeners.isEmpty()) {
            runtimeBuilder.listeners(builder.listeners);
        }

        this.runtime = runtimeBuilder.build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public void fit(Iterable<Batch> trainLoader) {
        try (Acceleration.Scope ignored = Acceleration.prefer(preferredDevice)) {
            prepareFitRun();
            accelerationStatusAtStart = Acceleration.status(preferredDevice);
            ensureCheckpointStateLoaded();
            runtime.fit(trainLoader);
            persistReportSafely(runtime.summary());
            latestSummary = enrichSummary(runtime.summary());
            throwIfNonFiniteDetected();
        }
    }

    public void fit(Iterable<Batch> trainLoader, Iterable<Batch> validationLoader) {
        try (Acceleration.Scope ignored = Acceleration.prefer(preferredDevice)) {
            prepareFitRun();
            accelerationStatusAtStart = Acceleration.status(preferredDevice);
            ensureCheckpointStateLoaded();
            runtime.fit(trainLoader, validationLoader);
            persistReportSafely(runtime.summary());
            latestSummary = enrichSummary(runtime.summary());
            throwIfNonFiniteDetected();
        }
    }

    public TrainerSession session() {
        return runtime;
    }

    public TrainingSummary summary() {
        TrainingSummary summary = latestSummary;
        if (summary != null) {
            return summary;
        }
        return enrichSummary(runtime.summary());
    }

    public boolean isStopped() {
        return runtime.isStopped();
    }

    public void stop() {
        runtime.stop();
    }

    private void prepareFitRun() {
        latestSummary = null;
        resetNonFiniteState();
    }

    private void resetNonFiniteState() {
        nonFiniteDetected = false;
        nonFinitePhase = null;
        nonFiniteKind = null;
        nonFiniteValue = Double.NaN;
        nonFiniteMessage = null;
        nonFiniteOptimizerStepSkipped = false;
    }

    private void throwIfNonFiniteDetected() {
        if (nonFiniteDetected) {
            throw new IllegalArgumentException(nonFiniteMessage);
        }
    }

    @Override
    public void close() {
        runtime.close();
    }

    private void ensureCheckpointStateLoaded() {
        ensureModelCheckpointLoaded();
        ensureOptimizerCheckpointLoaded();
        ensureSchedulerCheckpointLoaded();
        ensureHistoryLoaded();
    }

    private void ensureModelCheckpointLoaded() {
        if (!resumeFromCheckpoint || modelCheckpointFile == null) {
            return;
        }
        if (!modelCheckpointLoadAttempted.compareAndSet(false, true)) {
            return;
        }
        if (!Files.isRegularFile(modelCheckpointFile)) {
            return;
        }
        try {
            model.loadSafetensors(modelCheckpointFile);
            modelCheckpointLoaded = true;
            modelCheckpointLoadError = null;
        } catch (Exception error) {
            modelCheckpointLoadError = error.getMessage();
            if (failOnCheckpointLoadError) {
                throw new IllegalStateException("Failed to load model checkpoint from " + modelCheckpointFile, error);
            }
        }
    }

    private void ensureOptimizerCheckpointLoaded() {
        if (!resumeFromCheckpoint || optimizerCheckpointFile == null || !optimizer.supportsStateDict()) {
            return;
        }
        if (!optimizerCheckpointLoadAttempted.compareAndSet(false, true)) {
            return;
        }
        if (!Files.isRegularFile(optimizerCheckpointFile)) {
            return;
        }
        try {
            Map<String, Object> state = readOptimizerState(optimizerCheckpointFile);
            optimizer.loadStateDict(state);
            optimizerStepCount = Math.max(0, readInt(state.get("trainerOptimizerStepCount"), optimizerStepCount));
            optimizerCheckpointLoaded = true;
            optimizerCheckpointLoadError = null;
        } catch (Exception error) {
            optimizerCheckpointLoadError = error.getMessage();
            if (failOnCheckpointLoadError) {
                throw new IllegalStateException(
                        "Failed to load optimizer checkpoint from " + optimizerCheckpointFile, error);
            }
        }
    }

    private void ensureSchedulerCheckpointLoaded() {
        if (!resumeFromCheckpoint
                || schedulerCheckpointFile == null
                || learningRateScheduler == null
                || !learningRateScheduler.supportsStateDict()) {
            return;
        }
        if (!schedulerCheckpointLoadAttempted.compareAndSet(false, true)) {
            return;
        }
        if (!Files.isRegularFile(schedulerCheckpointFile)) {
            return;
        }
        try {
            Map<String, Object> state = readSchedulerState(schedulerCheckpointFile);
            requireSchedulerStepUnitMatch(state);
            learningRateScheduler.loadStateDict(state);
            schedulerStepCount = Math.max(0, readInt(state.get("trainerSchedulerStepCount"), schedulerStepCount));
            schedulerCheckpointLoaded = true;
            schedulerCheckpointLoadError = null;
        } catch (Exception error) {
            schedulerCheckpointLoadError = error.getMessage();
            if (failOnCheckpointLoadError) {
                throw new IllegalStateException(
                        "Failed to load scheduler checkpoint from " + schedulerCheckpointFile, error);
            }
        }
    }

    private void ensureHistoryLoaded() {
        if (!resumeFromCheckpoint || historyFile == null) {
            return;
        }
        if (!historyLoadAttempted.compareAndSet(false, true)) {
            return;
        }
        if (!Files.isRegularFile(historyFile)) {
            return;
        }
        try {
            List<Map<String, Object>> rows = readHistoryRows(historyFile);
            epochHistory.clear();
            epochHistory.addAll(rows);
            historyLoaded = true;
            historyLoadError = null;
        } catch (Exception error) {
            historyLoadError = error.getMessage();
        }
    }

    private void persistCheckpointsSafely() {
        if (nonFiniteDetected) {
            return;
        }
        persistModelCheckpointSafely();
        persistOptimizerCheckpointSafely();
        persistSchedulerCheckpointSafely();
        persistHistorySafely();
    }

    private void persistModelCheckpointSafely() {
        if (modelCheckpointFile == null) {
            return;
        }
        persistModelSafely(modelCheckpointFile, true);
    }

    private void persistBestModelCheckpointIfImproved(int epoch, double validationLoss) {
        if (!saveBestModelCheckpoint || bestModelCheckpointFile == null || !Double.isFinite(validationLoss)) {
            return;
        }
        double monitorValue = bestModelMonitorValue(validationLoss);
        if (!Double.isFinite(monitorValue)) {
            return;
        }
        boolean improved = Double.isNaN(bestModelCheckpointMonitorValue)
                || bestModelMonitorMode.isImproved(monitorValue, bestModelCheckpointMonitorValue, earlyStoppingMinDelta);
        if (!improved) {
            return;
        }
        if (persistModelSafely(bestModelCheckpointFile, false)) {
            bestModelCheckpointEpoch = epoch;
            bestModelCheckpointValidationLoss = validationLoss;
            bestModelCheckpointMonitorValue = monitorValue;
            bestModelCheckpointSaved = true;
            bestModelCheckpointSaveError = null;
        }
    }

    private double bestModelMonitorValue(double validationLoss) {
        return validationMonitorValue(
                bestModelMonitorMetric,
                validationLoss,
                "Best model monitor metric");
    }

    private void updateCustomEarlyStoppingMonitor(TrainerSession session, int epoch, double validationLoss) {
        if (earlyStoppingMonitorMetric == null || earlyStoppingPatience <= 0) {
            return;
        }
        double monitorValue = validationMonitorValue(
                earlyStoppingMonitorMetric,
                validationLoss,
                "Early stopping monitor metric");
        earlyStoppingMonitorLatestValue = monitorValue;
        if (!Double.isFinite(monitorValue)) {
            return;
        }
        boolean improved = Double.isNaN(earlyStoppingMonitorBestValue)
                || earlyStoppingMonitorMode.isImproved(
                        monitorValue,
                        earlyStoppingMonitorBestValue,
                        earlyStoppingMinDelta);
        if (improved) {
            earlyStoppingMonitorBestValue = monitorValue;
            earlyStoppingMonitorBestEpoch = epoch;
            earlyStoppingMonitorEpochsWithoutImprovement = 0;
            return;
        }
        earlyStoppingMonitorEpochsWithoutImprovement++;
        if (earlyStoppingMonitorEpochsWithoutImprovement >= earlyStoppingPatience) {
            earlyStoppingMonitorTriggered = true;
            earlyStoppingMonitorStopEpoch = epoch;
            session.stop();
        }
    }

    private double validationMonitorValue(
            String metricName,
            double validationLoss,
            String label) {
        if (metricName == null) {
            return validationLoss;
        }
        Double metricValue = latestValidationMetrics.get(metricName);
        if (metricValue == null) {
            throw new IllegalStateException(label + " '" + metricName
                    + "' is not available. Add it with .metric(...) or trainingOptions().*Metric().");
        }
        return metricValue;
    }

    private void restoreBestModelCheckpointAtEndSafely() {
        if (!restoreBestModelAtEnd || bestModelCheckpointFile == null) {
            return;
        }
        if (!Files.isRegularFile(bestModelCheckpointFile)) {
            return;
        }
        try {
            model.loadSafetensors(bestModelCheckpointFile);
            bestModelCheckpointRestored = true;
            bestModelCheckpointLoadError = null;
        } catch (Exception error) {
            bestModelCheckpointLoadError = error.getMessage();
            if (failOnCheckpointLoadError) {
                throw new IllegalStateException(
                        "Failed to restore best model checkpoint from " + bestModelCheckpointFile, error);
            }
        }
    }

    private boolean persistModelSafely(Path checkpointFile, boolean latestModelCheckpoint) {
        try {
            Path parent = checkpointFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempFile = checkpointFile.resolveSibling(checkpointFile.getFileName() + ".tmp");
            model.saveSafetensors(tempFile);
            try {
                Files.move(tempFile, checkpointFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveUnsupported) {
                Files.move(tempFile, checkpointFile, StandardCopyOption.REPLACE_EXISTING);
            }
            if (latestModelCheckpoint) {
                modelCheckpointSaved = true;
                modelCheckpointSaveError = null;
            }
            return true;
        } catch (Exception error) {
            if (!latestModelCheckpoint) {
                bestModelCheckpointSaveError = error.getMessage();
                return false;
            }
            modelCheckpointSaveError = error.getMessage();
            return false;
        }
    }

    private void persistOptimizerCheckpointSafely() {
        if (optimizerCheckpointFile == null || !optimizer.supportsStateDict()) {
            return;
        }
        try {
            Path parent = optimizerCheckpointFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempFile = optimizerCheckpointFile.resolveSibling(optimizerCheckpointFile.getFileName() + ".tmp");
            Map<String, Object> optimizerState = optimizer.stateDict();
            Map<String, Object> state = new HashMap<>(optimizerState == null ? Map.of() : optimizerState);
            state.put("trainerOptimizerStepCount", optimizerStepCount);
            writeOptimizerState(tempFile, state);
            try {
                Files.move(tempFile, optimizerCheckpointFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveUnsupported) {
                Files.move(tempFile, optimizerCheckpointFile, StandardCopyOption.REPLACE_EXISTING);
            }
            optimizerCheckpointSaved = true;
            optimizerCheckpointSaveError = null;
        } catch (Exception error) {
            optimizerCheckpointSaveError = error.getMessage();
        }
    }

    private void persistSchedulerCheckpointSafely() {
        if (schedulerCheckpointFile == null
                || learningRateScheduler == null
                || !learningRateScheduler.supportsStateDict()) {
            return;
        }
        try {
            Path parent = schedulerCheckpointFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempFile = schedulerCheckpointFile.resolveSibling(schedulerCheckpointFile.getFileName() + ".tmp");
            Map<String, Object> schedulerState = learningRateScheduler.stateDict();
            Map<String, Object> state = new HashMap<>(schedulerState == null ? Map.of() : schedulerState);
            state.put("trainerSchedulerStepUnit", schedulerStepUnit.name());
            state.put("trainerSchedulerStepCount", schedulerStepCount);
            writeSchedulerState(tempFile, state);
            try {
                Files.move(tempFile, schedulerCheckpointFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveUnsupported) {
                Files.move(tempFile, schedulerCheckpointFile, StandardCopyOption.REPLACE_EXISTING);
            }
            schedulerCheckpointSaved = true;
            schedulerCheckpointSaveError = null;
        } catch (Exception error) {
            schedulerCheckpointSaveError = error.getMessage();
        }
    }

    private void persistHistorySafely() {
        if (historyFile == null) {
            return;
        }
        try {
            Path parent = historyFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempFile = historyFile.resolveSibling(historyFile.getFileName() + ".tmp");
            Files.writeString(tempFile, historyCsv(), StandardCharsets.UTF_8);
            try {
                Files.move(tempFile, historyFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveUnsupported) {
                Files.move(tempFile, historyFile, StandardCopyOption.REPLACE_EXISTING);
            }
            historySaved = true;
            historySaveError = null;
        } catch (Exception error) {
            historySaveError = error.getMessage();
        }
    }

    private void persistReportSafely(TrainingSummary baseSummary) {
        if (reportFile == null || baseSummary == null) {
            return;
        }
        try {
            Path parent = reportFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            TrainingSummary summary = enrichSummary(baseSummary);
            Map<String, Object> report = trainingReportPayload(summary);
            Path tempFile = reportFile.resolveSibling(reportFile.getFileName() + ".tmp");
            Files.writeString(tempFile, toJson(report) + "\n", StandardCharsets.UTF_8);
            try {
                Files.move(tempFile, reportFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicMoveUnsupported) {
                Files.move(tempFile, reportFile, StandardCopyOption.REPLACE_EXISTING);
            }
            reportSaved = true;
            reportSaveError = null;
        } catch (Exception error) {
            reportSaveError = error.getMessage();
        }
    }

    private Map<String, Object> trainingReportPayload(TrainingSummary summary) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema", "gollek.canonical-trainer.report.v1");
        payload.put("generatedAt", Instant.now().toString());
        payload.put("epochCount", summary.epochCount());
        payload.put("bestValidationLoss", summary.bestValidationLoss());
        payload.put("bestValidationEpoch", summary.bestValidationEpoch());
        payload.put("latestTrainLoss", summary.latestTrainLoss());
        payload.put("latestValidationLoss", summary.latestValidationLoss());
        payload.put("durationMs", summary.durationMs());
        payload.put("metadata", summary.metadata());
        return payload;
    }

    private static String toJson(Object value) {
        StringBuilder json = new StringBuilder();
        appendJson(json, value);
        return json.toString();
    }

    private static void appendJson(StringBuilder json, Object value) {
        if (value == null) {
            json.append("null");
        } else if (value instanceof String text) {
            appendJsonString(json, text);
        } else if (value instanceof Number number) {
            appendJsonNumber(json, number);
        } else if (value instanceof Boolean bool) {
            json.append(bool.booleanValue());
        } else if (value instanceof Map<?, ?> map) {
            appendJsonMap(json, map);
        } else if (value instanceof Iterable<?> iterable) {
            appendJsonIterable(json, iterable);
        } else if (value.getClass().isArray()) {
            appendJsonArray(json, value);
        } else {
            appendJsonString(json, String.valueOf(value));
        }
    }

    private static void appendJsonMap(StringBuilder json, Map<?, ?> map) {
        json.append('{');
        List<Map.Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Comparator.comparing(entry -> String.valueOf(entry.getKey())));
        for (int i = 0; i < entries.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            Map.Entry<?, ?> entry = entries.get(i);
            appendJsonString(json, String.valueOf(entry.getKey()));
            json.append(':');
            appendJson(json, entry.getValue());
        }
        json.append('}');
    }

    private static void appendJsonIterable(StringBuilder json, Iterable<?> values) {
        json.append('[');
        int index = 0;
        for (Object value : values) {
            if (index++ > 0) {
                json.append(',');
            }
            appendJson(json, value);
        }
        json.append(']');
    }

    private static void appendJsonArray(StringBuilder json, Object array) {
        json.append('[');
        int length = java.lang.reflect.Array.getLength(array);
        for (int i = 0; i < length; i++) {
            if (i > 0) {
                json.append(',');
            }
            appendJson(json, java.lang.reflect.Array.get(array, i));
        }
        json.append(']');
    }

    private static void appendJsonNumber(StringBuilder json, Number number) {
        if (number instanceof Double doubleValue && !Double.isFinite(doubleValue.doubleValue())) {
            json.append("null");
            return;
        }
        if (number instanceof Float floatValue && !Float.isFinite(floatValue.floatValue())) {
            json.append("null");
            return;
        }
        json.append(number);
    }

    private static void appendJsonString(StringBuilder json, String value) {
        json.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"' -> json.append("\\\"");
                case '\\' -> json.append("\\\\");
                case '\b' -> json.append("\\b");
                case '\f' -> json.append("\\f");
                case '\n' -> json.append("\\n");
                case '\r' -> json.append("\\r");
                case '\t' -> json.append("\\t");
                default -> {
                    if (ch < 0x20) {
                        json.append(String.format("\\u%04x", (int) ch));
                    } else {
                        json.append(ch);
                    }
                }
            }
        }
        json.append('"');
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readOptimizerState(Path checkpointFile)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream input = new ObjectInputStream(Files.newInputStream(checkpointFile))) {
            Object payload = input.readObject();
            if (!(payload instanceof Map<?, ?> rawMap)) {
                throw new IOException("Invalid optimizer checkpoint payload: expected Map");
            }
            Map<String, Object> state = new HashMap<>();
            for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
                if (!(entry.getKey() instanceof String key)) {
                    throw new IOException("Invalid optimizer checkpoint payload: non-string key");
                }
                state.put(key, entry.getValue());
            }
            return state;
        }
    }

    private static Map<String, Object> readSchedulerState(Path checkpointFile)
            throws IOException, ClassNotFoundException {
        return readOptimizerState(checkpointFile);
    }

    private static void writeOptimizerState(Path checkpointFile, Map<String, Object> state) throws IOException {
        try (ObjectOutputStream output = new ObjectOutputStream(Files.newOutputStream(checkpointFile))) {
            output.writeObject(state);
        }
    }

    private static void writeSchedulerState(Path checkpointFile, Map<String, Object> state) throws IOException {
        writeOptimizerState(checkpointFile, state);
    }

    private void requireSchedulerStepUnitMatch(Map<String, Object> state) {
        Object rawStepUnit = state.get("trainerSchedulerStepUnit");
        if (rawStepUnit == null) {
            return;
        }
        String loadedStepUnit = String.valueOf(rawStepUnit);
        if (!schedulerStepUnit.name().equals(loadedStepUnit)) {
            throw new IllegalArgumentException(
                    "Invalid scheduler checkpoint payload: step unit mismatch (expected "
                            + schedulerStepUnit.name() + ", got " + loadedStepUnit + ")");
        }
    }

    private static int readInt(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private TrainingSummary enrichSummary(TrainingSummary base) {
        Map<String, Object> metadata = new HashMap<>(base.metadata());
        metadata.put("modelCheckpointEnabled", modelCheckpointFile != null);
        metadata.put("modelCheckpointResumeRequested", resumeFromCheckpoint);
        metadata.put("modelCheckpointLoaded", modelCheckpointLoaded);
        metadata.put("modelCheckpointSaved", modelCheckpointSaved);
        metadata.put("modelCheckpointLoadFailed", modelCheckpointLoadError != null);
        metadata.put("modelCheckpointSaveFailed", modelCheckpointSaveError != null);
        metadata.put("bestModelCheckpointEnabled", saveBestModelCheckpoint && bestModelCheckpointFile != null);
        metadata.put("bestModelCheckpointRestoreRequested", restoreBestModelAtEnd);
        metadata.put("bestModelCheckpointSaved", bestModelCheckpointSaved);
        metadata.put("bestModelCheckpointRestored", bestModelCheckpointRestored);
        metadata.put("bestModelCheckpointPresent",
                bestModelCheckpointFile != null && Files.isRegularFile(bestModelCheckpointFile));
        metadata.put("bestModelCheckpointMonitor", bestModelCheckpointMonitorLabel());
        metadata.put("bestModelCheckpointMonitorMode", bestModelMonitorMode.name());
        metadata.put("bestModelCheckpointEpoch", bestModelCheckpointEpoch);
        metadata.put("bestModelCheckpointValidationLoss", bestModelCheckpointValidationLoss);
        metadata.put("bestModelCheckpointMonitorValue", bestModelCheckpointMonitorValue);
        metadata.put("bestModelCheckpointSaveFailed", bestModelCheckpointSaveError != null);
        metadata.put("bestModelCheckpointLoadFailed", bestModelCheckpointLoadError != null);
        metadata.put("optimizerCheckpointEnabled", optimizerCheckpointFile != null);
        metadata.put("optimizerCheckpointSupported", optimizer.supportsStateDict());
        metadata.put("optimizerCheckpointResumeRequested", resumeFromCheckpoint);
        metadata.put("optimizerCheckpointLoaded", optimizerCheckpointLoaded);
        metadata.put("optimizerCheckpointSaved", optimizerCheckpointSaved);
        metadata.put("optimizerCheckpointLoadFailed", optimizerCheckpointLoadError != null);
        metadata.put("optimizerCheckpointSaveFailed", optimizerCheckpointSaveError != null);
        metadata.put("learningRateSchedulerEnabled", learningRateScheduler != null);
        metadata.put("learningRateSchedulerStepUnit", schedulerStepUnit.name());
        metadata.put("learningRateSchedulerStepCount", schedulerStepCount);
        metadata.put("learningRateSchedulerType",
                learningRateScheduler == null ? "none" : learningRateScheduler.getClass().getSimpleName());
        metadata.put("learningRateSchedulerMonitor", schedulerMonitorLabel());
        metadata.put("learningRateSchedulerMonitorMetricDriven", schedulerMonitorMetric != null);
        Map<String, Object> schedulerState = schedulerStateSnapshot();
        metadata.put("learningRateSchedulerState", schedulerState);
        flattenMetricDetails(metadata, "learningRateSchedulerState.", schedulerState);
        metadata.put("learningRate", optimizer.learningRate());
        metadata.put("trainingHistoryEnabled", historyFile != null);
        metadata.put("trainingHistoryLoaded", historyLoaded);
        metadata.put("trainingHistorySaved", historySaved);
        metadata.put("trainingHistoryLoadFailed", historyLoadError != null);
        metadata.put("trainingHistorySaveFailed", historySaveError != null);
        metadata.put("trainingReportEnabled", reportFile != null);
        metadata.put("trainingReportSaved", reportSaved);
        metadata.put("trainingReportSaveFailed", reportSaveError != null);
        metadata.put("earlyStoppingMonitor", earlyStoppingMonitorLabel());
        metadata.put("earlyStoppingMonitorMode", earlyStoppingMonitorMode.name());
        metadata.put("earlyStoppingMonitorMetricDriven", earlyStoppingMonitorMetric != null);
        metadata.put("earlyStoppingMonitorBestEpoch", earlyStoppingMonitorBestEpoch(base));
        metadata.put("earlyStoppingMonitorBestValue", earlyStoppingMonitorBestValue(base));
        metadata.put("earlyStoppingMonitorLatestValue", earlyStoppingMonitorLatestValue(base));
        metadata.put("earlyStoppingMonitorEpochsWithoutImprovement",
                earlyStoppingMonitorEpochsWithoutImprovement(base));
        if (earlyStoppingMonitorTriggered) {
            metadata.put("earlyStoppingTriggered", true);
            metadata.put("earlyStoppingEpoch", earlyStoppingMonitorStopEpoch);
            metadata.put("stopReason", "early-stopping");
        }
        metadata.put("nonFiniteGuardEnabled", true);
        metadata.put("nonFiniteDetected", nonFiniteDetected);
        if (nonFiniteDetected) {
            metadata.put("nonFinitePhase", nonFinitePhase);
            metadata.put("nonFiniteKind", nonFiniteKind);
            metadata.put("nonFiniteValue", nonFiniteValue);
            metadata.put("nonFiniteMessage", nonFiniteMessage);
            metadata.put("nonFiniteOptimizerStepSkipped", nonFiniteOptimizerStepSkipped);
            metadata.put("stopReason", "non-finite-" + nonFinitePhase + "-" + nonFiniteKind);
        }
        metadata.put("metricsEnabled", !trainMetrics.isEmpty());
        metadata.put("latestTrainMetrics", latestTrainMetrics);
        metadata.put("latestValidationMetrics", latestValidationMetrics);
        metadata.put("latestTrainMetricDetails", latestTrainMetricDetails);
        metadata.put("latestValidationMetricDetails", latestValidationMetricDetails);
        metadata.put("epochHistory", epochHistorySnapshot());
        metadata.put("epochHistorySize", epochHistory.size());
        flattenMetrics(metadata, "trainMetric.", latestTrainMetrics);
        flattenMetrics(metadata, "validationMetric.", latestValidationMetrics);
        flattenMetricDetails(metadata, "trainMetricDetails.", latestTrainMetricDetails);
        flattenMetricDetails(metadata, "validationMetricDetails.", latestValidationMetricDetails);
        metadata.put("gradientAccumulationSteps", gradientAccumulationSteps);
        metadata.put("pendingGradientAccumulationBatches", pendingGradientAccumulationBatches);
        metadata.put("optimizerStepCount", optimizerStepCount);
        metadata.put("latestGradientL2NormBeforeClip", latestGradientL2NormBeforeClip);
        metadata.put("latestGradientL2Norm", latestGradientL2Norm);
        metadata.put("latestGradientMaxAbsBeforeClip", latestGradientMaxAbsBeforeClip);
        metadata.put("latestGradientMaxAbs", latestGradientMaxAbs);
        metadata.put("latestGradientParameterCount", latestGradientParameterCount);
        metadata.put("latestGradientValueCount", latestGradientValueCount);
        metadata.put("latestGradientClipped", latestGradientClipped);
        metadata.put("latestParameterL2Norm", latestParameterL2Norm);
        metadata.put("latestParameterMaxAbs", latestParameterMaxAbs);
        metadata.put("latestParameterCount", latestParameterCount);
        metadata.put("latestParameterValueCount", latestParameterValueCount);
        metadata.put("trainBatchCount", trainBatchCount);
        metadata.put("validationBatchCount", validationBatchCount);
        metadata.put("trainSampleCount", trainSampleCount);
        metadata.put("validationSampleCount", validationSampleCount);
        metadata.put("trainInputElementCount", trainInputElementCount);
        metadata.put("validationInputElementCount", validationInputElementCount);
        metadata.put("trainLabelElementCount", trainLabelElementCount);
        metadata.put("validationLabelElementCount", validationLabelElementCount);
        metadata.put("trainComputeMillis", nanosToMillis(trainComputeNanos));
        metadata.put("validationComputeMillis", nanosToMillis(validationComputeNanos));
        metadata.put("trainSamplesPerSecond", samplesPerSecond(trainSampleCount, trainComputeNanos));
        metadata.put("validationSamplesPerSecond", samplesPerSecond(validationSampleCount, validationComputeNanos));
        Acceleration.BackendStatus accelerationStatus = Acceleration.status(preferredDevice);
        metadata.put("requestedDevice", preferredDevice);
        metadata.put("executionBackend", accelerationStatus.id());
        metadata.put("executionDeviceName", accelerationStatus.deviceName());
        metadata.put("executionAccelerated", accelerationStatus.accelerated());
        metadata.put("requestedDeviceAvailable", accelerationStatus.available());
        metadata.put("acceleratedMatmulCalls", accelerationStatus.acceleratedMatmulCalls());
        if (accelerationStatusAtStart != null) {
            metadata.put("acceleratedMatmulCallsAtStart", accelerationStatusAtStart.acceleratedMatmulCalls());
        }
        metadata.put("schedulerCheckpointEnabled", schedulerCheckpointFile != null);
        metadata.put("schedulerCheckpointSupported",
                learningRateScheduler != null && learningRateScheduler.supportsStateDict());
        metadata.put("schedulerCheckpointResumeRequested", resumeFromCheckpoint);
        metadata.put("schedulerCheckpointLoaded", schedulerCheckpointLoaded);
        metadata.put("schedulerCheckpointSaved", schedulerCheckpointSaved);
        metadata.put("schedulerCheckpointLoadFailed", schedulerCheckpointLoadError != null);
        metadata.put("schedulerCheckpointSaveFailed", schedulerCheckpointSaveError != null);
        if (modelCheckpointFile != null) {
            metadata.put("modelCheckpointFile", modelCheckpointFile.toString());
        }
        if (bestModelCheckpointFile != null) {
            metadata.put("bestModelCheckpointFile", bestModelCheckpointFile.toString());
        }
        if (optimizerCheckpointFile != null) {
            metadata.put("optimizerCheckpointFile", optimizerCheckpointFile.toString());
        }
        if (schedulerCheckpointFile != null) {
            metadata.put("schedulerCheckpointFile", schedulerCheckpointFile.toString());
        }
        if (historyFile != null) {
            metadata.put("trainingHistoryFile", historyFile.toString());
        }
        if (reportFile != null) {
            metadata.put("trainingReportFile", reportFile.toString());
        }
        if (modelCheckpointLoadError != null) {
            metadata.put("modelCheckpointLoadError", modelCheckpointLoadError);
        }
        if (modelCheckpointSaveError != null) {
            metadata.put("modelCheckpointSaveError", modelCheckpointSaveError);
        }
        if (bestModelCheckpointSaveError != null) {
            metadata.put("bestModelCheckpointSaveError", bestModelCheckpointSaveError);
        }
        if (bestModelCheckpointLoadError != null) {
            metadata.put("bestModelCheckpointLoadError", bestModelCheckpointLoadError);
        }
        if (optimizerCheckpointLoadError != null) {
            metadata.put("optimizerCheckpointLoadError", optimizerCheckpointLoadError);
        }
        if (optimizerCheckpointSaveError != null) {
            metadata.put("optimizerCheckpointSaveError", optimizerCheckpointSaveError);
        }
        if (schedulerCheckpointLoadError != null) {
            metadata.put("schedulerCheckpointLoadError", schedulerCheckpointLoadError);
        }
        if (schedulerCheckpointSaveError != null) {
            metadata.put("schedulerCheckpointSaveError", schedulerCheckpointSaveError);
        }
        if (historyLoadError != null) {
            metadata.put("trainingHistoryLoadError", historyLoadError);
        }
        if (historySaveError != null) {
            metadata.put("trainingHistorySaveError", historySaveError);
        }
        if (reportSaveError != null) {
            metadata.put("trainingReportSaveError", reportSaveError);
        }
        return new TrainingSummary(
                base.epochCount(),
                base.bestValidationLoss(),
                base.bestValidationEpoch(),
                base.latestTrainLoss(),
                base.latestValidationLoss(),
                base.durationMs(),
                Map.copyOf(metadata));
    }

    private String bestModelCheckpointMonitorLabel() {
        return bestModelMonitorMetric == null ? "validation_loss" : "validationMetric." + bestModelMonitorMetric;
    }

    private String earlyStoppingMonitorLabel() {
        return earlyStoppingMonitorMetric == null ? "validation_loss" : "validationMetric." + earlyStoppingMonitorMetric;
    }

    private String schedulerMonitorLabel() {
        return schedulerMonitorMetric == null ? "validation_loss" : "validationMetric." + schedulerMonitorMetric;
    }

    private double schedulerMonitorValue(double validationLoss) {
        if (schedulerMonitorMetric == null) {
            return validationLoss;
        }
        Double metricValue = latestValidationMetrics.get(schedulerMonitorMetric);
        return metricValue == null ? Double.NaN : metricValue;
    }

    private int earlyStoppingMonitorBestEpoch(TrainingSummary base) {
        return earlyStoppingMonitorMetric == null ? base.bestValidationEpoch() : earlyStoppingMonitorBestEpoch;
    }

    private double earlyStoppingMonitorBestValue(TrainingSummary base) {
        return earlyStoppingMonitorMetric == null ? base.bestValidationLoss() : earlyStoppingMonitorBestValue;
    }

    private double earlyStoppingMonitorLatestValue(TrainingSummary base) {
        return earlyStoppingMonitorMetric == null
                ? (base.latestValidationLoss() == null ? Double.NaN : base.latestValidationLoss())
                : earlyStoppingMonitorLatestValue;
    }

    private int earlyStoppingMonitorEpochsWithoutImprovement(TrainingSummary base) {
        if (earlyStoppingMonitorMetric != null) {
            return earlyStoppingMonitorEpochsWithoutImprovement;
        }
        Object value = base.metadata().get("epochsWithoutImprovement");
        return value instanceof Number number ? number.intValue() : 0;
    }

    private static Path resolveModelCheckpointFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(MODEL_CHECKPOINT_FILE_NAME);
    }

    private static Path resolveBestModelCheckpointFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(BEST_MODEL_CHECKPOINT_FILE_NAME);
    }

    private static Path resolveOptimizerCheckpointFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(OPTIMIZER_CHECKPOINT_FILE_NAME);
    }

    private static Path resolveSchedulerCheckpointFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(SCHEDULER_CHECKPOINT_FILE_NAME);
    }

    private static Path resolveHistoryFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(HISTORY_CHECKPOINT_FILE_NAME);
    }

    private static Path resolveReportFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(REPORT_CHECKPOINT_FILE_NAME);
    }

    private static String normalizeDevice(String deviceId) {
        if (deviceId == null || deviceId.isBlank()) {
            return "auto";
        }
        return deviceId.trim().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    private static String normalizeBestModelMonitorMetric(String metricName) {
        if (metricName == null || metricName.isBlank()) {
            return null;
        }
        String normalized = metricName.trim();
        if (normalized.startsWith("validationMetric.")) {
            normalized = normalized.substring("validationMetric.".length());
        }
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("best model monitor metric must not be blank");
        }
        return requireMetricName(normalized);
    }

    private static void requireBestModelMonitorMetricPresent(
            String metricName,
            List<Metric> validationMetrics) {
        if (metricName == null) {
            return;
        }
        for (Metric metric : validationMetrics) {
            if (metricName.equals(metric.name())) {
                return;
            }
        }
        throw new IllegalArgumentException("best model monitor metric '" + metricName
                + "' is not registered. Add the matching metric before build().");
    }

    private static List<Metric> instantiateMetrics(
            List<Supplier<? extends Metric>> metricFactories,
            String phase) {
        if (metricFactories.isEmpty()) {
            return List.of();
        }
        List<Metric> metrics = new ArrayList<>(metricFactories.size());
        for (Supplier<? extends Metric> factory : metricFactories) {
            Metric metric = Objects.requireNonNull(factory.get(), phase + " metric factory returned null");
            requireMetricName(metric.name());
            metrics.add(metric);
        }
        return List.copyOf(metrics);
    }

    private static String requireMetricName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("metric name must not be blank");
        }
        return name.trim();
    }

    private void resetMetrics() {
        for (Metric metric : trainMetrics) {
            metric.reset();
        }
        for (Metric metric : validationMetrics) {
            metric.reset();
        }
        latestTrainMetrics = Map.of();
        latestValidationMetrics = Map.of();
        latestTrainMetricDetails = Map.of();
        latestValidationMetricDetails = Map.of();
    }

    private void resetEpochThroughputCounters() {
        epochTrainBatchCount = 0L;
        epochValidationBatchCount = 0L;
        epochTrainSampleCount = 0L;
        epochValidationSampleCount = 0L;
        epochTrainInputElementCount = 0L;
        epochValidationInputElementCount = 0L;
        epochTrainLabelElementCount = 0L;
        epochValidationLabelElementCount = 0L;
        epochTrainComputeNanos = 0L;
        epochValidationComputeNanos = 0L;
    }

    private static Map<String, Double> snapshotMetrics(List<Metric> metrics) {
        if (metrics.isEmpty()) {
            return Map.of();
        }
        Map<String, Double> values = new LinkedHashMap<>();
        for (Metric metric : metrics) {
            values.put(requireMetricName(metric.name()), metric.value());
        }
        return Map.copyOf(values);
    }

    private static Map<String, Object> snapshotMetricDetails(List<Metric> metrics) {
        if (metrics.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (Metric metric : metrics) {
            if (metric instanceof DetailedMetric detailedMetric) {
                Map<String, Object> details = detailedMetric.details();
                if (details != null && !details.isEmpty()) {
                    values.put(requireMetricName(metric.name()), immutableMetricDetail(details));
                }
            }
        }
        return values.isEmpty() ? Map.of() : Collections.unmodifiableMap(values);
    }

    private static Object immutableMetricDetail(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                copy.put(String.valueOf(entry.getKey()), immutableMetricDetail(entry.getValue()));
            }
            return Collections.unmodifiableMap(copy);
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> copy = new ArrayList<>();
            for (Object item : iterable) {
                copy.add(immutableMetricDetail(item));
            }
            return Collections.unmodifiableList(copy);
        }
        if (value != null && value.getClass().isArray()) {
            int length = java.lang.reflect.Array.getLength(value);
            List<Object> copy = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                copy.add(immutableMetricDetail(java.lang.reflect.Array.get(value, i)));
            }
            return Collections.unmodifiableList(copy);
        }
        return value;
    }

    private static void flattenMetrics(
            Map<String, Object> metadata,
            String prefix,
            Map<String, Double> metrics) {
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            metadata.put(prefix + entry.getKey(), entry.getValue());
        }
    }

    private static void flattenMetricDetails(
            Map<String, Object> metadata,
            String prefix,
            Map<String, Object> metricDetails) {
        for (Map.Entry<String, Object> entry : metricDetails.entrySet()) {
            metadata.put(prefix + entry.getKey(), entry.getValue());
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> schedulerStateSnapshot() {
        if (learningRateScheduler == null || !learningRateScheduler.supportsStateDict()) {
            return Map.of();
        }
        Object state = immutableMetricDetail(learningRateScheduler.stateDict());
        if (state instanceof Map<?, ?> stateMap) {
            return (Map<String, Object>) stateMap;
        }
        return Map.of();
    }

    private void recordEpochHistory(int epoch, double trainLoss) {
        Map<String, Object> row = historyRow(epoch);
        row.put("epoch", epoch);
        row.put("trainLoss", trainLoss);
        row.put("learningRate", optimizer.learningRate());
        row.put("optimizerStepCount", optimizerStepCount);
        row.put("schedulerStepCount", schedulerStepCount);
        row.put("gradientL2NormBeforeClip", latestGradientL2NormBeforeClip);
        row.put("gradientL2Norm", latestGradientL2Norm);
        row.put("gradientMaxAbsBeforeClip", latestGradientMaxAbsBeforeClip);
        row.put("gradientMaxAbs", latestGradientMaxAbs);
        row.put("gradientParameterCount", latestGradientParameterCount);
        row.put("gradientValueCount", latestGradientValueCount);
        row.put("gradientClipped", latestGradientClipped);
        row.put("parameterL2Norm", latestParameterL2Norm);
        row.put("parameterMaxAbs", latestParameterMaxAbs);
        row.put("parameterCount", latestParameterCount);
        row.put("parameterValueCount", latestParameterValueCount);
        row.put("trainBatchCount", epochTrainBatchCount);
        row.put("trainSampleCount", epochTrainSampleCount);
        row.put("trainInputElementCount", epochTrainInputElementCount);
        row.put("trainLabelElementCount", epochTrainLabelElementCount);
        row.put("trainComputeMillis", nanosToMillis(epochTrainComputeNanos));
        row.put("trainSamplesPerSecond", samplesPerSecond(epochTrainSampleCount, epochTrainComputeNanos));
        row.put("trainMetrics", latestTrainMetrics);
        row.put("trainMetricDetails", latestTrainMetricDetails);
        flattenMetrics(row, "trainMetric.", latestTrainMetrics);
        flattenMetricDetails(row, "trainMetricDetails.", latestTrainMetricDetails);
        if (!epochHistory.contains(row)) {
            epochHistory.add(row);
        }
    }

    private void updateEpochHistoryValidation(int epoch, double validationLoss) {
        Map<String, Object> row = historyRow(epoch);
        row.put("epoch", epoch);
        row.put("validationLoss", validationLoss);
        row.put("learningRate", optimizer.learningRate());
        row.put("schedulerStepCount", schedulerStepCount);
        row.put("validationBatchCount", epochValidationBatchCount);
        row.put("validationSampleCount", epochValidationSampleCount);
        row.put("validationInputElementCount", epochValidationInputElementCount);
        row.put("validationLabelElementCount", epochValidationLabelElementCount);
        row.put("validationComputeMillis", nanosToMillis(epochValidationComputeNanos));
        row.put("validationSamplesPerSecond", samplesPerSecond(epochValidationSampleCount, epochValidationComputeNanos));
        row.put("validationMetrics", latestValidationMetrics);
        row.put("validationMetricDetails", latestValidationMetricDetails);
        flattenMetrics(row, "validationMetric.", latestValidationMetrics);
        flattenMetricDetails(row, "validationMetricDetails.", latestValidationMetricDetails);
        double monitorValue = bestModelMonitorValue(validationLoss);
        if (Double.isFinite(monitorValue)) {
            row.put("bestModelMonitor", bestModelCheckpointMonitorLabel());
            row.put("bestModelMonitorMode", bestModelMonitorMode.name());
            row.put("bestModelMonitorValue", monitorValue);
        }
        if (!epochHistory.contains(row)) {
            epochHistory.add(row);
        }
    }

    private Map<String, Object> historyRow(int epoch) {
        for (int i = epochHistory.size() - 1; i >= 0; i--) {
            Map<String, Object> row = epochHistory.get(i);
            Object rowEpoch = row.get("epoch");
            if (rowEpoch instanceof Number number && number.intValue() == epoch) {
                return row;
            }
        }
        return new LinkedHashMap<>();
    }

    private List<Map<String, Object>> epochHistorySnapshot() {
        if (epochHistory.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> snapshot = new ArrayList<>(epochHistory.size());
        for (Map<String, Object> row : epochHistory) {
            snapshot.add(Collections.unmodifiableMap(copyHistoryRow(row)));
        }
        return List.copyOf(snapshot);
    }

    private String historyCsv() {
        List<Map<String, Object>> rows = epochHistorySnapshot();
        List<String> columns = historyCsvColumns(rows);
        StringBuilder csv = new StringBuilder();
        appendCsvRow(csv, columns);
        for (Map<String, Object> row : rows) {
            List<String> values = new ArrayList<>(columns.size());
            for (String column : columns) {
                values.add(historyCsvValue(row.get(column)));
            }
            appendCsvRow(csv, values);
        }
        return csv.toString();
    }

    private static List<String> historyCsvColumns(List<Map<String, Object>> rows) {
        Set<String> columns = new LinkedHashSet<>(List.of(
                "epoch",
                "trainLoss",
                "validationLoss",
                "learningRate",
                "optimizerStepCount",
                "schedulerStepCount",
                "bestModelMonitor",
                "bestModelMonitorMode",
                "bestModelMonitorValue"));
        for (Map<String, Object> row : rows) {
            for (Map.Entry<String, Object> entry : row.entrySet()) {
                if (!(entry.getValue() instanceof Map<?, ?>)) {
                    columns.add(entry.getKey());
                }
            }
        }
        return List.copyOf(columns);
    }

    private static String historyCsvValue(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof Number number) {
            return number.toString();
        }
        if (value instanceof Boolean bool) {
            return bool.toString();
        }
        return String.valueOf(value);
    }

    private static void appendCsvRow(StringBuilder csv, List<String> values) {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                csv.append(',');
            }
            appendCsvCell(csv, values.get(i));
        }
        csv.append('\n');
    }

    private static void appendCsvCell(StringBuilder csv, String value) {
        boolean quote = value.indexOf(',') >= 0
                || value.indexOf('"') >= 0
                || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0;
        if (!quote) {
            csv.append(value);
            return;
        }
        csv.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (ch == '"') {
                csv.append('"');
            }
            csv.append(ch);
        }
        csv.append('"');
    }

    private static List<Map<String, Object>> readHistoryRows(Path csvFile) throws IOException {
        String csv = Files.readString(csvFile, StandardCharsets.UTF_8);
        List<List<String>> records = parseCsv(csv);
        if (records.isEmpty()) {
            return List.of();
        }
        List<String> columns = records.get(0);
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int rowIndex = 1; rowIndex < records.size(); rowIndex++) {
            List<String> record = records.get(rowIndex);
            if (record.isEmpty() || record.stream().allMatch(String::isBlank)) {
                continue;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            for (int columnIndex = 0; columnIndex < columns.size() && columnIndex < record.size(); columnIndex++) {
                String column = columns.get(columnIndex);
                String rawValue = record.get(columnIndex);
                if (column == null || column.isBlank() || rawValue == null || rawValue.isBlank()) {
                    continue;
                }
                row.put(column, parseHistoryScalar(rawValue));
            }
            restoreNestedHistoryMetrics(row, "trainMetric.", "trainMetrics");
            restoreNestedHistoryMetrics(row, "validationMetric.", "validationMetrics");
            if (!row.isEmpty()) {
                rows.add(row);
            }
        }
        return rows;
    }

    private static List<List<String>> parseCsv(String csv) throws IOException {
        List<List<String>> records = new ArrayList<>();
        List<String> record = new ArrayList<>();
        StringBuilder cell = new StringBuilder();
        boolean quoted = false;
        for (int index = 0; index < csv.length(); index++) {
            char ch = csv.charAt(index);
            if (quoted) {
                if (ch == '"') {
                    if (index + 1 < csv.length() && csv.charAt(index + 1) == '"') {
                        cell.append('"');
                        index++;
                    } else {
                        quoted = false;
                    }
                } else {
                    cell.append(ch);
                }
                continue;
            }
            if (ch == '"') {
                quoted = true;
            } else if (ch == ',') {
                record.add(cell.toString());
                cell.setLength(0);
            } else if (ch == '\n') {
                record.add(cell.toString());
                cell.setLength(0);
                records.add(record);
                record = new ArrayList<>();
            } else if (ch != '\r') {
                cell.append(ch);
            }
        }
        if (quoted) {
            throw new IOException("Invalid training history CSV: unterminated quoted cell");
        }
        if (cell.length() > 0 || !record.isEmpty()) {
            record.add(cell.toString());
            records.add(record);
        }
        return records;
    }

    private static Object parseHistoryScalar(String rawValue) {
        String value = rawValue.trim();
        if ("true".equalsIgnoreCase(value)) {
            return Boolean.TRUE;
        }
        if ("false".equalsIgnoreCase(value)) {
            return Boolean.FALSE;
        }
        try {
            if (!value.contains(".") && !value.contains("e") && !value.contains("E")) {
                long parsed = Long.parseLong(value);
                if (parsed >= Integer.MIN_VALUE && parsed <= Integer.MAX_VALUE) {
                    return (int) parsed;
                }
                return parsed;
            }
            return Double.parseDouble(value);
        } catch (NumberFormatException integerError) {
            try {
                return Double.parseDouble(value);
            } catch (NumberFormatException floatingPointError) {
                return rawValue;
            }
        }
    }

    private static void restoreNestedHistoryMetrics(
            Map<String, Object> row,
            String metricPrefix,
            String metricMapKey) {
        Map<String, Object> metrics = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            if (entry.getKey().startsWith(metricPrefix)) {
                metrics.put(entry.getKey().substring(metricPrefix.length()), entry.getValue());
            }
        }
        if (!metrics.isEmpty()) {
            row.put(metricMapKey, Collections.unmodifiableMap(metrics));
        }
    }

    private static Map<String, Object> copyHistoryRow(Map<String, Object> row) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Map<?, ?> rawMap) {
                Map<String, Object> nested = new LinkedHashMap<>();
                for (Map.Entry<?, ?> nestedEntry : rawMap.entrySet()) {
                    nested.put(String.valueOf(nestedEntry.getKey()), nestedEntry.getValue());
                }
                copy.put(entry.getKey(), Collections.unmodifiableMap(nested));
            } else {
                copy.put(entry.getKey(), value);
            }
        }
        return copy;
    }

    private static void updateMetrics(
            List<Metric> metrics,
            GradTensor predictions,
            GradTensor targets) {
        for (Metric metric : metrics) {
            metric.update(predictions, targets);
        }
    }

    private double runTrainingBatch(Object rawBatch) {
        Batch batch = toBatch(rawBatch, "train");
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.train();
        try {
            if (pendingGradientAccumulationBatches == 0) {
                model.zeroGrad();
            }

            GradTensor prediction = model.forward(batch.inputs());
            GradTensor lossTensor = Objects.requireNonNull(
                    lossFunction.compute(prediction, batch.labels()),
                    "loss function returned null");
            double loss = requireFiniteLoss(lossTensor.item(), "train");

            updateMetrics(trainMetrics, prediction, batch.labels());
            lossTensor.backward();
            pendingGradientAccumulationBatches++;
            if (pendingGradientAccumulationBatches >= gradientAccumulationSteps) {
                applyOptimizerStep();
            }
            recordThroughput(batch, true, System.nanoTime() - startedAt);
            counted = true;
            return loss;
        } finally {
            if (!counted) {
                recordThroughput(batch, true, System.nanoTime() - startedAt);
            }
        }
    }

    private void flushGradientAccumulation() {
        flushGradientAccumulation(null);
    }

    private void flushGradientAccumulation(TrainerSession session) {
        if (pendingGradientAccumulationBatches > 0) {
            try {
                applyOptimizerStep();
            } catch (RuntimeException error) {
                if (nonFiniteDetected && session != null) {
                    session.stop();
                    return;
                }
                throw error;
            }
        }
    }

    private void applyOptimizerStep() {
        if (pendingGradientAccumulationBatches <= 0) {
            return;
        }
        try {
            scaleAccumulatedGradients(1.0f / pendingGradientAccumulationBatches);
            TensorDiagnostics gradientBeforeClip = gradientDiagnostics();
            requireFiniteDiagnostics(gradientBeforeClip, "train", "gradient", true);
            if (gradientClip > 0.0) {
                GradientClipper.clipByNorm(optimizer.parameters(), (float) gradientClip);
            }
            TensorDiagnostics gradientAfterClip = gradientDiagnostics();
            requireFiniteDiagnostics(gradientAfterClip, "train", "clipped-gradient", true);
            optimizer.step();
            optimizerStepCount++;
            TensorDiagnostics parametersAfterStep = parameterDiagnostics();
            requireFiniteDiagnostics(parametersAfterStep, "train", "parameter", false);
            updateTensorDiagnostics(gradientBeforeClip, gradientAfterClip, parametersAfterStep);
            stepScheduler(SchedulerStepUnit.BATCH);
            optimizer.zeroGrad();
            pendingGradientAccumulationBatches = 0;
        } catch (RuntimeException error) {
            if (nonFiniteDetected) {
                discardPendingGradients();
            }
            throw error;
        }
    }

    private void updateTensorDiagnostics(
            TensorDiagnostics gradientBeforeClip,
            TensorDiagnostics gradientAfterClip,
            TensorDiagnostics parametersAfterStep) {
        latestGradientL2NormBeforeClip = gradientBeforeClip.l2Norm();
        latestGradientL2Norm = gradientAfterClip.l2Norm();
        latestGradientMaxAbsBeforeClip = gradientBeforeClip.maxAbs();
        latestGradientMaxAbs = gradientAfterClip.maxAbs();
        latestGradientParameterCount = gradientAfterClip.tensorCount();
        latestGradientValueCount = gradientAfterClip.valueCount();
        latestGradientClipped = gradientAfterClip.l2Norm() + 1e-9 < gradientBeforeClip.l2Norm();
        latestParameterL2Norm = parametersAfterStep.l2Norm();
        latestParameterMaxAbs = parametersAfterStep.maxAbs();
        latestParameterCount = parametersAfterStep.tensorCount();
        latestParameterValueCount = parametersAfterStep.valueCount();
    }

    private TensorDiagnostics gradientDiagnostics() {
        double sumSquares = 0.0;
        double maxAbs = 0.0;
        int tensorCount = 0;
        long valueCount = 0;
        for (var parameter : optimizer.parameters()) {
            GradTensor gradient = parameter.grad();
            if (gradient == null) {
                continue;
            }
            tensorCount++;
            float[] values = gradient.data();
            valueCount += values.length;
            for (float value : values) {
                double absolute = Math.abs(value);
                sumSquares += absolute * absolute;
                maxAbs = Math.max(maxAbs, absolute);
            }
        }
        return new TensorDiagnostics(tensorCount, valueCount, Math.sqrt(sumSquares), maxAbs);
    }

    private TensorDiagnostics parameterDiagnostics() {
        double sumSquares = 0.0;
        double maxAbs = 0.0;
        int tensorCount = 0;
        long valueCount = 0;
        for (var parameter : optimizer.parameters()) {
            tensorCount++;
            float[] values = parameter.data().data();
            valueCount += values.length;
            for (float value : values) {
                double absolute = Math.abs(value);
                sumSquares += absolute * absolute;
                maxAbs = Math.max(maxAbs, absolute);
            }
        }
        return new TensorDiagnostics(tensorCount, valueCount, Math.sqrt(sumSquares), maxAbs);
    }

    private void scaleAccumulatedGradients(float scale) {
        if (scale == 1.0f) {
            return;
        }
        for (var parameter : optimizer.parameters()) {
            GradTensor grad = parameter.grad();
            if (grad == null) {
                continue;
            }
            float[] values = grad.data();
            for (int i = 0; i < values.length; i++) {
                values[i] *= scale;
            }
        }
    }

    private void stepScheduler(SchedulerStepUnit unit) {
        stepScheduler(unit, Double.NaN);
    }

    private void stepScheduler(SchedulerStepUnit unit, double monitorValue) {
        if (learningRateScheduler == null || schedulerStepUnit != unit) {
            return;
        }
        if (unit == SchedulerStepUnit.VALIDATION) {
            learningRateScheduler.step(monitorValue);
        } else {
            learningRateScheduler.step();
        }
        schedulerStepCount++;
    }

    private double runValidationBatch(Object rawBatch) {
        Batch batch = toBatch(rawBatch, "validation");
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.eval();
        try {
            try (NoGrad ignored = NoGrad.enter()) {
                GradTensor prediction = model.forward(batch.inputs());
                GradTensor lossTensor = Objects.requireNonNull(
                        lossFunction.compute(prediction, batch.labels()),
                        "loss function returned null");
                double loss = requireFiniteLoss(lossTensor.item(), "validation");
                updateMetrics(validationMetrics, prediction, batch.labels());
                recordThroughput(batch, false, System.nanoTime() - startedAt);
                counted = true;
                return loss;
            }
        } finally {
            if (!counted) {
                recordThroughput(batch, false, System.nanoTime() - startedAt);
            }
        }
    }

    private void recordThroughput(Batch batch, boolean trainPhase, long elapsedNanos) {
        long safeElapsed = Math.max(0L, elapsedNanos);
        long samples = sampleCount(batch);
        long inputElements = batch.inputs().numel();
        long labelElements = batch.labels().numel();
        if (trainPhase) {
            trainBatchCount++;
            epochTrainBatchCount++;
            trainSampleCount += samples;
            epochTrainSampleCount += samples;
            trainInputElementCount += inputElements;
            epochTrainInputElementCount += inputElements;
            trainLabelElementCount += labelElements;
            epochTrainLabelElementCount += labelElements;
            trainComputeNanos += safeElapsed;
            epochTrainComputeNanos += safeElapsed;
        } else {
            validationBatchCount++;
            epochValidationBatchCount++;
            validationSampleCount += samples;
            epochValidationSampleCount += samples;
            validationInputElementCount += inputElements;
            epochValidationInputElementCount += inputElements;
            validationLabelElementCount += labelElements;
            epochValidationLabelElementCount += labelElements;
            validationComputeNanos += safeElapsed;
            epochValidationComputeNanos += safeElapsed;
        }
    }

    private static long sampleCount(Batch batch) {
        long[] shape = batch.inputs().shape();
        return shape.length == 0 ? 1L : Math.max(0L, shape[0]);
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static double samplesPerSecond(long samples, long nanos) {
        if (samples <= 0L || nanos <= 0L) {
            return 0.0;
        }
        return samples * 1_000_000_000.0 / nanos;
    }

    private static Batch toBatch(Object rawBatch, String phase) {
        if (rawBatch instanceof Batch batch) {
            return batch;
        }
        throw new IllegalArgumentException(
                "Expected DataLoader.Batch for " + phase + " step but got: "
                        + (rawBatch == null ? "null" : rawBatch.getClass().getName()));
    }

    private double requireFiniteLoss(double value, String phase) {
        if (!Double.isFinite(value)) {
            recordNonFinite(phase, "loss", value, "loss", "train".equals(phase));
            if ("train".equals(phase)) {
                discardPendingGradients();
            }
            throw new IllegalArgumentException(nonFiniteMessage);
        }
        return value;
    }

    private void requireFiniteDiagnostics(
            TensorDiagnostics diagnostics,
            String phase,
            String kind,
            boolean optimizerStepSkipped) {
        if (Double.isFinite(diagnostics.l2Norm()) && Double.isFinite(diagnostics.maxAbs())) {
            return;
        }
        double value = Double.isFinite(diagnostics.l2Norm()) ? diagnostics.maxAbs() : diagnostics.l2Norm();
        recordNonFinite(phase, kind, value, kind, optimizerStepSkipped);
        throw new IllegalArgumentException(nonFiniteMessage);
    }

    private void recordNonFinite(
            String phase,
            String kind,
            double value,
            String label,
            boolean optimizerStepSkipped) {
        String safePhase = phase == null || phase.isBlank() ? "unknown" : phase;
        String safeKind = kind == null || kind.isBlank() ? "value" : kind;
        String message = safePhase + " " + label + " must be finite, got " + value;
        if (!nonFiniteDetected) {
            nonFiniteDetected = true;
            nonFinitePhase = safePhase;
            nonFiniteKind = safeKind;
            nonFiniteValue = value;
            nonFiniteMessage = message;
            nonFiniteOptimizerStepSkipped = optimizerStepSkipped;
        }
    }

    private void discardPendingGradients() {
        optimizer.zeroGrad();
        pendingGradientAccumulationBatches = 0;
    }

    @FunctionalInterface
    public interface LossFunction {
        GradTensor compute(GradTensor predictions, GradTensor targets);
    }

    public interface Metric {
        String name();

        void reset();

        void update(GradTensor predictions, GradTensor targets);

        double value();
    }

    public interface DetailedMetric extends Metric {
        Map<String, Object> details();
    }

    public static final class Metrics {
        private Metrics() {
        }

        public static Supplier<Metric> classificationAccuracy() {
            return AccuracyMetric::new;
        }

        public static Supplier<Metric> accuracy() {
            return classificationAccuracy();
        }

        public static Supplier<Metric> classificationConfusionMatrix() {
            return ClassificationConfusionMatrixMetric::new;
        }

        public static Supplier<Metric> confusionMatrix() {
            return classificationConfusionMatrix();
        }

        public static Supplier<Metric> topKAccuracy(int k) {
            if (k < 1) {
                throw new IllegalArgumentException("top-k accuracy requires k >= 1, got " + k);
            }
            return () -> new TopKAccuracyMetric(k);
        }

        public static Supplier<Metric> binaryAccuracy() {
            return BinaryAccuracyMetric::new;
        }

        public static Supplier<Metric> binaryAccuracy(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new BinaryAccuracyMetric(threshold);
        }

        public static Supplier<Metric> binaryConfusionMatrix() {
            return BinaryConfusionMatrixMetric::new;
        }

        public static Supplier<Metric> binaryConfusionMatrix(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new BinaryConfusionMatrixMetric(threshold);
        }

        public static Supplier<Metric> binaryPrecision() {
            return BinaryPrecisionMetric::new;
        }

        public static Supplier<Metric> binaryPrecision(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new BinaryPrecisionMetric(threshold);
        }

        public static Supplier<Metric> binaryRecall() {
            return BinaryRecallMetric::new;
        }

        public static Supplier<Metric> binaryRecall(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new BinaryRecallMetric(threshold);
        }

        public static Supplier<Metric> binaryF1() {
            return BinaryF1Metric::new;
        }

        public static Supplier<Metric> binaryF1(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new BinaryF1Metric(threshold);
        }

        public static Supplier<Metric> binaryRocAuc() {
            return BinaryRocAucMetric::new;
        }

        public static Supplier<Metric> binaryAuroc() {
            return binaryRocAuc();
        }

        public static Supplier<Metric> binaryAveragePrecision() {
            return BinaryAveragePrecisionMetric::new;
        }

        public static Supplier<Metric> multiLabelExactMatch() {
            return MultiLabelExactMatchMetric::new;
        }

        public static Supplier<Metric> multiLabelExactMatch(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new MultiLabelExactMatchMetric(threshold);
        }

        public static Supplier<Metric> multiLabelHammingLoss() {
            return MultiLabelHammingLossMetric::new;
        }

        public static Supplier<Metric> multiLabelHammingLoss(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new MultiLabelHammingLossMetric(threshold);
        }

        public static Supplier<Metric> multiLabelMacroPrecision() {
            return MultiLabelMacroPrecisionMetric::new;
        }

        public static Supplier<Metric> multiLabelMacroPrecision(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new MultiLabelMacroPrecisionMetric(threshold);
        }

        public static Supplier<Metric> multiLabelMacroRecall() {
            return MultiLabelMacroRecallMetric::new;
        }

        public static Supplier<Metric> multiLabelMacroRecall(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new MultiLabelMacroRecallMetric(threshold);
        }

        public static Supplier<Metric> multiLabelMacroF1() {
            return MultiLabelMacroF1Metric::new;
        }

        public static Supplier<Metric> multiLabelMacroF1(float logitThreshold) {
            float threshold = requireFiniteLogitThreshold(logitThreshold);
            return () -> new MultiLabelMacroF1Metric(threshold);
        }

        public static Supplier<Metric> multiLabelMacroRocAuc() {
            return MultiLabelMacroRocAucMetric::new;
        }

        public static Supplier<Metric> multiLabelMacroAuroc() {
            return multiLabelMacroRocAuc();
        }

        public static Supplier<Metric> multiLabelMacroAveragePrecision() {
            return MultiLabelMacroAveragePrecisionMetric::new;
        }

        public static Supplier<Metric> precision() {
            return PrecisionMetric::new;
        }

        public static Supplier<Metric> recall() {
            return RecallMetric::new;
        }

        public static Supplier<Metric> f1() {
            return F1Metric::new;
        }

        public static Supplier<Metric> macroF1() {
            return f1();
        }

        public static Supplier<Metric> classificationMacroRocAuc() {
            return ClassificationMacroRocAucMetric::new;
        }

        public static Supplier<Metric> classificationMacroAuroc() {
            return classificationMacroRocAuc();
        }

        public static Supplier<Metric> classificationMacroAveragePrecision() {
            return ClassificationMacroAveragePrecisionMetric::new;
        }

        public static Supplier<Metric> meanAbsoluteError() {
            return MeanAbsoluteErrorMetric::new;
        }

        public static Supplier<Metric> mae() {
            return meanAbsoluteError();
        }

        public static Supplier<Metric> meanSquaredError() {
            return MeanSquaredErrorMetric::new;
        }

        public static Supplier<Metric> mse() {
            return meanSquaredError();
        }

        public static Supplier<Metric> rootMeanSquaredError() {
            return RootMeanSquaredErrorMetric::new;
        }

        public static Supplier<Metric> rmse() {
            return rootMeanSquaredError();
        }

        public static Supplier<Metric> r2Score() {
            return R2ScoreMetric::new;
        }

        public static Supplier<Metric> r2() {
            return r2Score();
        }

        private static float requireFiniteLogitThreshold(float logitThreshold) {
            if (!Float.isFinite(logitThreshold)) {
                throw new IllegalArgumentException("logitThreshold must be finite, got: " + logitThreshold);
            }
            return logitThreshold;
        }
    }

    public enum SchedulerStepUnit {
        BATCH,
        EPOCH,
        VALIDATION
    }

    public enum BestModelMonitorMode {
        MIN {
            @Override
            boolean isImproved(double current, double best, double minDelta) {
                return current < best - minDelta;
            }
        },
        MAX {
            @Override
            boolean isImproved(double current, double best, double minDelta) {
                return current > best + minDelta;
            }
        };

        abstract boolean isImproved(double current, double best, double minDelta);
    }

    private static final class AccuracyMetric implements Metric {
        private long correct;
        private long total;

        @Override
        public String name() {
            return "accuracy";
        }

        @Override
        public void reset() {
            correct = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        "accuracy expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int classes = Math.toIntExact(predictionShape[1]);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int predictedClass = argmax(predictionData, row * classes, classes);
                int targetClass = targetClass(targets, row, batch, classes);
                if (predictedClass == targetClass) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }
    }

    private static final class ClassificationConfusionMatrixMetric implements DetailedMetric {
        private int classes = -1;
        private long[][] matrix = new long[0][0];
        private long correct;
        private long total;

        @Override
        public String name() {
            return "confusion_matrix_accuracy";
        }

        @Override
        public void reset() {
            classes = -1;
            matrix = new long[0][0];
            correct = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int currentClasses = Math.toIntExact(predictionShape[1]);
            ensureClassStorage(currentClasses);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int predictedClass = argmax(predictionData, row * currentClasses, currentClasses);
                int actualClass = targetClass(targets, row, batch, currentClasses);
                matrix[actualClass][predictedClass]++;
                if (predictedClass == actualClass) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "classification_confusion_matrix");
            details.put("classes", Math.max(0, classes));
            details.put("total", total);
            details.put("correct", correct);
            details.put("accuracy", value());
            details.put("rowMeaning", "actual_class");
            details.put("columnMeaning", "predicted_class");
            details.put("labels", classLabels());
            details.put("matrix", matrixRows());
            details.put("perClassPrecision", perClassPrecision());
            details.put("perClassRecall", perClassRecall());
            details.put("perClassF1", perClassF1());
            return details;
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                matrix = new long[classes][classes];
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        private List<Integer> classLabels() {
            if (classes <= 0) {
                return List.of();
            }
            List<Integer> labels = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                labels.add(classIndex);
            }
            return labels;
        }

        private List<List<Long>> matrixRows() {
            if (classes <= 0) {
                return List.of();
            }
            List<List<Long>> rows = new ArrayList<>(classes);
            for (int row = 0; row < classes; row++) {
                List<Long> values = new ArrayList<>(classes);
                for (int column = 0; column < classes; column++) {
                    values.add(matrix[row][column]);
                }
                rows.add(Collections.unmodifiableList(values));
            }
            return Collections.unmodifiableList(rows);
        }

        private List<Double> perClassPrecision() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                long predicted = 0L;
                for (int row = 0; row < classes; row++) {
                    predicted += matrix[row][classIndex];
                }
                values.add(predicted == 0L ? 0.0 : (double) matrix[classIndex][classIndex] / predicted);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perClassRecall() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                long actual = 0L;
                for (int column = 0; column < classes; column++) {
                    actual += matrix[classIndex][column];
                }
                values.add(actual == 0L ? 0.0 : (double) matrix[classIndex][classIndex] / actual);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perClassF1() {
            List<Double> precision = perClassPrecision();
            List<Double> recall = perClassRecall();
            if (precision.isEmpty()) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(precision.size());
            for (int classIndex = 0; classIndex < precision.size(); classIndex++) {
                double p = precision.get(classIndex);
                double r = recall.get(classIndex);
                values.add(p + r == 0.0 ? 0.0 : (2.0 * p * r) / (p + r));
            }
            return Collections.unmodifiableList(values);
        }
    }

    private static final class TopKAccuracyMetric implements Metric {
        private final int k;
        private long correct;
        private long total;

        private TopKAccuracyMetric(int k) {
            this.k = k;
        }

        @Override
        public String name() {
            return "top" + k + "_accuracy";
        }

        @Override
        public void reset() {
            correct = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int classes = Math.toIntExact(predictionShape[1]);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int targetClass = targetClass(targets, row, batch, classes);
                if (containsTopK(predictionData, row * classes, classes, targetClass, Math.min(k, classes))) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }
    }

    private static final class MeanAbsoluteErrorMetric implements Metric {
        private double totalError;
        private long count;

        @Override
        public String name() {
            return "mae";
        }

        @Override
        public void reset() {
            totalError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape("mae", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                totalError += Math.abs(predictionData[i] - targetData[i]);
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalError / count;
        }
    }

    private abstract static class BinaryStatsMetric implements Metric {
        private final float logitThreshold;
        private long truePositive;
        private long trueNegative;
        private long falsePositive;
        private long falseNegative;
        private long total;

        BinaryStatsMetric() {
            this(0.0f);
        }

        BinaryStatsMetric(float logitThreshold) {
            this.logitThreshold = logitThreshold;
        }

        @Override
        public void reset() {
            truePositive = 0;
            trueNegative = 0;
            falsePositive = 0;
            falseNegative = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                boolean predictedPositive = predictionData[i] >= logitThreshold;
                boolean actualPositive = binaryTarget(targetData[i]);
                if (predictedPositive && actualPositive) {
                    truePositive++;
                } else if (predictedPositive) {
                    falsePositive++;
                } else if (actualPositive) {
                    falseNegative++;
                } else {
                    trueNegative++;
                }
                total++;
            }
        }

        protected double binaryAccuracy() {
            return total == 0 ? Double.NaN : (double) (truePositive + trueNegative) / total;
        }

        protected double binaryPrecision() {
            long denominator = truePositive + falsePositive;
            return denominator == 0 ? 0.0 : (double) truePositive / denominator;
        }

        protected double binaryRecall() {
            long denominator = truePositive + falseNegative;
            return denominator == 0 ? 0.0 : (double) truePositive / denominator;
        }

        protected double binaryF1() {
            long denominator = 2 * truePositive + falsePositive + falseNegative;
            return denominator == 0 ? 0.0 : (double) (2 * truePositive) / denominator;
        }

        protected double binarySpecificity() {
            long denominator = trueNegative + falsePositive;
            return denominator == 0 ? 0.0 : (double) trueNegative / denominator;
        }

        protected double binaryNegativePredictiveValue() {
            long denominator = trueNegative + falseNegative;
            return denominator == 0 ? 0.0 : (double) trueNegative / denominator;
        }

        protected double binaryBalancedAccuracy() {
            return total == 0 ? Double.NaN : (binaryRecall() + binarySpecificity()) / 2.0;
        }

        protected Map<String, Object> binaryConfusionDetails(String type) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", type);
            details.put("threshold", logitThreshold);
            details.put("total", total);
            details.put("trueNegative", trueNegative);
            details.put("falsePositive", falsePositive);
            details.put("falseNegative", falseNegative);
            details.put("truePositive", truePositive);
            details.put("accuracy", binaryAccuracy());
            details.put("precision", binaryPrecision());
            details.put("recall", binaryRecall());
            details.put("f1", binaryF1());
            details.put("specificity", binarySpecificity());
            details.put("negativePredictiveValue", binaryNegativePredictiveValue());
            details.put("balancedAccuracy", binaryBalancedAccuracy());
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("labels", List.of(0, 1));
            details.put("matrix", List.of(
                    List.of(trueNegative, falsePositive),
                    List.of(falseNegative, truePositive)));
            return details;
        }
    }

    private static final class BinaryAccuracyMetric extends BinaryStatsMetric {
        BinaryAccuracyMetric() {
        }

        BinaryAccuracyMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_accuracy";
        }

        @Override
        public double value() {
            return binaryAccuracy();
        }
    }

    private static final class BinaryConfusionMatrixMetric extends BinaryStatsMetric implements DetailedMetric {
        BinaryConfusionMatrixMetric() {
        }

        BinaryConfusionMatrixMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_confusion_matrix_accuracy";
        }

        @Override
        public double value() {
            return binaryAccuracy();
        }

        @Override
        public Map<String, Object> details() {
            return binaryConfusionDetails("binary_confusion_matrix");
        }
    }

    private static final class BinaryPrecisionMetric extends BinaryStatsMetric {
        BinaryPrecisionMetric() {
        }

        BinaryPrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_precision";
        }

        @Override
        public double value() {
            return binaryPrecision();
        }
    }

    private static final class BinaryRecallMetric extends BinaryStatsMetric {
        BinaryRecallMetric() {
        }

        BinaryRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_recall";
        }

        @Override
        public double value() {
            return binaryRecall();
        }
    }

    private static final class BinaryF1Metric extends BinaryStatsMetric {
        BinaryF1Metric() {
        }

        BinaryF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_f1";
        }

        @Override
        public double value() {
            return binaryF1();
        }
    }

    private abstract static class BinaryRankingMetric implements Metric {
        private final List<BinaryScore> scores = new ArrayList<>();

        @Override
        public void reset() {
            scores.clear();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                scores.add(new BinaryScore(predictionData[i], binaryTarget(targetData[i])));
            }
        }

        protected List<BinaryScore> scores() {
            return scores;
        }

        protected long positiveCount() {
            return scores.stream().filter(BinaryScore::positive).count();
        }
    }

    private static final class BinaryRocAucMetric extends BinaryRankingMetric {
        @Override
        public String name() {
            return "binary_roc_auc";
        }

        @Override
        public double value() {
            return binaryRocAuc(scores());
        }
    }

    private static final class BinaryAveragePrecisionMetric extends BinaryRankingMetric {
        @Override
        public String name() {
            return "binary_average_precision";
        }

        @Override
        public double value() {
            return binaryAveragePrecision(scores());
        }
    }

    private record BinaryScore(float score, boolean positive) {
    }

    private static double binaryRocAuc(List<BinaryScore> rawScores) {
        List<BinaryScore> values = new ArrayList<>(rawScores);
        long positives = values.stream().filter(BinaryScore::positive).count();
        long negatives = values.size() - positives;
        if (positives == 0 || negatives == 0) {
            return Double.NaN;
        }

        values.sort(Comparator.comparingDouble(BinaryScore::score));
        double positiveRankSum = 0.0;
        int index = 0;
        while (index < values.size()) {
            int groupEnd = index + 1;
            while (groupEnd < values.size()
                    && Float.compare(values.get(groupEnd).score(), values.get(index).score()) == 0) {
                groupEnd++;
            }
            double averageRank = ((index + 1.0) + groupEnd) / 2.0;
            for (int i = index; i < groupEnd; i++) {
                if (values.get(i).positive()) {
                    positiveRankSum += averageRank;
                }
            }
            index = groupEnd;
        }

        double positiveRankBaseline = positives * (positives + 1.0) / 2.0;
        return (positiveRankSum - positiveRankBaseline) / (positives * (double) negatives);
    }

    private static double binaryAveragePrecision(List<BinaryScore> rawScores) {
        List<BinaryScore> values = new ArrayList<>(rawScores);
        long positives = values.stream().filter(BinaryScore::positive).count();
        if (positives == 0) {
            return Double.NaN;
        }

        values.sort(Comparator.comparingDouble(BinaryScore::score).reversed());
        long truePositive = 0;
        long falsePositive = 0;
        double ap = 0.0;
        int index = 0;
        while (index < values.size()) {
            int groupEnd = index + 1;
            long groupPositive = values.get(index).positive() ? 1 : 0;
            long groupNegative = values.get(index).positive() ? 0 : 1;
            while (groupEnd < values.size()
                    && Float.compare(values.get(groupEnd).score(), values.get(index).score()) == 0) {
                if (values.get(groupEnd).positive()) {
                    groupPositive++;
                } else {
                    groupNegative++;
                }
                groupEnd++;
            }

            truePositive += groupPositive;
            falsePositive += groupNegative;
            if (groupPositive > 0) {
                double precisionAtThreshold = truePositive / (double) (truePositive + falsePositive);
                double recallIncrease = groupPositive / (double) positives;
                ap += recallIncrease * precisionAtThreshold;
            }
            index = groupEnd;
        }
        return ap;
    }

    private abstract static class MultiLabelStatsMetric implements Metric {
        private final float logitThreshold;
        private int labels = -1;
        private long[] truePositive = new long[0];
        private long[] falsePositive = new long[0];
        private long[] falseNegative = new long[0];
        private long exactMatchCount;
        private long sampleCount;
        private long labelCount;
        private long labelMismatchCount;

        MultiLabelStatsMetric() {
            this(0.0f);
        }

        MultiLabelStatsMetric(float logitThreshold) {
            this.logitThreshold = logitThreshold;
        }

        @Override
        public void reset() {
            labels = -1;
            truePositive = new long[0];
            falsePositive = new long[0];
            falseNegative = new long[0];
            exactMatchCount = 0;
            sampleCount = 0;
            labelCount = 0;
            labelMismatchCount = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape(name(), predictions, targets);
            long[] shape = predictions.shape();
            int currentSamples = multiLabelSampleCount(shape);
            int currentLabels = multiLabelLabelsPerSample(shape);
            ensureLabelStorage(currentLabels);

            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int row = 0; row < currentSamples; row++) {
                boolean exactMatch = true;
                int offset = row * currentLabels;
                for (int label = 0; label < currentLabels; label++) {
                    int index = offset + label;
                    boolean predictedPositive = predictionData[index] >= logitThreshold;
                    boolean actualPositive = binaryTarget(targetData[index]);
                    if (predictedPositive && actualPositive) {
                        truePositive[label]++;
                    } else if (predictedPositive) {
                        falsePositive[label]++;
                    } else if (actualPositive) {
                        falseNegative[label]++;
                    }
                    if (predictedPositive != actualPositive) {
                        exactMatch = false;
                        labelMismatchCount++;
                    }
                    labelCount++;
                }
                if (exactMatch) {
                    exactMatchCount++;
                }
                sampleCount++;
            }
        }

        private void ensureLabelStorage(int currentLabels) {
            if (labels < 0) {
                labels = currentLabels;
                truePositive = new long[labels];
                falsePositive = new long[labels];
                falseNegative = new long[labels];
                return;
            }
            if (labels != currentLabels) {
                throw new IllegalArgumentException(
                        name() + " expected " + labels + " labels per sample but got " + currentLabels);
            }
        }

        protected double exactMatch() {
            return sampleCount == 0 ? Double.NaN : (double) exactMatchCount / sampleCount;
        }

        protected double hammingLoss() {
            return labelCount == 0 ? Double.NaN : (double) labelMismatchCount / labelCount;
        }

        protected double macroPrecision() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = truePositive[label] + falsePositive[label];
                total += denominator == 0 ? 0.0 : (double) truePositive[label] / denominator;
            }
            return total / labels;
        }

        protected double macroRecall() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = truePositive[label] + falseNegative[label];
                total += denominator == 0 ? 0.0 : (double) truePositive[label] / denominator;
            }
            return total / labels;
        }

        protected double macroF1() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = 2 * truePositive[label] + falsePositive[label] + falseNegative[label];
                total += denominator == 0 ? 0.0 : (double) (2 * truePositive[label]) / denominator;
            }
            return total / labels;
        }
    }

    private static final class MultiLabelExactMatchMetric extends MultiLabelStatsMetric {
        MultiLabelExactMatchMetric() {
        }

        MultiLabelExactMatchMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_exact_match";
        }

        @Override
        public double value() {
            return exactMatch();
        }
    }

    private static final class MultiLabelHammingLossMetric extends MultiLabelStatsMetric {
        MultiLabelHammingLossMetric() {
        }

        MultiLabelHammingLossMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_hamming_loss";
        }

        @Override
        public double value() {
            return hammingLoss();
        }
    }

    private static final class MultiLabelMacroPrecisionMetric extends MultiLabelStatsMetric {
        MultiLabelMacroPrecisionMetric() {
        }

        MultiLabelMacroPrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_precision";
        }

        @Override
        public double value() {
            return macroPrecision();
        }
    }

    private static final class MultiLabelMacroRecallMetric extends MultiLabelStatsMetric {
        MultiLabelMacroRecallMetric() {
        }

        MultiLabelMacroRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_recall";
        }

        @Override
        public double value() {
            return macroRecall();
        }
    }

    private static final class MultiLabelMacroF1Metric extends MultiLabelStatsMetric {
        MultiLabelMacroF1Metric() {
        }

        MultiLabelMacroF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_f1";
        }

        @Override
        public double value() {
            return macroF1();
        }
    }

    private abstract static class MultiLabelRankingMetric implements Metric {
        private int labels = -1;
        private List<List<BinaryScore>> scoresByLabel = List.of();

        @Override
        public void reset() {
            labels = -1;
            scoresByLabel = List.of();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape(name(), predictions, targets);
            long[] shape = predictions.shape();
            int currentSamples = multiLabelSampleCount(shape);
            int currentLabels = multiLabelLabelsPerSample(shape);
            ensureLabelStorage(currentLabels);

            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int row = 0; row < currentSamples; row++) {
                int offset = row * currentLabels;
                for (int label = 0; label < currentLabels; label++) {
                    int index = offset + label;
                    scoresByLabel.get(label).add(new BinaryScore(
                            predictionData[index],
                            binaryTarget(targetData[index])));
                }
            }
        }

        private void ensureLabelStorage(int currentLabels) {
            if (labels < 0) {
                labels = currentLabels;
                List<List<BinaryScore>> lists = new ArrayList<>(labels);
                for (int i = 0; i < labels; i++) {
                    lists.add(new ArrayList<>());
                }
                scoresByLabel = lists;
                return;
            }
            if (labels != currentLabels) {
                throw new IllegalArgumentException(
                        name() + " expected " + labels + " labels per sample but got " + currentLabels);
            }
        }

        protected double macroRocAuc() {
            return macroDefinedScore(true);
        }

        protected double macroAveragePrecision() {
            return macroDefinedScore(false);
        }

        private double macroDefinedScore(boolean rocAuc) {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            int defined = 0;
            for (List<BinaryScore> labelScores : scoresByLabel) {
                double score = rocAuc
                        ? binaryRocAuc(labelScores)
                        : binaryAveragePrecision(labelScores);
                if (Double.isFinite(score)) {
                    total += score;
                    defined++;
                }
            }
            return defined == 0 ? Double.NaN : total / defined;
        }
    }

    private static final class MultiLabelMacroRocAucMetric extends MultiLabelRankingMetric {
        @Override
        public String name() {
            return "multilabel_macro_roc_auc";
        }

        @Override
        public double value() {
            return macroRocAuc();
        }
    }

    private static final class MultiLabelMacroAveragePrecisionMetric extends MultiLabelRankingMetric {
        @Override
        public String name() {
            return "multilabel_macro_average_precision";
        }

        @Override
        public double value() {
            return macroAveragePrecision();
        }
    }

    private abstract static class ClassificationStatsMetric implements Metric {
        private int classes = -1;
        private long[] truePositive = new long[0];
        private long[] falsePositive = new long[0];
        private long[] falseNegative = new long[0];

        @Override
        public void reset() {
            classes = -1;
            truePositive = new long[0];
            falsePositive = new long[0];
            falseNegative = new long[0];
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int currentClasses = Math.toIntExact(predictionShape[1]);
            ensureClassStorage(currentClasses);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int predictedClass = argmax(predictionData, row * currentClasses, currentClasses);
                int actualClass = targetClass(targets, row, batch, currentClasses);
                if (predictedClass == actualClass) {
                    truePositive[actualClass]++;
                } else {
                    falsePositive[predictedClass]++;
                    falseNegative[actualClass]++;
                }
            }
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                truePositive = new long[classes];
                falsePositive = new long[classes];
                falseNegative = new long[classes];
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        protected double macroPrecision() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = truePositive[i] + falsePositive[i];
                total += denominator == 0 ? 0.0 : (double) truePositive[i] / denominator;
            }
            return total / classes;
        }

        protected double macroRecall() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = truePositive[i] + falseNegative[i];
                total += denominator == 0 ? 0.0 : (double) truePositive[i] / denominator;
            }
            return total / classes;
        }

        protected double macroF1() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = 2 * truePositive[i] + falsePositive[i] + falseNegative[i];
                total += denominator == 0 ? 0.0 : (double) (2 * truePositive[i]) / denominator;
            }
            return total / classes;
        }
    }

    private static final class PrecisionMetric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "precision";
        }

        @Override
        public double value() {
            return macroPrecision();
        }
    }

    private static final class RecallMetric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "recall";
        }

        @Override
        public double value() {
            return macroRecall();
        }
    }

    private static final class F1Metric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "f1";
        }

        @Override
        public double value() {
            return macroF1();
        }
    }

    private abstract static class ClassificationRankingMetric implements Metric {
        private int classes = -1;
        private List<List<BinaryScore>> scoresByClass = List.of();

        @Override
        public void reset() {
            classes = -1;
            scoresByClass = List.of();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int currentClasses = Math.toIntExact(predictionShape[1]);
            ensureClassStorage(currentClasses);

            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int actualClass = targetClass(targets, row, batch, currentClasses);
                int offset = row * currentClasses;
                for (int classIndex = 0; classIndex < currentClasses; classIndex++) {
                    scoresByClass.get(classIndex).add(new BinaryScore(
                            predictionData[offset + classIndex],
                            classIndex == actualClass));
                }
            }
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                List<List<BinaryScore>> lists = new ArrayList<>(classes);
                for (int i = 0; i < classes; i++) {
                    lists.add(new ArrayList<>());
                }
                scoresByClass = lists;
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        protected double macroRocAuc() {
            return macroDefinedScore(true);
        }

        protected double macroAveragePrecision() {
            return macroDefinedScore(false);
        }

        private double macroDefinedScore(boolean rocAuc) {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            int defined = 0;
            for (List<BinaryScore> classScores : scoresByClass) {
                double score = rocAuc
                        ? binaryRocAuc(classScores)
                        : binaryAveragePrecision(classScores);
                if (Double.isFinite(score)) {
                    total += score;
                    defined++;
                }
            }
            return defined == 0 ? Double.NaN : total / defined;
        }
    }

    private static final class ClassificationMacroRocAucMetric extends ClassificationRankingMetric {
        @Override
        public String name() {
            return "classification_macro_roc_auc";
        }

        @Override
        public double value() {
            return macroRocAuc();
        }
    }

    private static final class ClassificationMacroAveragePrecisionMetric extends ClassificationRankingMetric {
        @Override
        public String name() {
            return "classification_macro_average_precision";
        }

        @Override
        public double value() {
            return macroAveragePrecision();
        }
    }

    private static final class MeanSquaredErrorMetric implements Metric {
        private double totalSquaredError;
        private long count;

        @Override
        public String name() {
            return "mse";
        }

        @Override
        public void reset() {
            totalSquaredError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape("mse", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double diff = predictionData[i] - targetData[i];
                totalSquaredError += diff * diff;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalSquaredError / count;
        }
    }

    private static final class RootMeanSquaredErrorMetric implements Metric {
        private double totalSquaredError;
        private long count;

        @Override
        public String name() {
            return "rmse";
        }

        @Override
        public void reset() {
            totalSquaredError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape("rmse", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double diff = predictionData[i] - targetData[i];
                totalSquaredError += diff * diff;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : Math.sqrt(totalSquaredError / count);
        }
    }

    private static final class R2ScoreMetric implements Metric {
        private double sumSquaredError;
        private double targetSum;
        private double targetSquaredSum;
        private long count;

        @Override
        public String name() {
            return "r2";
        }

        @Override
        public void reset() {
            sumSquaredError = 0.0;
            targetSum = 0.0;
            targetSquaredSum = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            requireSameShape("r2", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = targetData[i];
                double diff = predictionData[i] - target;
                sumSquaredError += diff * diff;
                targetSum += target;
                targetSquaredSum += target * target;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            if (count == 0) {
                return Double.NaN;
            }
            double totalVariance = targetSquaredSum - (targetSum * targetSum / count);
            if (Math.abs(totalVariance) < 1e-12) {
                return Math.abs(sumSquaredError) < 1e-12 ? 1.0 : 0.0;
            }
            return 1.0 - (sumSquaredError / totalVariance);
        }
    }

    private static void requireSameShape(String metricName, GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (!Arrays.equals(predictionShape, targetShape)) {
            throw new IllegalArgumentException(
                    metricName + " expects predictions and targets with the same shape, got "
                            + Arrays.toString(predictionShape) + " vs " + Arrays.toString(targetShape));
        }
    }

    private static void requireSameElementCount(String metricName, GradTensor predictions, GradTensor targets) {
        if (predictions.numel() != targets.numel()) {
            throw new IllegalArgumentException(
                    metricName + " expects predictions and targets with the same element count, got "
                            + predictions.numel() + " vs " + targets.numel());
        }
    }

    private static int multiLabelSampleCount(long[] shape) {
        if (shape.length == 0) {
            return 1;
        }
        return Math.toIntExact(shape[0]);
    }

    private static int multiLabelLabelsPerSample(long[] shape) {
        if (shape.length <= 1) {
            return 1;
        }
        long labels = 1;
        for (int i = 1; i < shape.length; i++) {
            labels = Math.multiplyExact(labels, shape[i]);
        }
        return Math.toIntExact(labels);
    }

    private static boolean binaryTarget(float value) {
        if (Math.abs(value) <= 1e-6f) {
            return false;
        }
        if (Math.abs(value - 1.0f) <= 1e-6f) {
            return true;
        }
        throw new IllegalArgumentException("binary targets must contain only 0.0 or 1.0, got: " + value);
    }

    private static int argmax(float[] values, int offset, int length) {
        int best = 0;
        float bestValue = values[offset];
        for (int i = 1; i < length; i++) {
            float value = values[offset + i];
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    private static boolean containsTopK(float[] values, int offset, int length, int targetIndex, int k) {
        float targetValue = values[offset + targetIndex];
        int strictlyGreater = 0;
        int equalBeforeTarget = 0;
        for (int i = 0; i < length; i++) {
            if (i == targetIndex) {
                continue;
            }
            float value = values[offset + i];
            if (value > targetValue) {
                strictlyGreater++;
            } else if (value == targetValue && i < targetIndex) {
                equalBeforeTarget++;
            }
        }
        return strictlyGreater + equalBeforeTarget < k;
    }

    private static int targetClass(GradTensor targets, int row, int expectedBatch, int classes) {
        long[] targetShape = targets.shape();
        float[] targetData = targets.data();
        if (targetShape.length == 1 && targetShape[0] == expectedBatch) {
            return checkedTargetClass((int) targetData[row], classes);
        }
        if (targetShape.length == 2 && targetShape[0] == expectedBatch && targetShape[1] == classes) {
            return argmax(targetData, row * classes, classes);
        }
        throw new IllegalArgumentException(
                "classification metric expects targets shaped [batch] class indices or [batch, classes] one-hot labels, got "
                        + Arrays.toString(targetShape));
    }

    private static int checkedTargetClass(int targetClass, int classes) {
        if (targetClass < 0 || targetClass >= classes) {
            throw new IllegalArgumentException(
                    "target class " + targetClass + " out of range [0, " + (classes - 1) + "]");
        }
        return targetClass;
    }

    private record TensorDiagnostics(int tensorCount, long valueCount, double l2Norm, double maxAbs) {
    }

    public static final class Builder {
        private NNModule model;
        private Optimizer optimizer;
        private LRScheduler learningRateScheduler;
        private SchedulerStepUnit schedulerStepUnit = SchedulerStepUnit.BATCH;
        private String schedulerMonitorMetric;
        private LossFunction lossFunction;
        private final List<Supplier<? extends Metric>> metricFactories = new ArrayList<>();
        private int epochs = 1;
        private double gradientClip = 0.0;
        private int gradientAccumulationSteps = 1;
        private boolean mixedPrecision = false;
        private Path checkpointDir;
        private int earlyStoppingPatience = 0;
        private double earlyStoppingMinDelta = 0.0;
        private String earlyStoppingMonitorMetric;
        private BestModelMonitorMode earlyStoppingMonitorMode = BestModelMonitorMode.MIN;
        private boolean resumeFromCheckpoint = false;
        private boolean failOnCheckpointLoadError = true;
        private boolean saveBestModelCheckpoint = true;
        private boolean restoreBestModelAtEnd = false;
        private String bestModelMonitorMetric;
        private BestModelMonitorMode bestModelMonitorMode = BestModelMonitorMode.MIN;
        private String preferredDevice = "auto";
        private final List<TrainingListener> listeners = new ArrayList<>();

        private Builder() {
        }

        public Builder model(NNModule model) {
            this.model = model;
            return this;
        }

        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder scheduler(LRScheduler scheduler) {
            return scheduler(scheduler, SchedulerStepUnit.BATCH);
        }

        public Builder scheduler(LRScheduler scheduler, SchedulerStepUnit stepUnit) {
            this.learningRateScheduler = scheduler;
            this.schedulerStepUnit = stepUnit == null ? SchedulerStepUnit.BATCH : stepUnit;
            return this;
        }

        public Builder learningRateScheduler(LRScheduler scheduler) {
            return scheduler(scheduler);
        }

        public Builder learningRateScheduler(LRScheduler scheduler, SchedulerStepUnit stepUnit) {
            return scheduler(scheduler, stepUnit);
        }

        public Builder schedulerMonitorMetric(String metricName) {
            this.schedulerMonitorMetric = normalizeBestModelMonitorMetric(metricName);
            return this;
        }

        public Builder schedulerMonitorValidationLoss() {
            this.schedulerMonitorMetric = null;
            return this;
        }

        public Builder loss(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder metric(Supplier<? extends Metric> metricFactory) {
            this.metricFactories.add(Objects.requireNonNull(metricFactory, "metric factory must not be null"));
            return this;
        }

        public Builder metrics(List<? extends Supplier<? extends Metric>> metricFactories) {
            if (metricFactories == null) {
                return this;
            }
            for (Supplier<? extends Metric> metricFactory : metricFactories) {
                metric(metricFactory);
            }
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = Math.max(1, epochs);
            return this;
        }

        public Builder gradientClip(double gradientClip) {
            this.gradientClip = Math.max(0.0, gradientClip);
            return this;
        }

        public Builder gradientAccumulationSteps(int steps) {
            this.gradientAccumulationSteps = Math.max(1, steps);
            return this;
        }

        public Builder mixedPrecision(boolean mixedPrecision) {
            this.mixedPrecision = mixedPrecision;
            return this;
        }

        public Builder device(String deviceId) {
            this.preferredDevice = normalizeDevice(deviceId);
            return this;
        }

        public Builder accelerator(String deviceId) {
            return device(deviceId);
        }

        public Builder checkpointDir(Path checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        public Builder resumeFromCheckpoint() {
            return resumeFromCheckpoint(true);
        }

        public Builder resumeFromCheckpoint(boolean resumeFromCheckpoint) {
            this.resumeFromCheckpoint = resumeFromCheckpoint;
            return this;
        }

        public Builder failOnCheckpointLoadError(boolean failOnCheckpointLoadError) {
            this.failOnCheckpointLoadError = failOnCheckpointLoadError;
            return this;
        }

        public Builder saveBestModelCheckpoint(boolean saveBestModelCheckpoint) {
            this.saveBestModelCheckpoint = saveBestModelCheckpoint;
            return this;
        }

        public Builder bestModelCheckpoint(boolean saveBestModelCheckpoint) {
            return saveBestModelCheckpoint(saveBestModelCheckpoint);
        }

        public Builder restoreBestModelAtEnd() {
            return restoreBestModelAtEnd(true);
        }

        public Builder restoreBestModelAtEnd(boolean restoreBestModelAtEnd) {
            this.restoreBestModelAtEnd = restoreBestModelAtEnd;
            if (restoreBestModelAtEnd) {
                this.saveBestModelCheckpoint = true;
            }
            return this;
        }

        public Builder bestModelMonitorMetric(String metricName) {
            return bestModelMonitorMetric(metricName, BestModelMonitorMode.MAX);
        }

        public Builder bestModelMonitorMetric(String metricName, BestModelMonitorMode mode) {
            this.bestModelMonitorMetric = normalizeBestModelMonitorMetric(metricName);
            this.bestModelMonitorMode = mode == null ? BestModelMonitorMode.MAX : mode;
            return this;
        }

        public Builder bestModelMonitorValidationLoss() {
            return bestModelMonitorValidationLoss(BestModelMonitorMode.MIN);
        }

        public Builder bestModelMonitorValidationLoss(BestModelMonitorMode mode) {
            this.bestModelMonitorMetric = null;
            this.bestModelMonitorMode = mode == null ? BestModelMonitorMode.MIN : mode;
            return this;
        }

        public Builder earlyStoppingMonitorMetric(String metricName) {
            return earlyStoppingMonitorMetric(metricName, BestModelMonitorMode.MAX);
        }

        public Builder earlyStoppingMonitorMetric(String metricName, BestModelMonitorMode mode) {
            this.earlyStoppingMonitorMetric = normalizeBestModelMonitorMetric(metricName);
            this.earlyStoppingMonitorMode = mode == null ? BestModelMonitorMode.MAX : mode;
            return this;
        }

        public Builder earlyStoppingMonitorValidationLoss() {
            this.earlyStoppingMonitorMetric = null;
            this.earlyStoppingMonitorMode = BestModelMonitorMode.MIN;
            return this;
        }

        public Builder earlyStopping(int patience) {
            this.earlyStoppingPatience = Math.max(0, patience);
            this.earlyStoppingMinDelta = Math.max(0.0, this.earlyStoppingMinDelta);
            return this;
        }

        public Builder earlyStopping(int patience, double minDelta) {
            this.earlyStoppingPatience = Math.max(0, patience);
            this.earlyStoppingMinDelta = Math.max(0.0, minDelta);
            return this;
        }

        public Builder callback(TrainingListener listener) {
            return listener(listener);
        }

        public Builder listener(TrainingListener listener) {
            if (listener != null) {
                listeners.add(listener);
            }
            return this;
        }

        public Builder listeners(List<? extends TrainingListener> listeners) {
            if (listeners == null) {
                return this;
            }
            for (TrainingListener listener : listeners) {
                if (listener != null) {
                    this.listeners.add(listener);
                }
            }
            return this;
        }

        public CanonicalTrainer build() {
            return new CanonicalTrainer(this);
        }
    }
}
