package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.optim.GradScaler;
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

    private static final int MODEL_CHECKPOINT_METADATA_VERSION = 1;
    private static final int CHECKPOINT_MANIFEST_VERSION = 1;

    private final NNModule model;
    private final Optimizer optimizer;
    private final GradScaler gradScaler;
    private final boolean mixedPrecision;
    private final LRScheduler learningRateScheduler;
    private final SchedulerStepUnit schedulerStepUnit;
    private final TrainerLearningRateSchedulerStepper schedulerStepper;
    private final String schedulerMonitorMetric;
    private final TrainingLossFunction lossFunction;
    private final List<TrainingMetric> trainMetrics;
    private final List<TrainingMetric> validationMetrics;
    private final TrainerFailureState failureState = new TrainerFailureState();
    private final TrainerMetricSnapshots.FailureRecorder metricFailures =
            new TrainerMetricFailureRecorder(failureState);
    private final TrainerBatchGuards.FailureRecorder batchFailures;
    private final TrainerMetricRuntime metricRuntime;
    private final TrainerBatchRuntime batchRuntime;
    private final double gradientClip;
    private final int earlyStoppingPatience;
    private final double earlyStoppingMinDelta;
    private final String earlyStoppingMonitorMetric;
    private final BestModelMonitorMode earlyStoppingMonitorMode;
    private final TrainerEarlyStoppingMonitor earlyStoppingMonitor;
    private final int gradientAccumulationSteps;
    private final boolean resumeFromCheckpoint;
    private final boolean failOnCheckpointLoadError;
    private final boolean saveBestModelCheckpoint;
    private final boolean restoreBestModelAtEnd;
    private final String bestModelMonitorMetric;
    private final BestModelMonitorMode bestModelMonitorMode;
    private final TrainerBestModelCheckpointMonitor bestModelCheckpointMonitor;
    private final String preferredDevice;
    private final TrainerCheckpointLayout checkpointLayout;
    private final Path modelCheckpointFile;
    private final Path modelCheckpointMetadataFile;
    private final Path bestModelCheckpointFile;
    private final Path optimizerCheckpointFile;
    private final Path schedulerCheckpointFile;
    private final Path gradScalerCheckpointFile;
    private final Path historyFile;
    private final Path reportFile;
    private final Path runtimeCheckpointFile;
    private final Path checkpointManifestFile;
    private final AtomicBoolean modelCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean optimizerCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean schedulerCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean gradScalerCheckpointLoadAttempted = new AtomicBoolean(false);
    private final AtomicBoolean historyLoadAttempted = new AtomicBoolean(false);
    private volatile boolean modelCheckpointLoaded;
    private volatile boolean modelCheckpointSaved;
    private volatile boolean modelCheckpointMetadataLoaded;
    private volatile boolean modelCheckpointMetadataSaved;
    private volatile boolean modelCheckpointMetadataMissingOnResume;
    private volatile boolean modelCheckpointCompatibilityMismatch;
    private volatile boolean optimizerCheckpointLoaded;
    private volatile boolean optimizerCheckpointSaved;
    private volatile boolean schedulerCheckpointLoaded;
    private volatile boolean schedulerCheckpointSaved;
    private volatile boolean gradScalerCheckpointLoaded;
    private volatile boolean gradScalerCheckpointSaved;
    private volatile boolean gradScalerCheckpointFallbackUsed;
    private volatile boolean modelCheckpointMissingOnResume;
    private volatile boolean optimizerCheckpointMissingOnResume;
    private volatile boolean schedulerCheckpointMissingOnResume;
    private volatile boolean gradScalerCheckpointMissingOnResume;
    private volatile boolean historyMissingOnResume;
    private volatile boolean historyLoaded;
    private volatile boolean historySaved;
    private volatile boolean reportSaved;
    private volatile boolean checkpointManifestLoaded;
    private volatile boolean checkpointManifestSaved;
    private volatile boolean checkpointManifestMissingOnResume;
    private volatile boolean checkpointManifestIntegrityMismatch;
    private volatile boolean runtimeCheckpointIntegrityMismatch;
    private volatile boolean runtimeCheckpointResumeSkipped;
    private volatile String modelCheckpointLoadError;
    private volatile String modelCheckpointSaveError;
    private volatile String modelCheckpointMetadataLoadError;
    private volatile String modelCheckpointMetadataSaveError;
    private volatile String bestModelCheckpointSaveError;
    private volatile String bestModelCheckpointLoadError;
    private volatile String optimizerCheckpointLoadError;
    private volatile String optimizerCheckpointSaveError;
    private volatile String schedulerCheckpointLoadError;
    private volatile String schedulerCheckpointSaveError;
    private volatile String gradScalerCheckpointLoadError;
    private volatile String gradScalerCheckpointSaveError;
    private volatile String historyLoadError;
    private volatile String historySaveError;
    private volatile String reportSaveError;
    private volatile String checkpointManifestLoadError;
    private volatile String checkpointManifestSaveError;
    private volatile String runtimeCheckpointLoadError;
    private TrainerOptimizationRuntime optimizationRuntime;
    private volatile Acceleration.BackendStatus accelerationStatusAtStart;
    private volatile Map<String, Double> latestTrainMetrics = Map.of();
    private volatile Map<String, Double> latestValidationMetrics = Map.of();
    private volatile Map<String, Object> latestTrainMetricDetails = Map.of();
    private volatile Map<String, Object> latestValidationMetricDetails = Map.of();
    private final TrainerThroughputStats throughputStats = new TrainerThroughputStats();
    private final TrainerEpochHistory epochHistory = new TrainerEpochHistory();
    private final TrainerCheckpointResumeDiagnostics checkpointResumeDiagnostics =
            new TrainerCheckpointResumeDiagnostics();
    private final CanonicalTrainerRuntime runtime;
    private volatile TrainingSummary latestSummary;
    private volatile RuntimeException postTrainingCheckpointError;

    private CanonicalTrainer(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model must not be null");
        this.optimizer = Objects.requireNonNull(builder.optimizer, "optimizer must not be null");
        this.batchFailures = new TrainerBatchFailureRecorder(failureState, this::discardPendingGradients);
        this.gradScaler = builder.gradScaler != null
                ? builder.gradScaler
                : (builder.mixedPrecision ? GradScaler.builder().build() : null);
        this.mixedPrecision = this.gradScaler != null;
        this.learningRateScheduler = builder.learningRateScheduler;
        this.schedulerStepUnit = builder.schedulerStepUnit;
        this.schedulerStepper = new TrainerLearningRateSchedulerStepper(
                this.learningRateScheduler,
                this.schedulerStepUnit);
        this.schedulerMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                builder.schedulerMonitorMetric,
                "scheduler monitor metric");
        this.lossFunction = Objects.requireNonNull(builder.lossFunction, "loss function must not be null");
        this.trainMetrics = TrainerMetricSnapshots.instantiate(builder.metricFactories, "train");
        this.validationMetrics = TrainerMetricSnapshots.instantiate(builder.metricFactories, "validation");
        this.metricRuntime = new TrainerMetricRuntime(trainMetrics, validationMetrics, metricFailures);
        this.batchRuntime = new TrainerBatchRuntime(
                model,
                lossFunction,
                gradScaler,
                metricRuntime,
                throughputStats,
                batchFailures);
        this.gradientClip = Math.max(0.0, builder.gradientClip);
        this.earlyStoppingPatience = Math.max(0, builder.earlyStoppingPatience);
        this.earlyStoppingMinDelta = Math.max(0.0, builder.earlyStoppingMinDelta);
        this.earlyStoppingMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                builder.earlyStoppingMonitorMetric,
                "early stopping monitor metric");
        this.earlyStoppingMonitorMode = builder.earlyStoppingMonitorMode == null
                ? (this.earlyStoppingMonitorMetric == null ? BestModelMonitorMode.MIN : BestModelMonitorMode.MAX)
                : builder.earlyStoppingMonitorMode;
        this.earlyStoppingMonitor = new TrainerEarlyStoppingMonitor(
                this.earlyStoppingMonitorMetric,
                this.earlyStoppingPatience,
                this.earlyStoppingMinDelta,
                this.earlyStoppingMonitorMode);
        this.gradientAccumulationSteps = Math.max(1, builder.gradientAccumulationSteps);
        this.resumeFromCheckpoint = builder.resumeFromCheckpoint;
        this.failOnCheckpointLoadError = builder.failOnCheckpointLoadError;
        this.saveBestModelCheckpoint = builder.saveBestModelCheckpoint;
        this.restoreBestModelAtEnd = builder.restoreBestModelAtEnd;
        this.bestModelMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                builder.bestModelMonitorMetric,
                "best model monitor metric");
        this.bestModelMonitorMode = builder.bestModelMonitorMode == null
                ? (this.bestModelMonitorMetric == null ? BestModelMonitorMode.MIN : BestModelMonitorMode.MAX)
                : builder.bestModelMonitorMode;
        TrainerMonitorSupport.requireMetricPresent(
                this.bestModelMonitorMetric,
                this.validationMetrics,
                "best model monitor metric");
        TrainerMonitorSupport.requireMetricPresent(
                this.earlyStoppingMonitorMetric,
                this.validationMetrics,
                "early stopping monitor metric");
        TrainerMonitorSupport.requireMetricPresent(
                this.schedulerMonitorMetric,
                this.validationMetrics,
                "scheduler monitor metric");
        this.preferredDevice = TrainerMetadataSupport.normalizeDevice(builder.preferredDevice);
        this.checkpointLayout = TrainerCheckpointLayout.from(builder.checkpointDir);
        this.modelCheckpointFile = checkpointLayout.model();
        this.modelCheckpointMetadataFile = checkpointLayout.modelMetadata();
        this.bestModelCheckpointFile = checkpointLayout.bestModel();
        this.optimizerCheckpointFile = checkpointLayout.optimizer();
        this.schedulerCheckpointFile = checkpointLayout.scheduler();
        this.gradScalerCheckpointFile = checkpointLayout.gradScaler();
        this.historyFile = checkpointLayout.history();
        this.reportFile = checkpointLayout.report();
        this.runtimeCheckpointFile = checkpointLayout.runtime();
        this.checkpointManifestFile = checkpointLayout.manifest();
        this.bestModelCheckpointMonitor = new TrainerBestModelCheckpointMonitor(
                this.saveBestModelCheckpoint && this.bestModelCheckpointFile != null,
                this.bestModelMonitorMetric,
                this.bestModelMonitorMode,
                this.earlyStoppingMinDelta);
        this.optimizationRuntime = new TrainerOptimizationRuntime(
                optimizer,
                new TrainerOptimizerStepRunner(
                        optimizer,
                        gradScaler,
                        this.gradientClip,
                        batchFailures,
                        this::stepBatchScheduler),
                this.gradientAccumulationSteps,
                this.mixedPrecision,
                this.gradScaler,
                failureState::nonFiniteDetected);

        boolean runtimeResumeAllowed = runtimeResumeAllowedAfterManifestValidation();
        int runtimeEarlyStoppingPatience = this.earlyStoppingMonitorMetric == null
                ? this.earlyStoppingPatience
                : 0;
        CanonicalTrainerRuntime.Builder runtimeBuilder = Trainers.canonicalBuilder()
                .model(model)
                .optimizer(optimizer)
                .epochs(builder.epochs)
                .gradientClip(this.gradientClip)
                .mixedPrecision(this.mixedPrecision)
                .earlyStopping(runtimeEarlyStoppingPatience, this.earlyStoppingMinDelta)
                .resumeFromCheckpoint(runtimeResumeAllowed)
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
                        TrainerMetricRuntime.MetricSnapshot trainSnapshot =
                                metricRuntime.snapshotTrain(session, throughputStats.trainEpoch());
                        latestTrainMetrics = trainSnapshot.values();
                        latestTrainMetricDetails = trainSnapshot.details();
                        stepScheduler(SchedulerStepUnit.EPOCH);
                        recordEpochHistory(epoch, trainLoss);
                        persistCheckpointsSafely();
                        persistReportSafely(runtime.summary());
                    }

                    @Override
                    public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                        TrainerMetricRuntime.MetricSnapshot validationSnapshot =
                                metricRuntime.snapshotValidation(session, throughputStats.validationEpoch());
                        latestValidationMetrics = validationSnapshot.values();
                        latestValidationMetricDetails = validationSnapshot.details();
                        persistBestModelCheckpointIfImproved(epoch, valLoss);
                        updateCustomEarlyStoppingMonitor(session, epoch, valLoss);
                        stepScheduler(SchedulerStepUnit.VALIDATION, schedulerMonitorValue(valLoss));
                        updateEpochHistoryValidation(epoch, valLoss);
                        persistHistorySafely();
                        persistReportSafely(runtime.summary());
                    }

                    @Override
                    public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
                        if (!failureState.terminalFailureDetected()) {
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
            throwIfPostTrainingCheckpointFailed();
            throwIfNonFiniteDetected();
            throwIfInvalidMetricDetected();
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
            throwIfPostTrainingCheckpointFailed();
            throwIfNonFiniteDetected();
            throwIfInvalidMetricDetected();
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
        postTrainingCheckpointError = null;
        failureState.reset();
    }

    private void throwIfNonFiniteDetected() {
        failureState.throwIfNonFiniteDetected();
    }

    private void throwIfInvalidMetricDetected() {
        failureState.throwIfInvalidMetricDetected();
    }

    private void throwIfPostTrainingCheckpointFailed() {
        RuntimeException error = postTrainingCheckpointError;
        if (error != null) {
            throw error;
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
        ensureGradScalerCheckpointLoaded();
        ensureHistoryLoaded();
    }

    private boolean runtimeResumeAllowedAfterManifestValidation() {
        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                resumeFromCheckpoint,
                runtimeCheckpointFile,
                this::checkCheckpointArtifactIntegrity);
        runtimeCheckpointIntegrityMismatch = decision.integrityMismatch();
        runtimeCheckpointResumeSkipped = decision.resumeSkipped();
        runtimeCheckpointLoadError = decision.loadError();
        if (decision.shouldFail(failOnCheckpointLoadError)) {
            throw new IllegalStateException(
                    "Runtime checkpoint integrity mismatch for resume at "
                            + checkpointManifestFile + ": " + decision.loadError());
        }
        return decision.resumeAllowed();
    }

    private void ensureModelCheckpointLoaded() {
        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                resumeFromCheckpoint,
                modelCheckpointFile,
                modelCheckpointMetadataFile,
                modelCheckpointLoadAttempted,
                new TrainerModelCheckpointMetadata.ExpectedModel(
                        model.getClass().getName(),
                        modelParameterSignature(),
                        model.parameterCount()),
                MODEL_CHECKPOINT_METADATA_VERSION,
                failOnCheckpointLoadError,
                checkpointResumeDiagnostics,
                model::loadSafetensors);
        if (!result.stateChanged()) {
            return;
        }
        modelCheckpointMissingOnResume = result.missingOnResume();
        modelCheckpointLoaded = result.loaded();
        modelCheckpointCompatibilityMismatch = result.compatibilityMismatch();
        modelCheckpointMetadataLoaded = result.metadataLoaded();
        modelCheckpointMetadataMissingOnResume = result.metadataMissingOnResume();
        modelCheckpointMetadataLoadError = result.metadataLoadError();
        modelCheckpointLoadError = result.loadError();
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureOptimizerCheckpointLoaded() {
        if (!resumeFromCheckpoint || optimizerCheckpointFile == null || !optimizer.supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                optimizerCheckpointFile,
                optimizerCheckpointLoadAttempted,
                "optimizer",
                "optimizer",
                checkpointManifestFile,
                "Optimizer checkpoint integrity mismatch for resume",
                "optimizer",
                failOnCheckpointLoadError,
                this::checkCheckpointArtifactIntegrity,
                path -> {
                    TrainerStateCheckpointReader.OptimizerState checkpoint =
                            TrainerStateCheckpointReader.readOptimizer(
                                    path,
                                    optimizationRuntime.optimizerStepCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.stepCount());
                },
                state -> TrainerStateCheckpointMetadataValidator.requireOptimizer(
                        state,
                        optimizer.getClass().getName(),
                        TrainerMetadataSupport.parameterSignature(optimizer.parameters()),
                        checkpointResumeDiagnostics::recordCompatibilityMismatch),
                optimizer::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        optimizerCheckpointMissingOnResume = result.missingOnResume();
        optimizerCheckpointLoaded = result.loaded();
        optimizerCheckpointLoadError = result.loadError();
        if (result.loaded()) {
            optimizationRuntime.restoreOptimizerStepCount(result.counter());
        }
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureSchedulerCheckpointLoaded() {
        if (!resumeFromCheckpoint
                || schedulerCheckpointFile == null
                || !schedulerStepper.supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                schedulerCheckpointFile,
                schedulerCheckpointLoadAttempted,
                "scheduler",
                "scheduler",
                checkpointManifestFile,
                "Scheduler checkpoint integrity mismatch for resume",
                "scheduler",
                failOnCheckpointLoadError,
                this::checkCheckpointArtifactIntegrity,
                path -> {
                    TrainerStateCheckpointReader.SchedulerState checkpoint =
                            TrainerStateCheckpointReader.readScheduler(path, schedulerStepper.stepCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.stepCount());
                },
                state -> {
                    TrainerStateCheckpointMetadataValidator.requireScheduler(
                            state,
                            learningRateScheduler.getClass().getName(),
                            TrainerMetadataSupport.parameterSignature(optimizer.parameters()),
                            schedulerStepUnit.name(),
                            checkpointResumeDiagnostics::recordCompatibilityMismatch);
                },
                learningRateScheduler::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        schedulerCheckpointMissingOnResume = result.missingOnResume();
        schedulerCheckpointLoaded = result.loaded();
        schedulerCheckpointLoadError = result.loadError();
        if (result.loaded()) {
            schedulerStepper.restoreStepCount(result.counter());
        }
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureGradScalerCheckpointLoaded() {
        if (!resumeFromCheckpoint
                || gradScalerCheckpointFile == null
                || gradScaler == null
                || !gradScaler.supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                gradScalerCheckpointFile,
                gradScalerCheckpointLoadAttempted,
                "GradScaler",
                "gradScaler",
                checkpointManifestFile,
                "GradScaler checkpoint integrity mismatch for resume",
                "GradScaler",
                failOnCheckpointLoadError,
                this::checkCheckpointArtifactIntegrity,
                path -> {
                    TrainerStateCheckpointReader.GradScalerState checkpoint =
                            TrainerStateCheckpointReader.readGradScaler(
                                    path,
                                    optimizationRuntime.mixedPrecisionOverflowSkipCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.overflowSkipCount());
                },
                state -> {
                    TrainerStateCheckpointMetadataValidator.requireGradScaler(
                            state,
                            gradScaler.getClass().getName(),
                            checkpointResumeDiagnostics::recordCompatibilityMismatch);
                },
                gradScaler::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        gradScalerCheckpointMissingOnResume = result.missingOnResume();
        gradScalerCheckpointLoaded = result.loaded();
        gradScalerCheckpointLoadError = result.loadError();
        if (result.failure() != null) {
            throw result.failure();
        }
        if (result.loaded()) {
            optimizationRuntime.restoreMixedPrecisionState(
                    result.counter(),
                    gradScaler.getScale(),
                    gradScaler.overflowDetected());
            gradScalerCheckpointFallbackUsed = false;
        } else if (!result.missingOnResume() && result.loadError() != null) {
            gradScalerCheckpointFallbackUsed = true;
        }
    }

    private void ensureHistoryLoaded() {
        TrainerCheckpointLoadGate.Decision gate = TrainerCheckpointLoadGate.evaluate(
                resumeFromCheckpoint,
                historyFile,
                historyLoadAttempted,
                "history",
                false);
        if (gate.missing()) {
            historyMissingOnResume = true;
        }
        if (!gate.shouldLoad()) {
            return;
        }
        TrainerCheckpointIntegrityGate.Decision integrity = TrainerCheckpointIntegrityGate.evaluate(
                "history",
                historyFile,
                checkpointManifestFile,
                "Training history checkpoint integrity mismatch for resume",
                failOnCheckpointLoadError,
                this::checkCheckpointArtifactIntegrity);
        if (!integrity.compatible()) {
            historyLoadError = integrity.loadError();
            if (integrity.failure() != null) {
                throw integrity.failure();
            }
            return;
        }
        TrainerHistoryCheckpointReader.ReadResult result =
                TrainerHistoryCheckpointReader.readExisting(historyFile);
        if (result.loaded()) {
            epochHistory.replaceWith(result.rows());
            historyLoaded = true;
            historyMissingOnResume = false;
            historyLoadError = null;
        } else if (result.error() != null) {
            historyLoadError = result.error();
            if (failOnCheckpointLoadError) {
                throw new IllegalStateException(
                        "Failed to load training history checkpoint from " + historyFile,
                        result.cause());
            }
        }
    }

    private void persistCheckpointsSafely() {
        if (failureState.nonFiniteDetected()) {
            return;
        }
        persistModelCheckpointSafely();
        persistOptimizerCheckpointSafely();
        persistSchedulerCheckpointSafely();
        persistGradScalerCheckpointSafely();
        persistHistorySafely();
        persistCheckpointManifestSafely();
    }

    private void persistModelCheckpointSafely() {
        if (modelCheckpointFile == null) {
            return;
        }
        persistModelSafely(modelCheckpointFile, true);
    }

    private void persistBestModelCheckpointIfImproved(int epoch, double validationLoss) {
        TrainerBestModelCheckpointMonitor.Decision decision =
                bestModelCheckpointMonitor.evaluate(epoch, validationLoss, latestValidationMetrics);
        if (!decision.shouldSave()) {
            return;
        }
        if (persistModelSafely(bestModelCheckpointFile, false)) {
            bestModelCheckpointMonitor.markSaved(decision);
            bestModelCheckpointSaveError = null;
        }
    }

    private double bestModelMonitorValue(double validationLoss) {
        return TrainerMonitorSupport.valueOrThrow(
                bestModelMonitorMetric,
                validationLoss,
                latestValidationMetrics,
                "Best model monitor metric");
    }

    private void updateCustomEarlyStoppingMonitor(TrainerSession session, int epoch, double validationLoss) {
        TrainerEarlyStoppingMonitor.UpdateResult result =
                earlyStoppingMonitor.update(epoch, validationLoss, latestValidationMetrics);
        if (result.shouldStop()) {
            session.stop();
        }
    }

    private void restoreBestModelCheckpointAtEndSafely() {
        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                restoreBestModelAtEnd,
                bestModelCheckpointFile,
                checkpointManifestFile,
                failOnCheckpointLoadError,
                this::checkCheckpointArtifactIntegrity,
                model::loadSafetensors);
        bestModelCheckpointLoadError = result.loadError();
        if (result.restored()) {
            bestModelCheckpointMonitor.markRestored();
        }
        if (result.failure() != null) {
            postTrainingCheckpointError = result.failure();
            throw result.failure();
        }
    }

    private TrainerCheckpointCompatibilityReport checkCheckpointArtifactIntegrity(
            String artifactName,
            Path artifactFile) {
        TrainerCheckpointArtifactIntegrity.Result check = TrainerCheckpointArtifactIntegrity.check(
                checkpointManifestFile,
                artifactName,
                artifactFile,
                CHECKPOINT_MANIFEST_VERSION);
        checkpointManifestLoaded = check.manifestLoaded();
        checkpointManifestMissingOnResume = check.manifestMissing();
        checkpointManifestLoadError = check.manifestLoadError();
        if (check.integrityMismatch()) {
            checkpointManifestIntegrityMismatch = true;
            check.recordMismatch(checkpointResumeDiagnostics);
        }
        return check.report();
    }

    private boolean persistModelSafely(Path checkpointFile, boolean latestModelCheckpoint) {
        TrainerModelCheckpointWriter.WriteResult result =
                TrainerModelCheckpointWriter.writeModel(checkpointFile, model::saveSafetensors);
        if (!result.written()) {
            if (!latestModelCheckpoint) {
                bestModelCheckpointSaveError = result.error();
                return false;
            }
            modelCheckpointSaveError = result.error();
            return false;
        }
        if (latestModelCheckpoint) {
            modelCheckpointSaved = true;
            modelCheckpointSaveError = null;
            persistModelCheckpointMetadataSafely(checkpointFile);
        }
        return true;
    }

    private void persistModelCheckpointMetadataSafely(Path checkpointFile) {
        if (modelCheckpointMetadataFile == null) {
            return;
        }
        TrainerModelCheckpointWriter.WriteResult result = TrainerModelCheckpointWriter.writeMetadata(
                modelCheckpointMetadataFile,
                checkpointFile,
                new TrainerModelCheckpointMetadata.ExpectedModel(
                        model.getClass().getName(),
                        modelParameterSignature(),
                        model.parameterCount()),
                MODEL_CHECKPOINT_METADATA_VERSION);
        if (result.written()) {
            modelCheckpointMetadataSaved = true;
            modelCheckpointMetadataSaveError = null;
        } else if (result.error() != null) {
            modelCheckpointMetadataSaveError = result.error();
        }
    }

    private void persistOptimizerCheckpointSafely() {
        TrainerStateCheckpointPersistence.Result result = TrainerStateCheckpointPersistence.persistOptimizer(
                optimizerCheckpointFile,
                optimizer,
                optimizationRuntime.optimizerStepCount());
        if (result.saved()) {
            optimizerCheckpointSaved = true;
            optimizerCheckpointSaveError = null;
        } else if (result.saveError() != null) {
            optimizerCheckpointSaveError = result.saveError();
        }
    }

    private void persistSchedulerCheckpointSafely() {
        TrainerStateCheckpointPersistence.Result result = TrainerStateCheckpointPersistence.persistScheduler(
                schedulerCheckpointFile,
                learningRateScheduler,
                schedulerStepUnit.name(),
                schedulerStepper.stepCount(),
                optimizer);
        if (result.saved()) {
            schedulerCheckpointSaved = true;
            schedulerCheckpointSaveError = null;
        } else if (result.saveError() != null) {
            schedulerCheckpointSaveError = result.saveError();
        }
    }

    private void persistGradScalerCheckpointSafely() {
        TrainerStateCheckpointPersistence.Result result = TrainerStateCheckpointPersistence.persistGradScaler(
                gradScalerCheckpointFile,
                gradScaler,
                optimizationRuntime.mixedPrecisionOverflowSkipCount());
        if (result.saved()) {
            gradScalerCheckpointSaved = true;
            gradScalerCheckpointSaveError = null;
        } else if (result.saveError() != null) {
            gradScalerCheckpointSaveError = result.saveError();
        }
    }

    private void persistHistorySafely() {
        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistHistory(
                        historyFile,
                        epochHistory.snapshot(),
                        checkpointManifestRequest());
        if (result.artifact().saved()) {
            historySaved = true;
            historySaveError = null;
        } else if (result.artifact().error() != null) {
            historySaveError = result.artifact().error();
        }
        applyCheckpointManifestSaveResult(result.manifest());
    }

    private void persistReportSafely(TrainingSummary baseSummary) {
        if (baseSummary == null) {
            return;
        }
        TrainingSummary summary = enrichSummary(baseSummary);
        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistReport(
                        reportFile,
                        summary,
                        checkpointManifestRequest());
        if (result.artifact().saved()) {
            reportSaved = true;
            reportSaveError = null;
        } else if (result.artifact().error() != null) {
            reportSaveError = result.artifact().error();
        }
        applyCheckpointManifestSaveResult(result.manifest());
    }

    private void persistCheckpointManifestSafely() {
        applyCheckpointManifestSaveResult(
                TrainerRuntimeArtifactPersistence.persistManifest(checkpointManifestRequest()));
    }

    private TrainerRuntimeArtifactPersistence.ManifestRequest checkpointManifestRequest() {
        return new TrainerRuntimeArtifactPersistence.ManifestRequest(
                checkpointManifestFile,
                checkpointLayout.manifestArtifacts(),
                CHECKPOINT_MANIFEST_VERSION);
    }

    private void applyCheckpointManifestSaveResult(TrainerRuntimeArtifactPersistence.SaveResult result) {
        if (result.saved()) {
            checkpointManifestSaved = true;
            checkpointManifestSaveError = null;
        } else if (result.error() != null) {
            checkpointManifestSaveError = result.error();
        }
    }

    private TrainingSummary enrichSummary(TrainingSummary base) {
        return TrainerSummaryAssembler.enrich(base, summaryRequest(base));
    }

    private TrainerSummaryAssembler.Request summaryRequest(TrainingSummary base) {
        return new TrainerSummaryAssembler.Request(
                new TrainerSummaryAssembler.Resume(
                        resumeFromCheckpoint,
                        checkpointResumeMissingArtifacts(),
                        checkpointResumeDiagnostics.compatibilityMismatches()),
                new TrainerSummaryAssembler.ModelCheckpoint(
                        modelCheckpointFile,
                        modelCheckpointMissingOnResume,
                        modelCheckpointLoaded,
                        modelCheckpointSaved,
                        modelCheckpointLoadError,
                        modelCheckpointSaveError,
                        modelCheckpointCompatibilityMismatch),
                new TrainerSummaryAssembler.Artifact(
                        modelCheckpointMetadataFile,
                        modelCheckpointMetadataMissingOnResume,
                        modelCheckpointMetadataLoaded,
                        modelCheckpointMetadataSaved,
                        modelCheckpointMetadataLoadError,
                        modelCheckpointMetadataSaveError),
                new TrainerSummaryAssembler.BestModel(
                        saveBestModelCheckpoint && bestModelCheckpointFile != null,
                        restoreBestModelAtEnd,
                        bestModelCheckpointFile,
                        bestModelCheckpointMonitorLabel(),
                        bestModelMonitorMode,
                        bestModelCheckpointMonitor.state(),
                        bestModelCheckpointSaveError,
                        bestModelCheckpointLoadError),
                new TrainerSummaryAssembler.StateCheckpoint(
                        optimizerCheckpointFile != null,
                        optimizer.supportsStateDict(),
                        optimizerCheckpointFile,
                        optimizerCheckpointMissingOnResume,
                        optimizerCheckpointLoaded,
                        optimizerCheckpointSaved,
                        optimizerCheckpointLoadError,
                        optimizerCheckpointSaveError),
                new TrainerSummaryAssembler.MixedPrecision(
                        mixedPrecision,
                        optimizationRuntime.latestMixedPrecisionLossScale(),
                        optimizationRuntime.latestMixedPrecisionOverflowDetected(),
                        optimizationRuntime.mixedPrecisionOverflowSkipCount(),
                        optimizationRuntime.gradScalerStateSnapshot(),
                        new TrainerMixedPrecisionMetadata.GradScalerCheckpoint(
                                gradScalerCheckpointFile != null && gradScaler != null,
                                gradScaler != null && gradScaler.supportsStateDict(),
                                resumeFromCheckpoint,
                                gradScalerCheckpointFile,
                                gradScalerCheckpointMissingOnResume,
                                gradScalerCheckpointLoaded,
                                gradScalerCheckpointSaved,
                                gradScalerCheckpointLoadError,
                                gradScalerCheckpointSaveError,
                                gradScalerCheckpointFallbackUsed)),
                new TrainerSummaryAssembler.Scheduler(
                        schedulerStepper.enabled(),
                        schedulerStepUnit.name(),
                        schedulerStepper.stepCount(),
                        schedulerStepper.schedulerType(),
                        schedulerMonitorLabel(),
                        schedulerMonitorMetric != null,
                        schedulerStepper.stateSnapshot(),
                        optimizer.learningRate(),
                        new TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint(
                                schedulerCheckpointFile != null,
                                schedulerStepper.supportsStateDict(),
                                resumeFromCheckpoint,
                                schedulerCheckpointFile,
                                schedulerCheckpointMissingOnResume,
                                schedulerCheckpointLoaded,
                                schedulerCheckpointSaved,
                                schedulerCheckpointLoadError,
                                schedulerCheckpointSaveError)),
                new TrainerSummaryAssembler.RuntimeArtifacts(
                        new TrainerRuntimeArtifactMetadata.Artifact(
                                historyFile != null,
                                historyFile,
                                historyMissingOnResume,
                                historyLoaded,
                                historySaved,
                                historyLoadError,
                                historySaveError),
                        new TrainerRuntimeArtifactMetadata.SaveOnlyArtifact(
                                reportFile != null,
                                reportFile,
                                reportSaved,
                                reportSaveError),
                        new TrainerRuntimeArtifactMetadata.Artifact(
                                checkpointManifestFile != null,
                                checkpointManifestFile,
                                checkpointManifestMissingOnResume,
                                checkpointManifestLoaded,
                                checkpointManifestSaved,
                                checkpointManifestLoadError,
                                checkpointManifestSaveError),
                        checkpointManifestIntegrityMismatch,
                        new TrainerRuntimeArtifactMetadata.RuntimeCheckpoint(
                                runtimeCheckpointFile,
                                runtimeCheckpointIntegrityMismatch,
                                runtimeCheckpointResumeSkipped,
                                runtimeCheckpointLoadError)),
                new TrainerSummaryAssembler.EarlyStopping(
                        earlyStoppingPatience,
                        earlyStoppingMinDelta,
                        earlyStoppingMonitorLabel(),
                        earlyStoppingMonitorMode,
                        earlyStoppingMonitorMetric != null,
                        earlyStoppingMonitorState(base),
                        earlyStoppingMonitor.triggered(),
                        earlyStoppingMonitor.stopEpoch()),
                failureState,
                new TrainerSummaryAssembler.Metrics(
                        !trainMetrics.isEmpty(),
                        latestTrainMetrics,
                        latestValidationMetrics,
                        latestTrainMetricDetails,
                        latestValidationMetricDetails),
                new TrainerSummaryAssembler.History(epochHistory.snapshot(), epochHistory.size()),
                new TrainerSummaryAssembler.Optimization(
                        optimizationRuntime.gradientAccumulationSteps(),
                        optimizationRuntime.pendingGradientAccumulationBatches(),
                        optimizationRuntime.optimizerStepCount(),
                        gradientClip,
                        optimizationRuntime.latestGradientDiagnostics(),
                        optimizationRuntime.latestParameterDiagnostics()),
                new TrainerSummaryAssembler.Throughput(
                        throughputStats.trainTotal(),
                        throughputStats.validationTotal()),
                new TrainerSummaryAssembler.AccelerationInfo(
                        preferredDevice,
                        Acceleration.status(preferredDevice),
                        accelerationStatusAtStart),
                new TrainerSummaryAssembler.References(
                        new TrainerSummaryReferenceMetadata.Paths(
                                modelCheckpointFile,
                                modelCheckpointMetadataFile,
                                bestModelCheckpointFile,
                                optimizerCheckpointFile,
                                schedulerCheckpointFile,
                                gradScalerCheckpointFile,
                                gradScaler != null,
                                historyFile,
                                reportFile,
                                runtimeCheckpointFile,
                                checkpointManifestFile),
                        new TrainerSummaryReferenceMetadata.Errors(
                                modelCheckpointLoadError,
                                modelCheckpointSaveError,
                                modelCheckpointMetadataLoadError,
                                modelCheckpointMetadataSaveError,
                                bestModelCheckpointSaveError,
                                bestModelCheckpointLoadError,
                                optimizerCheckpointLoadError,
                                optimizerCheckpointSaveError,
                                schedulerCheckpointLoadError,
                                schedulerCheckpointSaveError,
                                gradScalerCheckpointLoadError,
                                gradScalerCheckpointSaveError,
                                historyLoadError,
                                historySaveError,
                                reportSaveError,
                                checkpointManifestLoadError,
                                checkpointManifestSaveError,
                                runtimeCheckpointLoadError)));
    }

    private String bestModelCheckpointMonitorLabel() {
        return TrainerMonitorSupport.label(bestModelMonitorMetric);
    }

    private String earlyStoppingMonitorLabel() {
        return TrainerMonitorSupport.label(earlyStoppingMonitorMetric);
    }

    private String schedulerMonitorLabel() {
        return TrainerMonitorSupport.label(schedulerMonitorMetric);
    }

    private double schedulerMonitorValue(double validationLoss) {
        return TrainerMonitorSupport.valueOrNaN(schedulerMonitorMetric, validationLoss, latestValidationMetrics);
    }

    private TrainerEarlyStoppingMetadata.MonitorState earlyStoppingMonitorState(TrainingSummary base) {
        if (earlyStoppingMonitorMetric != null) {
            return earlyStoppingMonitor.state();
        }
        return new TrainerEarlyStoppingMetadata.MonitorState(
                base.bestValidationEpoch(),
                base.bestValidationLoss(),
                base.latestValidationLoss() == null ? Double.NaN : base.latestValidationLoss(),
                TrainerMonitorSupport.earlyStoppingEpochsWithoutImprovement(null, base, 0));
    }

    private String modelParameterSignature() {
        return TrainerMetadataSupport.parameterSignature(model.namedParameters());
    }

    private List<String> checkpointResumeMissingArtifacts() {
        return TrainerCheckpointResumeDiagnostics.missingArtifacts(
                modelCheckpointMissingOnResume,
                optimizerCheckpointMissingOnResume,
                schedulerCheckpointMissingOnResume,
                gradScalerCheckpointMissingOnResume,
                historyMissingOnResume);
    }

    private void resetMetrics() {
        metricRuntime.reset();
        latestTrainMetrics = Map.of();
        latestValidationMetrics = Map.of();
        latestTrainMetricDetails = Map.of();
        latestValidationMetricDetails = Map.of();
    }

    private void resetEpochThroughputCounters() {
        throughputStats.resetEpoch();
    }

    private void recordEpochHistory(int epoch, double trainLoss) {
        epochHistory.recordTrain(TrainerEpochHistoryRecordFactory.train(
                epoch,
                trainLoss,
                optimizer.learningRate(),
                optimizationRuntime.optimizerStepCount(),
                schedulerStepper.stepCount(),
                optimizationRuntime.latestGradientDiagnostics(),
                optimizationRuntime.latestParameterDiagnostics(),
                optimizationRuntime.latestMixedPrecisionDiagnostics(),
                throughputStats.trainEpoch(),
                latestTrainMetrics,
                latestTrainMetricDetails));
    }

    private void updateEpochHistoryValidation(int epoch, double validationLoss) {
        double monitorValue = bestModelMonitorValue(validationLoss);
        epochHistory.recordValidation(TrainerEpochHistoryRecordFactory.validation(
                epoch,
                validationLoss,
                optimizer.learningRate(),
                schedulerStepper.stepCount(),
                throughputStats.validationEpoch(),
                latestValidationMetrics,
                latestValidationMetricDetails,
                monitorValue,
                bestModelCheckpointMonitorLabel(),
                bestModelMonitorMode.name()));
    }

    private double runTrainingBatch(Object rawBatch) {
        double loss = batchRuntime.train(rawBatch, optimizationRuntime.shouldZeroGradBeforeBackward());
        optimizationRuntime.afterBackwardBatch();
        return loss;
    }

    private void flushGradientAccumulation() {
        flushGradientAccumulation(null);
    }

    private void flushGradientAccumulation(TrainerSession session) {
        if (optimizationRuntime.hasPendingGradients()) {
            try {
                optimizationRuntime.flush();
            } catch (RuntimeException error) {
                if (failureState.nonFiniteDetected() && session != null) {
                    session.stop();
                    return;
                }
                throw error;
            }
        }
    }

    private void stepScheduler(SchedulerStepUnit unit) {
        schedulerStepper.step(unit);
    }

    private void stepBatchScheduler() {
        stepScheduler(SchedulerStepUnit.BATCH);
    }

    private void stepScheduler(SchedulerStepUnit unit, double monitorValue) {
        schedulerStepper.step(unit, monitorValue);
    }

    private double runValidationBatch(Object rawBatch) {
        return batchRuntime.validation(rawBatch);
    }

    private void discardPendingGradients() {
        if (optimizationRuntime == null) {
            optimizer.zeroGrad();
            return;
        }
        optimizationRuntime.discardPendingGradients();
    }

    /**
     * @deprecated Use {@link TrainingLossFunction}. Kept as a
     * source-compatible alias for existing code that references
     * {@code CanonicalTrainer.LossFunction}.
     */
    @Deprecated(since = "0.1.0", forRemoval = false)
    @FunctionalInterface
    public interface LossFunction extends TrainingLossFunction {
    }

    /**
     * @deprecated Use {@link TrainingMetric}. Kept as a source-compatible alias
     * for existing code that references {@code CanonicalTrainer.Metric}.
     */
    @Deprecated(since = "0.1.0", forRemoval = false)
    public interface Metric extends TrainingMetric {
    }

    /**
     * @deprecated Use {@link DetailedTrainingMetric}. Kept as a
     * source-compatible alias for existing code that references
     * {@code CanonicalTrainer.DetailedMetric}.
     */
    @Deprecated(since = "0.1.0", forRemoval = false)
    public interface DetailedMetric extends Metric, DetailedTrainingMetric {
    }

    public static final class Metrics extends CanonicalTrainerMetricFactories {
        private Metrics() {
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

    public static final class Builder {
        private NNModule model;
        private Optimizer optimizer;
        private LRScheduler learningRateScheduler;
        private SchedulerStepUnit schedulerStepUnit = SchedulerStepUnit.BATCH;
        private String schedulerMonitorMetric;
        private TrainingLossFunction lossFunction;
        private final List<Supplier<? extends TrainingMetric>> metricFactories = new ArrayList<>();
        private int epochs = 1;
        private double gradientClip = 0.0;
        private int gradientAccumulationSteps = 1;
        private boolean mixedPrecision = false;
        private GradScaler gradScaler;
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
            this.schedulerMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                    metricName,
                    "scheduler monitor metric");
            return this;
        }

        public Builder schedulerMonitorValidationLoss() {
            this.schedulerMonitorMetric = null;
            return this;
        }

        public Builder loss(TrainingLossFunction lossFunction) {
            return lossFunction(lossFunction);
        }

        public Builder lossFunction(TrainingLossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder metric(Supplier<? extends TrainingMetric> metricFactory) {
            this.metricFactories.add(Objects.requireNonNull(metricFactory, "metric factory must not be null"));
            return this;
        }

        public Builder metrics(List<? extends Supplier<? extends TrainingMetric>> metricFactories) {
            if (metricFactories == null) {
                return this;
            }
            for (Supplier<? extends TrainingMetric> metricFactory : metricFactories) {
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
            if (!mixedPrecision) {
                this.gradScaler = null;
            }
            return this;
        }

        public Builder gradScaler(GradScaler gradScaler) {
            this.gradScaler = Objects.requireNonNull(gradScaler, "gradScaler must not be null");
            this.mixedPrecision = true;
            return this;
        }

        public Builder mixedPrecision(GradScaler gradScaler) {
            return gradScaler(gradScaler);
        }

        public Builder device(String deviceId) {
            this.preferredDevice = TrainerMetadataSupport.normalizeDevice(deviceId);
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
            this.bestModelMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                    metricName,
                    "best model monitor metric");
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
            this.earlyStoppingMonitorMetric = TrainerMonitorSupport.normalizeMetric(
                    metricName,
                    "early stopping monitor metric");
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
