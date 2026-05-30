package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Objects;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;

/**
 * Coordinates checkpoint resume across model, runtime, optimizer, scheduler,
 * GradScaler, and history artifacts.
 */
final class TrainerCheckpointResumeCoordinator {
    private final Request request;

    TrainerCheckpointResumeCoordinator(Request request) {
        this.request = Objects.requireNonNull(request, "request must not be null");
    }

    boolean runtimeResumeAllowedAfterManifestValidation() {
        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                request.resumeFromCheckpoint(),
                request.runtimeCheckpointFile(),
                (artifact, file) -> request.integrityChecker().check(artifact, file));
        request.runtimeArtifactState().recordRuntimeResumeDecision(decision);
        if (decision.shouldFail(request.failOnCheckpointLoadError())) {
            throw new IllegalStateException(
                    "Runtime checkpoint integrity mismatch for resume at "
                            + request.checkpointManifestFile() + ": " + decision.loadError());
        }
        return decision.resumeAllowed();
    }

    void ensureCheckpointStateLoaded() {
        ensureModelCheckpointLoaded();
        ensureOptimizerCheckpointLoaded();
        ensureSchedulerCheckpointLoaded();
        ensureGradScalerCheckpointLoaded();
        ensureHistoryLoaded();
    }

    private void ensureModelCheckpointLoaded() {
        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                request.resumeFromCheckpoint(),
                request.modelCheckpointFile(),
                request.modelCheckpointMetadataFile(),
                request.modelCheckpointStatus().loadAttempted(),
                new TrainerModelCheckpointMetadata.ExpectedModel(
                        request.model().getClass().getName(),
                        TrainerMetadataSupport.parameterSignature(request.model().namedParameters()),
                        request.model().parameterCount()),
                request.modelCheckpointMetadataVersion(),
                request.failOnCheckpointLoadError(),
                request.checkpointResumeDiagnostics(),
                request.model()::loadSafetensors);
        if (!result.stateChanged()) {
            return;
        }
        request.modelCheckpointStatus().recordResume(result);
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureOptimizerCheckpointLoaded() {
        if (!request.resumeFromCheckpoint()
                || request.optimizerCheckpointFile() == null
                || !request.optimizer().supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                request.optimizerCheckpointFile(),
                request.stateCheckpointStatus().optimizerLoadAttempted(),
                "optimizer",
                "optimizer",
                request.checkpointManifestFile(),
                "Optimizer checkpoint integrity mismatch for resume",
                "optimizer",
                request.failOnCheckpointLoadError(),
                request.integrityChecker(),
                path -> {
                    TrainerStateCheckpointReader.OptimizerState checkpoint =
                            TrainerStateCheckpointReader.readOptimizer(
                                    path,
                                    request.optimizationRuntime().optimizerStepCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.stepCount());
                },
                state -> TrainerStateCheckpointMetadataValidator.requireOptimizer(
                        state,
                        request.optimizer().getClass().getName(),
                        TrainerMetadataSupport.parameterSignature(request.optimizer().parameters()),
                        request.checkpointResumeDiagnostics()::recordCompatibilityMismatch),
                request.optimizer()::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        request.stateCheckpointStatus().recordOptimizerResume(result);
        if (result.loaded()) {
            request.optimizationRuntime().restoreOptimizerStepCount(result.counter());
        }
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureSchedulerCheckpointLoaded() {
        if (!request.resumeFromCheckpoint()
                || request.schedulerCheckpointFile() == null
                || !request.schedulerStepper().supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                request.schedulerCheckpointFile(),
                request.stateCheckpointStatus().schedulerLoadAttempted(),
                "scheduler",
                "scheduler",
                request.checkpointManifestFile(),
                "Scheduler checkpoint integrity mismatch for resume",
                "scheduler",
                request.failOnCheckpointLoadError(),
                request.integrityChecker(),
                path -> {
                    TrainerStateCheckpointReader.SchedulerState checkpoint =
                            TrainerStateCheckpointReader.readScheduler(
                                    path,
                                    request.schedulerStepper().stepCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.stepCount());
                },
                state -> {
                    TrainerStateCheckpointMetadataValidator.requireScheduler(
                            state,
                            request.learningRateScheduler().getClass().getName(),
                            TrainerMetadataSupport.parameterSignature(request.optimizer().parameters()),
                            request.schedulerStepUnit().name(),
                            request.checkpointResumeDiagnostics()::recordCompatibilityMismatch);
                },
                request.learningRateScheduler()::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        request.stateCheckpointStatus().recordSchedulerResume(result);
        if (result.loaded()) {
            request.schedulerStepper().restoreStepCount(result.counter());
        }
        if (result.failure() != null) {
            throw result.failure();
        }
    }

    private void ensureGradScalerCheckpointLoaded() {
        if (!request.resumeFromCheckpoint()
                || request.gradScalerCheckpointFile() == null
                || request.gradScaler() == null
                || !request.gradScaler().supportsStateDict()) {
            return;
        }
        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                request.gradScalerCheckpointFile(),
                request.stateCheckpointStatus().gradScalerLoadAttempted(),
                "GradScaler",
                "gradScaler",
                request.checkpointManifestFile(),
                "GradScaler checkpoint integrity mismatch for resume",
                "GradScaler",
                request.failOnCheckpointLoadError(),
                request.integrityChecker(),
                path -> {
                    TrainerStateCheckpointReader.GradScalerState checkpoint =
                            TrainerStateCheckpointReader.readGradScaler(
                                    path,
                                    request.optimizationRuntime().mixedPrecisionOverflowSkipCount());
                    return new TrainerStateCheckpointResume.StateSnapshot(
                            checkpoint.state(),
                            checkpoint.overflowSkipCount());
                },
                state -> {
                    TrainerStateCheckpointMetadataValidator.requireGradScaler(
                            state,
                            request.gradScaler().getClass().getName(),
                            request.checkpointResumeDiagnostics()::recordCompatibilityMismatch);
                },
                request.gradScaler()::loadStateDict);
        if (!result.stateChanged()) {
            return;
        }
        request.stateCheckpointStatus().recordGradScalerResume(result);
        if (result.failure() != null) {
            throw result.failure();
        }
        if (result.loaded()) {
            request.optimizationRuntime().restoreMixedPrecisionState(
                    result.counter(),
                    request.gradScaler().getScale(),
                    request.gradScaler().overflowDetected());
        }
    }

    private void ensureHistoryLoaded() {
        TrainerCheckpointLoadGate.Decision gate = TrainerCheckpointLoadGate.evaluate(
                request.resumeFromCheckpoint(),
                request.historyFile(),
                request.runtimeArtifactState().historyLoadAttempted(),
                "history",
                false);
        if (gate.missing()) {
            request.runtimeArtifactState().recordHistoryMissingOnResume();
        }
        if (!gate.shouldLoad()) {
            return;
        }
        TrainerCheckpointIntegrityGate.Decision integrity = TrainerCheckpointIntegrityGate.evaluate(
                "history",
                request.historyFile(),
                request.checkpointManifestFile(),
                "Training history checkpoint integrity mismatch for resume",
                request.failOnCheckpointLoadError(),
                request.integrityChecker());
        if (!integrity.compatible()) {
            request.runtimeArtifactState().recordHistoryLoadError(integrity.loadError());
            if (integrity.failure() != null) {
                throw integrity.failure();
            }
            return;
        }
        TrainerHistoryCheckpointReader.ReadResult result =
                TrainerHistoryCheckpointReader.readExisting(request.historyFile());
        if (result.loaded()) {
            request.epochHistory().replaceWith(result.rows());
            request.runtimeArtifactState().recordHistoryLoaded();
        } else if (result.error() != null) {
            request.runtimeArtifactState().recordHistoryLoadError(result.error());
            if (request.failOnCheckpointLoadError()) {
                throw new IllegalStateException(
                        "Failed to load training history checkpoint from " + request.historyFile(),
                        result.cause());
            }
        }
    }

    record Request(
            boolean resumeFromCheckpoint,
            boolean failOnCheckpointLoadError,
            Path checkpointManifestFile,
            Path runtimeCheckpointFile,
            Path modelCheckpointFile,
            Path modelCheckpointMetadataFile,
            Path optimizerCheckpointFile,
            Path schedulerCheckpointFile,
            Path gradScalerCheckpointFile,
            Path historyFile,
            NNModule model,
            Optimizer optimizer,
            LRScheduler learningRateScheduler,
            CanonicalTrainer.SchedulerStepUnit schedulerStepUnit,
            TrainerLearningRateSchedulerStepper schedulerStepper,
            GradScaler gradScaler,
            TrainerOptimizationRuntime optimizationRuntime,
            TrainerCheckpointResumeDiagnostics checkpointResumeDiagnostics,
            TrainerRuntimeArtifactState runtimeArtifactState,
            TrainerModelCheckpointStatus modelCheckpointStatus,
            TrainerStateCheckpointStatus stateCheckpointStatus,
            TrainerEpochHistory epochHistory,
            TrainerCheckpointIntegrityGate.IntegrityChecker integrityChecker,
            int modelCheckpointMetadataVersion) {
        Request {
            Objects.requireNonNull(model, "model must not be null");
            Objects.requireNonNull(optimizer, "optimizer must not be null");
            Objects.requireNonNull(schedulerStepUnit, "schedulerStepUnit must not be null");
            Objects.requireNonNull(schedulerStepper, "schedulerStepper must not be null");
            Objects.requireNonNull(optimizationRuntime, "optimizationRuntime must not be null");
            Objects.requireNonNull(checkpointResumeDiagnostics, "checkpointResumeDiagnostics must not be null");
            Objects.requireNonNull(runtimeArtifactState, "runtimeArtifactState must not be null");
            Objects.requireNonNull(modelCheckpointStatus, "modelCheckpointStatus must not be null");
            Objects.requireNonNull(stateCheckpointStatus, "stateCheckpointStatus must not be null");
            Objects.requireNonNull(epochHistory, "epochHistory must not be null");
            Objects.requireNonNull(integrityChecker, "integrityChecker must not be null");
        }
    }
}
