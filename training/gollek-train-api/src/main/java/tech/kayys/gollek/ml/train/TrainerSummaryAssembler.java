package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Converts trainer runtime snapshots into the immutable public summary metadata.
 */
final class TrainerSummaryAssembler {
    private TrainerSummaryAssembler() {
    }

    static TrainingSummary enrich(TrainingSummary base, Request request) {
        Map<String, Object> metadata = new HashMap<>(base.metadata());
        TrainerSummaryMetadata.putCheckpointResumeOverview(
                metadata,
                request.resume().resumeRequested(),
                request.resume().missingArtifacts(),
                request.resume().compatibilityMismatches(),
                request.resume().manifestEntryMissingArtifacts());
        TrainerSummaryMetadata.putCheckpointStatus(
                metadata,
                "modelCheckpoint",
                request.model().file() != null,
                request.resume().resumeRequested(),
                request.model().file(),
                request.model().missingOnResume(),
                request.model().loaded(),
                request.model().saved(),
                request.model().loadError(),
                request.model().saveError());
        metadata.put("modelCheckpointCompatibilityMismatch", request.model().compatibilityMismatch());
        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "modelCheckpointMetadata",
                request.modelMetadata().file() != null,
                request.modelMetadata().file(),
                request.modelMetadata().missingOnResume(),
                request.modelMetadata().loaded(),
                request.modelMetadata().saved(),
                request.modelMetadata().loadError(),
                request.modelMetadata().saveError());
        TrainerBestModelCheckpointMetadata.put(
                metadata,
                request.bestModel().enabled(),
                request.bestModel().restoreRequested(),
                request.bestModel().file(),
                request.bestModel().monitorLabel(),
                request.bestModel().monitorMode(),
                request.bestModel().state(),
                request.bestModel().saveError(),
                request.bestModel().loadError());
        TrainerSummaryMetadata.putSupportedCheckpointStatus(
                metadata,
                "optimizerCheckpoint",
                request.optimizer().enabled(),
                request.optimizer().supported(),
                request.resume().resumeRequested(),
                request.optimizer().file(),
                request.optimizer().missingOnResume(),
                request.optimizer().loaded(),
                request.optimizer().saved(),
                request.optimizer().loadError(),
                request.optimizer().saveError());
        TrainerMixedPrecisionMetadata.put(
                metadata,
                request.mixedPrecision().enabled(),
                request.mixedPrecision().lossScale(),
                request.mixedPrecision().overflowDetected(),
                request.mixedPrecision().overflowSkipCount(),
                request.mixedPrecision().gradScalerState(),
                request.mixedPrecision().checkpoint());
        TrainerLearningRateSchedulerMetadata.put(
                metadata,
                request.scheduler().enabled(),
                request.scheduler().stepUnit(),
                request.scheduler().stepCount(),
                request.scheduler().schedulerType(),
                request.scheduler().monitorLabel(),
                request.scheduler().metricDriven(),
                request.scheduler().state(),
                request.scheduler().learningRate(),
                request.scheduler().checkpoint());
        TrainerRuntimeArtifactMetadata.put(
                metadata,
                request.artifacts().history(),
                request.artifacts().report(),
                request.artifacts().manifest(),
                request.artifacts().manifestIntegrityMismatch(),
                request.artifacts().runtimeCheckpoint());
        TrainerEarlyStoppingMetadata.put(
                metadata,
                request.earlyStopping().patience(),
                request.earlyStopping().minDelta(),
                request.earlyStopping().monitorLabel(),
                request.earlyStopping().monitorMode(),
                request.earlyStopping().metricDriven(),
                request.earlyStopping().state(),
                request.earlyStopping().triggered(),
                request.earlyStopping().stopEpoch());
        request.failureState().putMetadata(metadata);
        TrainerMetricsMetadata.putLatest(
                metadata,
                request.metrics().enabled(),
                request.metrics().train(),
                request.metrics().validation(),
                request.metrics().trainDetails(),
                request.metrics().validationDetails());
        metadata.putAll(request.dataLoaderPlans().metadata());
        TrainerDataLoaderPlanHealthMetadata.put(metadata);
        metadata.putAll(request.dataDistribution().metadata());
        TrainerDataDistributionHealthMetadata.put(metadata);
        TrainerGeneralizationMetadata.putLatest(metadata, base.latestTrainLoss(), base.latestValidationLoss());
        TrainerGeneralizationMetadata.putLatestTrend(metadata, request.history().rows());
        TrainerLossTrendMetadata.putLatest(metadata, request.history().rows());
        TrainerLossSlopeMetadata.putLatest(metadata, request.history().rows());
        TrainerLossWindowStatsMetadata.putLatest(metadata, request.history().rows());
        TrainerLossImprovementMetadata.putLatest(metadata, request.history().rows());
        TrainerValidationProgressMetadata.put(
                metadata,
                base.epochCount(),
                base.bestValidationLoss(),
                base.bestValidationEpoch(),
                base.latestValidationLoss());
        metadata.put("epochHistory", request.history().rows());
        metadata.put("epochHistorySize", request.history().size());
        TrainerOptimizationMetadata.put(
                metadata,
                request.optimization().gradientAccumulationSteps(),
                request.optimization().pendingGradientAccumulationBatches(),
                request.optimization().optimizerStepCount(),
                request.optimization().gradientClip(),
                request.optimization().parameterUpdateDiagnosticsIntervalSteps(),
                request.optimization().gradients(),
                request.optimization().parameters(),
                request.optimization().updates());
        TrainerThroughputStats.putPhaseMetadata(metadata, "train", request.throughput().trainTotal());
        TrainerThroughputStats.putPhaseMetadata(metadata, "validation", request.throughput().validationTotal());
        metadata.putAll(request.runtimeProfile().metadata());
        TrainerAccelerationMetadata.put(
                metadata,
                request.acceleration().requestedDevice(),
                request.acceleration().currentStatus(),
                request.acceleration().startStatus());
        TrainerSummaryReferenceMetadata.put(
                metadata,
                request.references().paths(),
                request.references().errors());
        TrainerRunHealthMetadata.put(metadata);
        return new TrainingSummary(
                base.epochCount(),
                base.bestValidationLoss(),
                base.bestValidationEpoch(),
                base.latestTrainLoss(),
                base.latestValidationLoss(),
                base.durationMs(),
                Map.copyOf(metadata));
    }

    record Request(
            Resume resume,
            ModelCheckpoint model,
            Artifact modelMetadata,
            BestModel bestModel,
            StateCheckpoint optimizer,
            MixedPrecision mixedPrecision,
            Scheduler scheduler,
            RuntimeArtifacts artifacts,
            EarlyStopping earlyStopping,
            TrainerFailureState failureState,
            Metrics metrics,
            DataLoaderPlans dataLoaderPlans,
            DataDistribution dataDistribution,
            History history,
            Optimization optimization,
            Throughput throughput,
            RuntimeProfile runtimeProfile,
            AccelerationInfo acceleration,
            References references) {
    }

    record Resume(
            boolean resumeRequested,
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
    }

    record ModelCheckpoint(
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError,
            boolean compatibilityMismatch) {
    }

    record Artifact(
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
    }

    record BestModel(
            boolean enabled,
            boolean restoreRequested,
            Path file,
            String monitorLabel,
            CanonicalTrainer.BestModelMonitorMode monitorMode,
            TrainerBestModelCheckpointMetadata.State state,
            String saveError,
            String loadError) {
    }

    record StateCheckpoint(
            boolean enabled,
            boolean supported,
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
    }

    record MixedPrecision(
            boolean enabled,
            double lossScale,
            boolean overflowDetected,
            int overflowSkipCount,
            Map<String, Object> gradScalerState,
            TrainerMixedPrecisionMetadata.GradScalerCheckpoint checkpoint) {
    }

    record Scheduler(
            boolean enabled,
            String stepUnit,
            int stepCount,
            String schedulerType,
            String monitorLabel,
            boolean metricDriven,
            Map<String, Object> state,
            double learningRate,
            TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint checkpoint) {
    }

    record RuntimeArtifacts(
            TrainerRuntimeArtifactMetadata.Artifact history,
            TrainerRuntimeArtifactMetadata.SaveOnlyArtifact report,
            TrainerRuntimeArtifactMetadata.Artifact manifest,
            boolean manifestIntegrityMismatch,
            TrainerRuntimeArtifactMetadata.RuntimeCheckpoint runtimeCheckpoint) {
    }

    record EarlyStopping(
            int patience,
            double minDelta,
            String monitorLabel,
            CanonicalTrainer.BestModelMonitorMode monitorMode,
            boolean metricDriven,
            TrainerEarlyStoppingMetadata.MonitorState state,
            boolean triggered,
            int stopEpoch) {
    }

    record Metrics(
            boolean enabled,
            Map<String, Double> train,
            Map<String, Double> validation,
            Map<String, Object> trainDetails,
            Map<String, Object> validationDetails) {
    }

    record DataLoaderPlans(Map<String, Object> metadata) {
        DataLoaderPlans {
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record DataDistribution(Map<String, Object> metadata) {
        DataDistribution {
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record History(
            List<Map<String, Object>> rows,
            int size) {
    }

    record Optimization(
            int gradientAccumulationSteps,
            int pendingGradientAccumulationBatches,
            int optimizerStepCount,
            TrainerGradientClipConfig gradientClip,
            int parameterUpdateDiagnosticsIntervalSteps,
            TrainerOptimizationMetadata.GradientDiagnostics gradients,
            TrainerOptimizationMetadata.ParameterDiagnostics parameters,
            TrainerOptimizationMetadata.UpdateDiagnostics updates) {
    }

    record Throughput(
            ThroughputSnapshot trainTotal,
            ThroughputSnapshot validationTotal) {
    }

    record RuntimeProfile(Map<String, Object> metadata) {
        RuntimeProfile {
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record AccelerationInfo(
            String requestedDevice,
            Acceleration.BackendStatus currentStatus,
            Acceleration.BackendStatus startStatus) {
    }

    record References(
            TrainerSummaryReferenceMetadata.Paths paths,
            TrainerSummaryReferenceMetadata.Errors errors) {
    }
}
