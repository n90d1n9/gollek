package tech.kayys.gollek.ml;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.train.CanonicalTrainer;
import tech.kayys.gollek.ml.train.TrainingLossFunction;
import tech.kayys.gollek.ml.train.TrainingMetric;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Runs the high-level Gollek.DL.fit convenience path without bloating the
 * public facade.
 */
final class GollekTrainingRuntime {
    private GollekTrainingRuntime() {
    }

    static TrainingSummary fit(Request request) {
        Objects.requireNonNull(request, "request must not be null");
        Objects.requireNonNull(request.model(), "model must not be null");
        Objects.requireNonNull(request.trainLoader(), "trainLoader must not be null");
        Objects.requireNonNull(request.preset(), "preset must not be null");

        float learningRate = request.learningRate() > 0 ? request.learningRate() : 1e-3f;
        Optimizer optimizer = Gollek.DL.presetOptimizer(request.model(), learningRate, request.preset());
        TrainingLossFunction loss = Gollek.DL.presetLoss(
                request.preset(),
                request.crossEntropyClassWeights(),
                request.focalGamma(),
                request.focalAlpha(),
                request.focalClassWeights(),
                request.causalLanguageModelingIgnoreIndex(),
                request.bcePositiveWeights(),
                request.pinballQuantiles(),
                request.intervalAlpha(),
                request.intervalCrossingPenalty(),
                request.tweediePower(),
                request.tweedieLogInput(),
                request.tweedieEps(),
                request.negativeBinomialLogInput(),
                request.negativeBinomialIncludeConstant(),
                request.negativeBinomialEps(),
                request.zeroInflatedPoissonLogRateInput(),
                request.zeroInflatedPoissonIncludeConstant(),
                request.zeroInflatedPoissonEps(),
                request.zeroInflatedNegativeBinomialLogInput(),
                request.zeroInflatedNegativeBinomialIncludeConstant(),
                request.zeroInflatedNegativeBinomialEps());
        LRScheduler scheduler = request.schedulerFactory() == null
                ? null
                : Objects.requireNonNull(
                        request.schedulerFactory().create(optimizer),
                        "scheduler factory returned null");

        CanonicalTrainer.Builder trainerBuilder = Gollek.DL.trainer()
                .model(request.model())
                .optimizer(optimizer)
                .loss(loss)
                .epochs(request.epochs())
                .gradientClip(request.gradientClip())
                .gradientClipByValue(request.gradientClipValue())
                .gradientAccumulationSteps(request.gradientAccumulationSteps())
                .earlyStopping(request.earlyStoppingPatience(), request.earlyStoppingMinDelta())
                .earlyStoppingMonitorMetric(
                        request.earlyStoppingMonitorMetric(),
                        request.earlyStoppingMonitorMode())
                .resumeFromCheckpoint(request.resumeFromCheckpoint())
                .failOnCheckpointLoadError(request.failOnCheckpointLoadError())
                .saveBestModelCheckpoint(request.saveBestModelCheckpoint())
                .restoreBestModelAtEnd(request.restoreBestModelAtEnd())
                .schedulerMonitorMetric(request.schedulerMonitorMetric())
                .device(request.deviceId())
                .failOnAcceleratorFallback(request.failOnAcceleratorFallback())
                .dataDistributionDiagnostics(request.dataDistributionDiagnostics());
        if (request.gradScaler() != null) {
            trainerBuilder.gradScaler(request.gradScaler());
        } else {
            trainerBuilder.mixedPrecision(request.mixedPrecision());
        }
        if (request.bestModelMonitorMetric() == null) {
            trainerBuilder.bestModelMonitorValidationLoss(request.bestModelMonitorMode());
        } else {
            trainerBuilder.bestModelMonitorMetric(
                    request.bestModelMonitorMetric(),
                    request.bestModelMonitorMode());
        }
        trainerBuilder.metrics(effectiveMetricFactories(request));
        if (scheduler != null) {
            trainerBuilder.scheduler(
                    scheduler,
                    request.schedulerStepUnit() == null
                            ? CanonicalTrainer.SchedulerStepUnit.BATCH
                            : request.schedulerStepUnit());
        }
        if (request.checkpointDir() != null) {
            trainerBuilder.checkpointDir(request.checkpointDir());
        }

        try (CanonicalTrainer trainer = trainerBuilder.build()) {
            trainer.fit(request.trainLoader(), request.validationLoader());
            return trainer.summary();
        }
    }

    private static List<? extends java.util.function.Supplier<? extends TrainingMetric>> effectiveMetricFactories(
            Request request) {
        List<? extends java.util.function.Supplier<? extends TrainingMetric>> configured = request.metricFactories();
        if (configured != null && !configured.isEmpty()) {
            return configured;
        }
        return GollekTrainingPresetSupport.defaultMetricFactories(
                request.preset(),
                request.causalLanguageModelingIgnoreIndex());
    }

    record Request(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            Gollek.DL.TrainingPreset preset,
            double gradientClip,
            double gradientClipValue,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta,
            String earlyStoppingMonitorMetric,
            CanonicalTrainer.BestModelMonitorMode earlyStoppingMonitorMode,
            Path checkpointDir,
            boolean resumeFromCheckpoint,
            boolean failOnCheckpointLoadError,
            boolean saveBestModelCheckpoint,
            boolean restoreBestModelAtEnd,
            String bestModelMonitorMetric,
            CanonicalTrainer.BestModelMonitorMode bestModelMonitorMode,
            int gradientAccumulationSteps,
            boolean mixedPrecision,
            GradScaler gradScaler,
            String deviceId,
            boolean failOnAcceleratorFallback,
            boolean dataDistributionDiagnostics,
            List<? extends java.util.function.Supplier<? extends TrainingMetric>> metricFactories,
            Gollek.DL.SchedulerFactory schedulerFactory,
            CanonicalTrainer.SchedulerStepUnit schedulerStepUnit,
            String schedulerMonitorMetric,
            float[] crossEntropyClassWeights,
            Float focalGamma,
            Float focalAlpha,
            float[] focalClassWeights,
            Float causalLanguageModelingIgnoreIndex,
            float[] bcePositiveWeights,
            float[] pinballQuantiles,
            Float intervalAlpha,
            Float intervalCrossingPenalty,
            Float tweediePower,
            Boolean tweedieLogInput,
            Float tweedieEps,
            Boolean negativeBinomialLogInput,
            Boolean negativeBinomialIncludeConstant,
            Float negativeBinomialEps,
            Boolean zeroInflatedPoissonLogRateInput,
            Boolean zeroInflatedPoissonIncludeConstant,
            Float zeroInflatedPoissonEps,
            Boolean zeroInflatedNegativeBinomialLogInput,
            Boolean zeroInflatedNegativeBinomialIncludeConstant,
            Float zeroInflatedNegativeBinomialEps) {
    }
}
