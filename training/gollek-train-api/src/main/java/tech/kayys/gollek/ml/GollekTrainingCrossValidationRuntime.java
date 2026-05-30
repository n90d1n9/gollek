package tech.kayys.gollek.ml;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.Gollek.DL.TrainingOptions;
import tech.kayys.gollek.ml.Gollek.DL.TrainingPreset;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Executes tensor fold training while keeping the public facade compact.
 */
final class GollekTrainingCrossValidationRuntime {
    private GollekTrainingCrossValidationRuntime() {
    }

    static CrossValidationTrainingSummary fit(
            Supplier<? extends NNModule> modelFactory,
            List<DataLoader.TensorDatasetFold> folds,
            int batchSize,
            boolean shuffleTraining,
            long shuffleSeed,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        Objects.requireNonNull(folds, "folds must not be null");
        Objects.requireNonNull(preset, "preset must not be null");
        if (folds.isEmpty()) {
            throw new IllegalArgumentException("folds must not be empty");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }

        TrainingOptions resolvedOptions = options == null ? Gollek.DL.trainingOptions().build() : options;
        List<CrossValidationTrainingSummary.Fold> foldSummaries = new ArrayList<>(folds.size());
        for (DataLoader.TensorDatasetFold fold : folds) {
            NNModule model = Objects.requireNonNull(
                    modelFactory.get(),
                    "modelFactory returned null for fold " + fold.foldIndex());
            TrainingOptions foldOptions = withFoldCheckpointDir(resolvedOptions, fold);
            TrainingSummary summary = GollekTrainingRuntime.fit(GollekTrainingRequests.fromOptions(
                    model,
                    fold.trainLoader(batchSize, shuffleTraining, shuffleSeed + fold.foldIndex()),
                    fold.validationLoader(batchSize),
                    epochs,
                    learningRate,
                    preset,
                    foldOptions));
            foldSummaries.add(new CrossValidationTrainingSummary.Fold(
                    fold.foldIndex(),
                    fold.foldCount(),
                    fold.train().size(),
                    fold.validation().size(),
                    summary));
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batchSize", batchSize);
        metadata.put("epochs", Math.max(1, epochs));
        metadata.put("learningRate", learningRate);
        metadata.put("preset", preset.name());
        metadata.put("shuffleTraining", shuffleTraining);
        metadata.put("shuffleSeed", shuffleSeed);
        metadata.put("device", resolvedOptions.device());
        metadata.put("checkpointDir", resolvedOptions.checkpointDir() == null
                ? null
                : resolvedOptions.checkpointDir().toString());
        return new CrossValidationTrainingSummary(
                foldSummaries.size(),
                foldSummaries,
                null,
                null,
                null,
                null,
                null,
                metadata);
    }

    private static TrainingOptions withFoldCheckpointDir(
            TrainingOptions options,
            DataLoader.TensorDatasetFold fold) {
        Path checkpointDir = options.checkpointDir();
        if (checkpointDir == null) {
            return options;
        }
        Path foldCheckpointDir = checkpointDir.resolve("fold-" + (fold.foldIndex() + 1));
        return new TrainingOptions(
                options.gradientClip(),
                options.gradientClipValue(),
                options.earlyStoppingPatience(),
                options.earlyStoppingMinDelta(),
                options.earlyStoppingMonitorMetric(),
                options.earlyStoppingMonitorMode(),
                foldCheckpointDir,
                options.resumeFromCheckpoint(),
                options.failOnCheckpointLoadError(),
                options.saveBestModelCheckpoint(),
                options.restoreBestModelAtEnd(),
                options.bestModelMonitorMetric(),
                options.bestModelMonitorMode(),
                options.gradientAccumulationSteps(),
                options.mixedPrecision(),
                options.gradScaler(),
                options.device(),
                options.failOnAcceleratorFallback(),
                options.dataDistributionDiagnostics(),
                options.metricFactories(),
                options.schedulerFactory(),
                options.schedulerStepUnit(),
                options.schedulerMonitorMetric(),
                options.crossEntropyClassWeights(),
                options.focalGamma(),
                options.focalAlpha(),
                options.focalClassWeights(),
                options.causalLanguageModelingIgnoreIndex(),
                options.bcePositiveWeights(),
                options.pinballQuantiles(),
                options.intervalAlpha(),
                options.intervalCrossingPenalty(),
                options.tweediePower(),
                options.tweedieLogInput(),
                options.tweedieEps(),
                options.negativeBinomialLogInput(),
                options.negativeBinomialIncludeConstant(),
                options.negativeBinomialEps(),
                options.zeroInflatedPoissonLogRateInput(),
                options.zeroInflatedPoissonIncludeConstant(),
                options.zeroInflatedPoissonEps(),
                options.zeroInflatedNegativeBinomialLogInput(),
                options.zeroInflatedNegativeBinomialIncludeConstant(),
                options.zeroInflatedNegativeBinomialEps());
    }
}
