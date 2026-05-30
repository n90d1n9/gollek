package tech.kayys.gollek.ml;

import java.nio.file.Path;
import java.util.Objects;
import tech.kayys.gollek.ml.Gollek.DL.TrainingOptions;
import tech.kayys.gollek.ml.Gollek.DL.TrainingPreset;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * One-call fit helpers exposed through {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlFitFacade extends GollekDlCrossValidationFacade {
    protected GollekDlFitFacade() {
    }

    public static TrainingOptions.Builder trainingOptions() {
        return TrainingOptions.builder();
    }

    public static TrainingOptions.Builder causalLanguageModelingTrainingOptions() {
        return trainingOptions().causalLanguageModelingDefaults();
    }

    public static TrainingOptions.Builder causalLanguageModelingTrainingOptions(float ignoreIndex) {
        return trainingOptions().causalLanguageModelingDefaults(ignoreIndex);
    }

    public static TrainingOptions.Builder nextTokenTrainingOptions() {
        return causalLanguageModelingTrainingOptions();
    }

    public static TrainingOptions.Builder nextTokenTrainingOptions(float ignoreIndex) {
        return causalLanguageModelingTrainingOptions(ignoreIndex);
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fit(model, trainLoader, null, epochs, learningRate, preset, 1.0, 0, 0.0);
    }

    public static TrainingSummary fit(
            NNModule model,
            DataLoader.TensorDatasetSplit split,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fit(
                model,
                split,
                batchSize,
                true,
                0L,
                epochs,
                learningRate,
                preset,
                trainingOptions().build());
    }

    public static TrainingSummary fit(
            NNModule model,
            DataLoader.TensorDatasetSplit split,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fit(model, split, batchSize, true, 0L, epochs, learningRate, preset, options);
    }

    public static TrainingSummary fit(
            NNModule model,
            DataLoader.TensorDatasetSplit split,
            int batchSize,
            boolean shuffleTraining,
            long shuffleSeed,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        Objects.requireNonNull(split, "split must not be null");
        DataLoader.TensorDataLoader trainLoader = split.trainLoader(batchSize, shuffleTraining, shuffleSeed);
        DataLoader.TensorDataLoader validationLoader = split.validationLoader(batchSize);
        return fit(model, trainLoader, validationLoader, epochs, learningRate, preset, options);
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            String deviceId) {
        return fit(
                model,
                trainLoader,
                null,
                epochs,
                learningRate,
                preset,
                trainingOptions().device(deviceId).build());
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip) {
        return fit(model, trainLoader, validationLoader, epochs, learningRate, preset, gradientClip, 0, 0.0);
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return GollekTrainingRuntime.fit(GollekTrainingRequests.fromOptions(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                options));
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta) {
        return fit(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                gradientClip,
                earlyStoppingPatience,
                earlyStoppingMinDelta,
                null,
                false,
                true,
                1,
                "auto");
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta,
            int gradientAccumulationSteps) {
        return fit(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                gradientClip,
                earlyStoppingPatience,
                earlyStoppingMinDelta,
                null,
                false,
                true,
                gradientAccumulationSteps,
                "auto");
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta,
            Path checkpointDir,
            boolean resumeFromCheckpoint,
            boolean failOnCheckpointLoadError) {
        return fit(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                gradientClip,
                earlyStoppingPatience,
                earlyStoppingMinDelta,
                checkpointDir,
                resumeFromCheckpoint,
                failOnCheckpointLoadError,
                1,
                "auto");
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta,
            Path checkpointDir,
            boolean resumeFromCheckpoint,
            boolean failOnCheckpointLoadError,
            int gradientAccumulationSteps) {
        return fit(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                gradientClip,
                earlyStoppingPatience,
                earlyStoppingMinDelta,
                checkpointDir,
                resumeFromCheckpoint,
                failOnCheckpointLoadError,
                gradientAccumulationSteps,
                "auto");
    }

    public static TrainingSummary fit(
            NNModule model,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            double gradientClip,
            int earlyStoppingPatience,
            double earlyStoppingMinDelta,
            Path checkpointDir,
            boolean resumeFromCheckpoint,
            boolean failOnCheckpointLoadError,
            int gradientAccumulationSteps,
            String deviceId) {
        return GollekTrainingRuntime.fit(GollekTrainingRequests.fromLegacy(
                model,
                trainLoader,
                validationLoader,
                epochs,
                learningRate,
                preset,
                gradientClip,
                earlyStoppingPatience,
                earlyStoppingMinDelta,
                checkpointDir,
                resumeFromCheckpoint,
                failOnCheckpointLoadError,
                gradientAccumulationSteps,
                deviceId));
    }
}
