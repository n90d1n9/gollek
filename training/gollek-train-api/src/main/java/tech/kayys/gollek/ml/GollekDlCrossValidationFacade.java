package tech.kayys.gollek.ml;

import java.util.List;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.Gollek.DL.TrainingOptions;
import tech.kayys.gollek.ml.Gollek.DL.TrainingPreset;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.train.data.DataLoader;

/**
 * Cross-validation helpers exposed through {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlCrossValidationFacade extends GollekDlEvaluationFacade {
    protected GollekDlCrossValidationFacade() {
    }

    public static CrossValidationTrainingSummary fitKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitKFold(
                modelFactory,
                dataset,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                dataset.kFold(folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitRepeatedKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitRepeatedKFold(
                modelFactory,
                dataset,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitRepeatedKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.repeatedKFold(dataset, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitGroupKFold(
                modelFactory,
                dataset,
                groups,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.groupKFold(dataset, groups, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitTimeSeriesSplit(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int splits,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitTimeSeriesSplit(
                modelFactory,
                dataset,
                splits,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitTimeSeriesSplit(
            Supplier<? extends NNModule> modelFactory,
            DataLoader.TensorDataset dataset,
            int splits,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.timeSeriesSplit(dataset, splits),
                batchSize,
                false,
                0L,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.kFold(inputs, labels, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitRepeatedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitRepeatedKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitRepeatedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.repeatedKFold(inputs, labels, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitGroupKFold(
                modelFactory,
                inputs,
                labels,
                groups,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.groupKFold(inputs, labels, groups, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitTimeSeriesSplit(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitTimeSeriesSplit(
                modelFactory,
                inputs,
                labels,
                splits,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitTimeSeriesSplit(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.timeSeriesSplit(inputs, labels, splits),
                batchSize,
                false,
                0L,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitClassificationStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.classificationStratifiedKFold(inputs, labels, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitClassificationRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitClassificationRepeatedStratifiedKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitClassificationRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.classificationRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitClassificationStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitClassificationStratifiedGroupKFold(
                modelFactory,
                inputs,
                labels,
                groups,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitClassificationStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.classificationStratifiedGroupKFold(inputs, labels, groups, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitBinaryStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.binaryStratifiedKFold(inputs, labels, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitBinaryRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitBinaryRepeatedStratifiedKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitBinaryRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.binaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitBinaryStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitBinaryStratifiedGroupKFold(
                modelFactory,
                inputs,
                labels,
                groups,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitBinaryStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.binaryStratifiedGroupKFold(inputs, labels, groups, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitMultiLabelBinaryStratifiedKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitMultiLabelBinaryRepeatedStratifiedKFold(
                modelFactory,
                inputs,
                labels,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryRepeatedStratifiedKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitMultiLabelBinaryStratifiedGroupKFold(
                modelFactory,
                inputs,
                labels,
                groups,
                folds,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryRepeatedStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitMultiLabelBinaryRepeatedStratifiedGroupKFold(
                modelFactory,
                inputs,
                labels,
                groups,
                folds,
                repeats,
                seed,
                batchSize,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitMultiLabelBinaryRepeatedStratifiedGroupKFold(
            Supplier<? extends NNModule> modelFactory,
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                DataLoader.multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed),
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitCrossValidation(
            Supplier<? extends NNModule> modelFactory,
            List<DataLoader.TensorDatasetFold> folds,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset) {
        return fitCrossValidation(
                modelFactory,
                folds,
                batchSize,
                true,
                0L,
                epochs,
                learningRate,
                preset,
                Gollek.DL.trainingOptions().build());
    }

    public static CrossValidationTrainingSummary fitCrossValidation(
            Supplier<? extends NNModule> modelFactory,
            List<DataLoader.TensorDatasetFold> folds,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                folds,
                batchSize,
                true,
                0L,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary fitCrossValidation(
            Supplier<? extends NNModule> modelFactory,
            List<DataLoader.TensorDatasetFold> folds,
            int batchSize,
            boolean shuffleTraining,
            long shuffleSeed,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return GollekTrainingCrossValidationRuntime.fit(
                modelFactory,
                folds,
                batchSize,
                shuffleTraining,
                shuffleSeed,
                epochs,
                learningRate,
                preset,
                options);
    }

    public static CrossValidationTrainingSummary crossValidate(
            Supplier<? extends NNModule> modelFactory,
            List<DataLoader.TensorDatasetFold> folds,
            int batchSize,
            int epochs,
            float learningRate,
            TrainingPreset preset,
            TrainingOptions options) {
        return fitCrossValidation(
                modelFactory,
                folds,
                batchSize,
                epochs,
                learningRate,
                preset,
                options);
    }
}
