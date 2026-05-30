package tech.kayys.gollek.ml;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader;

/**
 * Dataset split helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlSplitFacade extends GollekDlDatasetFacade {
    protected GollekDlSplitFacade() {
    }

    public static DataLoader.TensorDatasetSplit split(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            long seed) {
        return DataLoader.split(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit split(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.split(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> kFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed) {
        return DataLoader.kFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> repeatedKFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.repeatedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> groupKFold(
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.groupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits) {
        return DataLoader.timeSeriesSplit(inputs, labels, splits);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return DataLoader.timeSeriesSplit(inputs, labels, splits, validationSize, gap, maxTrainSize);
    }

    public static DataLoader.TensorDatasetSplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return DataLoader.classificationSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.classificationSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return DataLoader.classificationStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.classificationStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> classificationStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return DataLoader.classificationStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> classificationRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.classificationRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> classificationStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.classificationStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return DataLoader.binaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.binaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return DataLoader.binaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.binaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.binaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(
                inputs, labels, groups, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(
                inputs, labels, groups, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupSplit(
                inputs, labels, groups, trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return DataLoader.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return DataLoader.multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static DataLoader.TensorDatasetSplit split(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.split(trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit split(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.split(trainFraction, validationFraction, seed);
    }

    public static List<DataLoader.TensorDatasetFold> kFold(
            DataLoader.TensorDataset dataset,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.kFold(folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> repeatedKFold(
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.repeatedKFold(folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> groupKFold(
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.groupKFold(groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.stratifiedGroupKFold(labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            DataLoader.TensorDataset dataset,
            int splits) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.timeSeriesSplit(splits);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            DataLoader.TensorDataset dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.timeSeriesSplit(splits, validationSize, gap, maxTrainSize);
    }

    public static List<DataLoader.TensorDatasetFold> crossValidationFolds(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed) {
        return kFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> crossValidationFolds(
            DataLoader.TensorDataset dataset,
            int folds,
            long seed) {
        return kFold(dataset, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> repeatedCrossValidationFolds(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed) {
        return repeatedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> repeatedCrossValidationFolds(
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed) {
        return repeatedKFold(dataset, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> groupCrossValidationFolds(
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed) {
        return groupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> groupCrossValidationFolds(
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed) {
        return groupKFold(dataset, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> stratifiedGroupCrossValidationFolds(
            DataLoader.TensorDataset dataset,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return stratifiedGroupKFold(dataset, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesCrossValidationFolds(
            GradTensor inputs,
            GradTensor labels,
            int splits) {
        return timeSeriesSplit(inputs, labels, splits);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesCrossValidationFolds(
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return timeSeriesSplit(inputs, labels, splits, validationSize, gap, maxTrainSize);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesCrossValidationFolds(
            DataLoader.TensorDataset dataset,
            int splits) {
        return timeSeriesSplit(dataset, splits);
    }

    public static List<DataLoader.TensorDatasetFold> timeSeriesCrossValidationFolds(
            DataLoader.TensorDataset dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return timeSeriesSplit(dataset, splits, validationSize, gap, maxTrainSize);
    }

    public static List<DataLoader.TensorDatasetFold> classificationStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return classificationStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> classificationRepeatedStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return classificationRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> classificationStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return classificationStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return binaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryRepeatedStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return binaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> binaryStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return binaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedCrossValidationFolds(
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedCrossValidationFolds(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedCrossValidationFolds(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedCrossValidationFolds(
            GradTensor inputs,
            float[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedCrossValidationFolds(
            GradTensor inputs,
            float[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupCrossValidationFolds(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, labels, groups, folds, repeats, seed);
    }

    public static DataLoader.TensorDatasetSplit trainValidationSplit(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            long seed) {
        return split(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit trainValidationSplit(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            long seed) {
        return split(dataset, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit trainValidationTestSplit(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return split(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit trainValidationTestSplit(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            double validationFraction,
            long seed) {
        return split(dataset, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit classificationTrainValidationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return classificationSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit classificationTrainValidationTestSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return classificationSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit classificationStratifiedTrainValidationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return classificationStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit classificationStratifiedTrainValidationTestSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return classificationStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryTrainValidationTestSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryStratifiedTrainValidationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return binaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryStratifiedTrainValidationTestSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryTrainValidationTestSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryTrainValidationTestSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit binaryTrainValidationSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit binaryTrainValidationTestSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryTrainValidationTestSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binaryTrainValidationTestSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedTrainValidationTestSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupTrainValidationSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupTrainValidationTestSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryTrainValidationTestSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binaryTrainValidationTestSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedTrainValidationTestSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupTrainValidationSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupTrainValidationTestSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryTrainValidationSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binaryTrainValidationSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryTrainValidationTestSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binaryTrainValidationTestSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedTrainValidationSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedTrainValidationTestSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupTrainValidationSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, seed);
    }

    public static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupTrainValidationTestSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, labels, groups, trainFraction, validationFraction, seed);
    }
}
