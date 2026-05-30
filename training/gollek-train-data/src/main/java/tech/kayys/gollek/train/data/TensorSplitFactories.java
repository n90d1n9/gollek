package tech.kayys.gollek.train.data;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;

final class TensorSplitFactories {

    private TensorSplitFactories() {
    }

    private static BinaryLabelMatrix matrix(int[][] labels) {
        return BinaryLabelMatrix.from(labels);
    }

    private static BinaryLabelMatrix matrix(boolean[][] labels) {
        return BinaryLabelMatrix.from(labels);
    }

    private static BinaryLabelMatrix matrix(float[][] labels) {
        return BinaryLabelMatrix.from(labels);
    }

    static DataLoader.TensorDatasetSplit split(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            long seed) {
        return TensorLabelEncoding.tensorDataset(inputs, labels).split(trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit split(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorLabelEncoding.tensorDataset(inputs, labels).split(trainFraction, validationFraction, seed);
    }

    static List<DataLoader.TensorDatasetFold> kFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed) {
        return TensorLabelEncoding.tensorDataset(inputs, labels).kFold(folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> repeatedKFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed) {
        return repeatedKFold(TensorLabelEncoding.tensorDataset(inputs, labels), folds, repeats, seed);
    }

    static List<DataLoader.TensorDatasetFold> repeatedKFold(
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return TensorDatasetSplits.repeatedKFoldBySeed(dataset, folds, repeats, seed);
    }

    static List<DataLoader.TensorDatasetFold> groupKFold(
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed) {
        return groupKFold(TensorLabelEncoding.tensorDataset(inputs, labels), groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> groupKFold(
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return ClassStratifiedTensorDatasetSplits.groupKFold(dataset, groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return ClassStratifiedTensorDatasetSplits.stratifiedGroupKFold(dataset, labels, groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits) {
        return timeSeriesSplit(TensorLabelEncoding.tensorDataset(inputs, labels), splits);
    }

    static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return timeSeriesSplit(
                TensorLabelEncoding.tensorDataset(inputs, labels),
                splits,
                validationSize,
                gap,
                maxTrainSize);
    }

    static List<DataLoader.TensorDatasetFold> timeSeriesSplit(DataLoader.TensorDataset dataset, int splits) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        int validationSize = TensorDatasetSplits.defaultTimeSeriesValidationSize(dataset.size(), splits);
        return TensorDatasetSplits.timeSeriesFolds(dataset, splits, validationSize, 0, 0);
    }

    static List<DataLoader.TensorDatasetFold> timeSeriesSplit(
            DataLoader.TensorDataset dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return TensorDatasetSplits.timeSeriesFolds(dataset, splits, validationSize, gap, maxTrainSize);
    }

    static DataLoader.TensorDatasetSplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorLabelEncoding.classificationDataset(inputs, labels).split(trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorLabelEncoding.classificationDataset(inputs, labels).split(
                trainFraction,
                validationFraction,
                seed);
    }

    static DataLoader.TensorDatasetSplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.classificationDataset(inputs, labels),
                labels,
                trainFraction,
                seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.classificationDataset(inputs, labels),
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> classificationStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedKFold(
                TensorLabelEncoding.classificationDataset(inputs, labels),
                labels,
                folds,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> classificationRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.repeatedStratifiedKFold(
                TensorLabelEncoding.classificationDataset(inputs, labels),
                labels,
                folds,
                repeats,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> classificationStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return stratifiedGroupKFold(
                TensorLabelEncoding.classificationDataset(inputs, labels),
                labels,
                groups,
                folds,
                seed);
    }

    static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorLabelEncoding.binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorLabelEncoding.binaryDataset(inputs, labels).split(trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                trainFraction,
                seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> binaryStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.stratifiedKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                folds,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> binaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return ClassStratifiedTensorDatasetSplits.repeatedStratifiedKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                folds,
                repeats,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> binaryStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return stratifiedGroupKFold(TensorLabelEncoding.binaryDataset(inputs, labels), labels, groups, folds, seed);
    }

    static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        return TensorLabelEncoding.binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorLabelEncoding.binaryDataset(inputs, labels).split(trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                trainFraction,
                seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedGroupSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                groups,
                trainFraction,
                seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedGroupSplit(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int folds,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                folds,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int folds,
            int repeats,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.repeatedStratifiedKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                folds,
                repeats,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.stratifiedGroupKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                groups,
                folds,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return MultiLabelStratifiedTensorDatasetSplits.repeatedStratifiedGroupKFold(
                TensorLabelEncoding.binaryDataset(inputs, labels),
                labels,
                groups,
                folds,
                repeats,
                seed);
    }

    static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return binarySplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, matrix(labels), groups, trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(
                inputs,
                matrix(labels),
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, matrix(labels), folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, matrix(labels), folds, repeats, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, matrix(labels), groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, matrix(labels), groups, folds, repeats, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, matrix(labels), groups, trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(
                inputs,
                matrix(labels),
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, matrix(labels), folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, matrix(labels), folds, repeats, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, matrix(labels), groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, matrix(labels), groups, folds, repeats, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedSplit(inputs, matrix(labels), trainFraction, validationFraction, seed);
    }

    static DataLoader.TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(inputs, matrix(labels), groups, trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return multiLabelBinaryStratifiedGroupSplit(
                inputs,
                matrix(labels),
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedKFold(inputs, matrix(labels), folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedKFold(inputs, matrix(labels), folds, repeats, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return multiLabelBinaryStratifiedGroupKFold(inputs, matrix(labels), groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return multiLabelBinaryRepeatedStratifiedGroupKFold(inputs, matrix(labels), groups, folds, repeats, seed);
    }
}
