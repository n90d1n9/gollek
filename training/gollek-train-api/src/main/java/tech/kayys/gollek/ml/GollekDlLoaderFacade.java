package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.ClassBalancedBatchSampler;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.MultiLabelBalancedBatchSampler;
import tech.kayys.gollek.train.data.StratifiedBatchSampler;

/**
 * Data loader construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlLoaderFacade extends GollekDlSplitFacade {
    protected GollekDlLoaderFacade() {
    }

    public static DataLoader.TensorBuilder dataLoader(DataLoader.TensorDatasetAdapter dataset) {
        return DataLoader.tensorBuilder(dataset);
    }

    public static DataLoader.TensorDataLoader dataLoader(
            GradTensor inputs,
            GradTensor labels,
            int batchSize) {
        return DataLoader.tensors(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader classificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize) {
        return DataLoader.classification(inputs, labels, batchSize);
    }

    public static ClassBalancedBatchSampler classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.classBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    public static ClassBalancedBatchSampler classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.classBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static StratifiedBatchSampler stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.stratifiedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    public static StratifiedBatchSampler stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.stratifiedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static float[] multiLabelBalancedSampleWeights(int[][] labels) {
        return DataLoader.multiLabelBalancedSampleWeights(labels);
    }

    public static float[] multiLabelBalancedSampleWeights(boolean[][] labels) {
        return DataLoader.multiLabelBalancedSampleWeights(labels);
    }

    public static float[] multiLabelBalancedSampleWeights(float[][] labels) {
        return DataLoader.multiLabelBalancedSampleWeights(labels);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader classBalancedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.classBalancedClassification(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader classBalancedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.classBalancedClassification(inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader stratifiedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.stratifiedClassification(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader stratifiedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.stratifiedClassification(inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader classWeightedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return DataLoader.classWeightedClassification(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader classWeightedClassificationDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.classWeightedClassification(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader binaryDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize) {
        return DataLoader.binary(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader weightedBinaryDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return DataLoader.weightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader weightedBinaryDataLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.weightedBinary(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader binaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize) {
        return DataLoader.binary(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader binaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize) {
        return DataLoader.binary(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader binaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize) {
        return DataLoader.binary(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryDataLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.multiLabelBalancedBinary(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.ClassificationDistributionReport classificationDistribution(
            DataLoader.TensorDataLoader loader,
            int numClasses) {
        return DataLoader.classificationDistribution(loader, numClasses);
    }

    public static DataLoader.ClassificationDistributionReport classificationDistribution(
            DataLoader.TensorDataLoader loader) {
        return DataLoader.classificationDistribution(loader);
    }

    public static DataLoader.ClassificationDistributionReport binaryDistribution(
            DataLoader.TensorDataLoader loader) {
        return DataLoader.binaryDistribution(loader);
    }

    public static DataLoader.MultiLabelDistributionReport multiLabelDistribution(
            DataLoader.TensorDataLoader loader,
            int labelCount) {
        return DataLoader.multiLabelDistribution(loader, labelCount);
    }

    public static DataLoader.MultiLabelDistributionReport multiLabelDistribution(
            DataLoader.TensorDataLoader loader) {
        return DataLoader.multiLabelDistribution(loader);
    }

    public static DataLoader.TensorDataLoader classificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize) {
        return classificationDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader classBalancedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return classBalancedClassificationDataLoader(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader classBalancedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return classBalancedClassificationDataLoader(
                inputs,
                labels,
                batchSize,
                batchesPerEpoch,
                replacement,
                seed);
    }

    public static DataLoader.TensorDataLoader stratifiedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return stratifiedClassificationDataLoader(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader stratifiedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return stratifiedClassificationDataLoader(inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader classWeightedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return classWeightedClassificationDataLoader(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader classWeightedClassificationLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return classWeightedClassificationDataLoader(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader binaryLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader weightedBinaryLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return weightedBinaryDataLoader(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader weightedBinaryLoader(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return weightedBinaryDataLoader(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader binaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader binaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader binaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelBinaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize) {
        return binaryDataLoader(inputs, labels, batchSize);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelWeightedBinaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinaryDataLoader(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static DataLoader.TensorDataLoader multiLabelBalancedBinaryLoader(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinaryDataLoader(
                inputs, labels, batchSize, batchesPerEpoch, replacement, seed);
    }
}
