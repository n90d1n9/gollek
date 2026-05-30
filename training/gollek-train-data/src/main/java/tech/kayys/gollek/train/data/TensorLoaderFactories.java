package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

final class TensorLoaderFactories {

    private TensorLoaderFactories() {
    }

    static DataLoader.TensorDataLoader tensors(GradTensor inputs, GradTensor labels, int batchSize) {
        return plain(TensorLabelEncoding.tensorDataset(inputs, labels), batchSize);
    }

    static DataLoader.TensorDataLoader classification(GradTensor inputs, int[] labels, int batchSize) {
        return plain(TensorLabelEncoding.classificationDataset(inputs, labels), batchSize);
    }

    static DataLoader.TensorDataLoader classBalancedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return classBalancedClassification(inputs, labels, batchSize, batchesPerEpoch, true, seed);
    }

    static DataLoader.TensorDataLoader classBalancedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.classificationDataset(inputs, labels))
                .classBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed)
                .build();
    }

    static DataLoader.TensorDataLoader stratifiedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return stratifiedClassification(inputs, labels, batchSize, batchesPerEpoch, true, seed);
    }

    static DataLoader.TensorDataLoader stratifiedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.classificationDataset(inputs, labels))
                .stratifiedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed)
                .build();
    }

    static DataLoader.TensorDataLoader classWeightedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return classWeightedClassification(inputs, labels, batchSize, numSamples, true, seed);
    }

    static DataLoader.TensorDataLoader classWeightedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.classificationDataset(inputs, labels))
                .batchSize(batchSize)
                .weightedRandomSampler(LabelWeighting.classBalancedSampleWeights(labels), numSamples, replacement, seed)
                .build();
    }

    static DataLoader.TensorDataLoader binary(GradTensor inputs, int[] labels, int batchSize) {
        return plain(TensorLabelEncoding.binaryDataset(inputs, labels), batchSize);
    }

    static DataLoader.TensorDataLoader binary(GradTensor inputs, int[][] labels, int batchSize) {
        return binary(inputs, BinaryLabelMatrix.from(labels), batchSize);
    }

    static DataLoader.TensorDataLoader binary(GradTensor inputs, boolean[][] labels, int batchSize) {
        return binary(inputs, BinaryLabelMatrix.from(labels), batchSize);
    }

    static DataLoader.TensorDataLoader binary(GradTensor inputs, float[][] labels, int batchSize) {
        return binary(inputs, BinaryLabelMatrix.from(labels), batchSize);
    }

    static DataLoader.TensorDataLoader weightedBinary(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return weightedBinary(inputs, labels, batchSize, numSamples, true, seed);
    }

    static DataLoader.TensorDataLoader weightedBinary(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.binaryDataset(inputs, labels))
                .batchSize(batchSize)
                .weightedRandomSampler(LabelWeighting.binaryBalancedSampleWeights(labels), numSamples, replacement, seed)
                .build();
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, replacement, seed);
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, replacement, seed);
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return multiLabelWeightedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, numSamples, replacement, seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, batchesPerEpoch, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinary(
                inputs,
                BinaryLabelMatrix.from(labels),
                batchSize,
                batchesPerEpoch,
                replacement,
                seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, batchesPerEpoch, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinary(
                inputs,
                BinaryLabelMatrix.from(labels),
                batchSize,
                batchesPerEpoch,
                replacement,
                seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBinary(inputs, BinaryLabelMatrix.from(labels), batchSize, batchesPerEpoch, true, seed);
    }

    static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBinary(
                inputs,
                BinaryLabelMatrix.from(labels),
                batchSize,
                batchesPerEpoch,
                replacement,
                seed);
    }

    private static DataLoader.TensorDataLoader binary(
            GradTensor inputs,
            BinaryLabelMatrix matrix,
            int batchSize) {
        return plain(TensorLabelEncoding.binaryDataset(inputs, matrix), batchSize);
    }

    private static DataLoader.TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            BinaryLabelMatrix matrix,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.binaryDataset(inputs, matrix))
                .batchSize(batchSize)
                .weightedRandomSampler(
                        LabelWeighting.multiLabelBalancedSampleWeights(matrix),
                        numSamples,
                        replacement,
                        seed)
                .build();
    }

    private static DataLoader.TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            BinaryLabelMatrix matrix,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return DataLoader.tensorBuilder(TensorLabelEncoding.binaryDataset(inputs, matrix))
                .multiLabelBalancedBatchSampler(matrix, batchSize, batchesPerEpoch, replacement, seed)
                .build();
    }

    private static DataLoader.TensorDataLoader plain(DataLoader.TensorDataset dataset, int batchSize) {
        return DataLoader.tensorBuilder(dataset).batchSize(batchSize).build();
    }
}
