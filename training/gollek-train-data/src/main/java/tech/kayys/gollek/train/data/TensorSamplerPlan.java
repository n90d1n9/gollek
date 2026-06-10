package tech.kayys.gollek.train.data;

import java.util.Arrays;
import java.util.Objects;

final class TensorSamplerPlan {
    private static final TensorSamplerPlan NONE = new TensorSamplerPlan(
            null, null, null, null);

    private final IndexSampler explicitSampler;
    private final ClassBalancedSamplerConfig classBalanced;
    private final StratifiedSamplerConfig stratified;
    private final MultiLabelBalancedSamplerConfig multiLabelBalanced;

    private TensorSamplerPlan(
            IndexSampler explicitSampler,
            ClassBalancedSamplerConfig classBalanced,
            StratifiedSamplerConfig stratified,
            MultiLabelBalancedSamplerConfig multiLabelBalanced) {
        this.explicitSampler = explicitSampler;
        this.classBalanced = classBalanced;
        this.stratified = stratified;
        this.multiLabelBalanced = multiLabelBalanced;
    }

    static TensorSamplerPlan none() {
        return NONE;
    }

    static TensorSamplerPlan explicit(IndexSampler sampler) {
        return new TensorSamplerPlan(Objects.requireNonNull(sampler, "sampler must not be null"), null, null, null);
    }

    static TensorSamplerPlan classBalanced(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new TensorSamplerPlan(
                null,
                new ClassBalancedSamplerConfig(labels, batchesPerEpoch, replacement, seed),
                null,
                null);
    }

    static TensorSamplerPlan stratified(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new TensorSamplerPlan(
                null,
                null,
                new StratifiedSamplerConfig(labels, batchesPerEpoch, replacement, seed),
                null);
    }

    static TensorSamplerPlan multiLabelBalanced(
            BinaryLabelMatrix matrix,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        Objects.requireNonNull(matrix, "matrix must not be null");
        return new TensorSamplerPlan(
                null,
                null,
                null,
                new MultiLabelBalancedSamplerConfig(
                        matrix.values(), matrix.rows(), matrix.columns(), batchesPerEpoch, replacement, seed));
    }

    IndexSampler resolve(int batchSize) {
        if (classBalanced != null) {
            return classBalanced.toSampler(batchSize);
        }
        if (stratified != null) {
            return stratified.toSampler(batchSize);
        }
        if (multiLabelBalanced != null) {
            return multiLabelBalanced.toSampler(batchSize);
        }
        return explicitSampler;
    }

    private record ClassBalancedSamplerConfig(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        private ClassBalancedSamplerConfig {
            labels = Arrays.copyOf(Objects.requireNonNull(labels, "labels must not be null"), labels.length);
        }

        ClassBalancedBatchSampler toSampler(int batchSize) {
            return SamplerFactories.classBalanced(labels, batchSize, batchesPerEpoch, replacement, seed);
        }
    }

    private record StratifiedSamplerConfig(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        private StratifiedSamplerConfig {
            labels = Arrays.copyOf(Objects.requireNonNull(labels, "labels must not be null"), labels.length);
        }

        StratifiedBatchSampler toSampler(int batchSize) {
            return SamplerFactories.stratified(labels, batchSize, batchesPerEpoch, replacement, seed);
        }
    }

    private record MultiLabelBalancedSamplerConfig(
            float[] labels,
            int rows,
            int columns,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        private MultiLabelBalancedSamplerConfig {
            labels = Arrays.copyOf(Objects.requireNonNull(labels, "labels must not be null"), labels.length);
        }

        MultiLabelBalancedBatchSampler toSampler(int batchSize) {
            return SamplerFactories.multiLabelBalanced(
                    labels, rows, columns, batchSize, batchesPerEpoch, replacement, seed);
        }
    }
}
