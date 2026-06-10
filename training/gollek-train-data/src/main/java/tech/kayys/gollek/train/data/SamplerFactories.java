package tech.kayys.gollek.train.data;

import java.util.function.ToIntFunction;

final class SamplerFactories {

    private SamplerFactories() {
    }

    static WeightedRandomSampler weightedRandom(
            float[] sampleWeights,
            int numSamples,
            boolean replacement,
            long seed) {
        return new WeightedRandomSampler(sampleWeights, numSamples, replacement, seed);
    }

    static LengthBucketBatchSampler lengthBucket(
            int[] lengths,
            int batchSize,
            long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, seed);
    }

    static <T> LengthBucketBatchSampler lengthBucket(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        return lengthBucket(SequenceLengthSupport.lengths(dataset, lengthExtractor), batchSize, seed);
    }

    static LengthBucketBatchSampler lengthBucket(
            int[] lengths,
            int batchSize,
            boolean dropLast,
            long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, dropLast, seed);
    }

    static <T> LengthBucketBatchSampler lengthBucket(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            boolean dropLast,
            long seed) {
        return lengthBucket(SequenceLengthSupport.lengths(dataset, lengthExtractor), batchSize, dropLast, seed);
    }

    static LengthBucketBatchSampler lengthBucket(
            int[] lengths,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return new LengthBucketBatchSampler(
                lengths,
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    static <T> LengthBucketBatchSampler lengthBucket(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return lengthBucket(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    static TokenBudgetBatchSampler tokenBudget(int[] lengths, int maxTokens, long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, seed);
    }

    static <T> TokenBudgetBatchSampler tokenBudget(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return tokenBudget(SequenceLengthSupport.lengths(dataset, lengthExtractor), maxTokens, seed);
    }

    static TokenBudgetBatchSampler tokenBudget(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed);
    }

    static <T> TokenBudgetBatchSampler tokenBudget(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return tokenBudget(SequenceLengthSupport.lengths(dataset, lengthExtractor), maxTokens, maxExamples, seed);
    }

    static TokenBudgetBatchSampler tokenBudget(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return new TokenBudgetBatchSampler(
                lengths,
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    static <T> TokenBudgetBatchSampler tokenBudget(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return tokenBudget(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    static ClassBalancedBatchSampler classBalanced(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return new ClassBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    static ClassBalancedBatchSampler classBalanced(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new ClassBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    static StratifiedBatchSampler stratified(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return new StratifiedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    static StratifiedBatchSampler stratified(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new StratifiedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new MultiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    static MultiLabelBalancedBatchSampler multiLabelBalanced(
            float[] labels,
            int rows,
            int columns,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return new MultiLabelBalancedBatchSampler(
                labels, rows, columns, batchSize, batchesPerEpoch, replacement, seed);
    }

    static SequentialSampler sequential() {
        return new SequentialSampler();
    }

    static RandomSampler random(long seed) {
        return new RandomSampler(seed);
    }

    static RandomSampler random(int numSamples, boolean replacement, long seed) {
        return new RandomSampler(numSamples, replacement, seed);
    }

    static SubsetSampler subset(int... indices) {
        return new SubsetSampler(indices);
    }

    static DistributedSampler distributed(int numReplicas, int rank) {
        return new DistributedSampler(numReplicas, rank);
    }

    static DistributedSampler distributed(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed);
    }

    static DistributedSampler distributed(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }
}
