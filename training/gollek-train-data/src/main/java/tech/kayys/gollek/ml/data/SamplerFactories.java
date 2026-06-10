package tech.kayys.gollek.ml.data;

import java.util.function.ToIntFunction;

final class SamplerFactories {
    private SamplerFactories() {
    }

    static WeightedRandomSampler weightedRandomSampler(
            float[] sampleWeights,
            int numSamples,
            boolean replacement,
            long seed) {
        return new WeightedRandomSampler(sampleWeights, numSamples, replacement, seed);
    }

    static LengthBucketBatchSampler lengthBucketBatchSampler(int[] lengths, int batchSize, long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, seed);
    }

    static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        return lengthBucketBatchSampler(SequenceLengthSupport.lengths(dataset, lengthExtractor), batchSize, seed);
    }

    static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            boolean dropLast,
            long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, dropLast, seed);
    }

    static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                batchSize,
                dropLast,
                seed);
    }

    static LengthBucketBatchSampler lengthBucketBatchSampler(
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

    static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    static TokenBudgetBatchSampler tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, seed);
    }

    static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return tokenBudgetBatchSampler(SequenceLengthSupport.lengths(dataset, lengthExtractor), maxTokens, seed);
    }

    static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed);
    }

    static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return tokenBudgetBatchSampler(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                seed);
    }

    static TokenBudgetBatchSampler tokenBudgetBatchSampler(
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

    static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return tokenBudgetBatchSampler(
                SequenceLengthSupport.lengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    static SequentialSampler sequentialSampler() {
        return new SequentialSampler();
    }

    static RandomSampler randomSampler(long seed) {
        return new RandomSampler(seed);
    }

    static RandomSampler randomSampler(int numSamples, boolean replacement, long seed) {
        return new RandomSampler(numSamples, replacement, seed);
    }

    static SubsetSampler subsetSampler(int... indices) {
        return new SubsetSampler(indices);
    }

    static DistributedSampler distributedSampler(int numReplicas, int rank) {
        return new DistributedSampler(numReplicas, rank);
    }

    static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed);
    }

    static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }

}
