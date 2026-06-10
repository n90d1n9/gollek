package tech.kayys.gollek.train.data;

import java.util.Objects;
import java.util.function.ToIntFunction;

abstract class GenericDataLoaderBuilderSupport<T, B extends GenericDataLoaderBuilderSupport<T, B>> {
    private final Dataset<? extends T> dataset;
    private int batchSize = 32;
    private boolean shuffle;
    private boolean dropLast;
    private Long shuffleSeed;
    private boolean reshuffleEachEpoch;
    private long initialEpoch;
    private IndexSampler sampler;
    private BatchSampler batchSampler;

    GenericDataLoaderBuilderSupport(Dataset<? extends T> dataset) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
    }

    public B batchSize(int batchSize) {
        this.batchSize = DataLoaderBatchSizes.requirePositive(batchSize);
        return self();
    }

    public B shuffle(boolean shuffle) {
        this.shuffle = shuffle;
        if (!shuffle) {
            this.reshuffleEachEpoch = false;
        }
        return self();
    }

    public B shuffle() {
        return shuffle(true);
    }

    public B shuffle(long seed) {
        this.shuffle = true;
        this.shuffleSeed = seed;
        return self();
    }

    public B dropLast(boolean dropLast) {
        this.dropLast = dropLast;
        return self();
    }

    public B dropLast() {
        return dropLast(true);
    }

    public B seed(long seed) {
        this.shuffleSeed = seed;
        return self();
    }

    public B randomSeed(long seed) {
        return seed(seed);
    }

    public B reshuffleEachEpoch() {
        return reshuffleEachEpoch(true);
    }

    public B reshuffleEachEpoch(boolean reshuffleEachEpoch) {
        this.reshuffleEachEpoch = reshuffleEachEpoch;
        if (reshuffleEachEpoch) {
            this.shuffle = true;
        }
        return self();
    }

    public B initialEpoch(long initialEpoch) {
        DataLoaderEpochs.requireInitialEpoch(initialEpoch);
        this.initialEpoch = initialEpoch;
        return self();
    }

    public B startEpoch(long initialEpoch) {
        return initialEpoch(initialEpoch);
    }

    public B sampler(IndexSampler sampler) {
        this.sampler = DataLoaderSamplerSelection.requireSampler(sampler);
        this.batchSampler = null;
        return self();
    }

    public B batchSampler(BatchSampler batchSampler) {
        this.batchSampler = DataLoaderSamplerSelection.requireBatchSampler(batchSampler);
        this.sampler = null;
        return self();
    }

    public B weightedRandomSampler(
            float[] sampleWeights,
            int numSamples,
            boolean replacement,
            long seed) {
        return sampler(DataLoader.weightedRandomSampler(sampleWeights, numSamples, replacement, seed));
    }

    public B lengthBucketBatchSampler(int[] lengths, long seed) {
        return sampler(DataLoader.lengthBucketBatchSampler(lengths, batchSize, dropLast, seed));
    }

    public B lengthBucketBatchSampler(ToIntFunction<? super T> lengthExtractor, long seed) {
        return lengthBucketBatchSampler(DataLoader.sequenceLengths(dataset, lengthExtractor), seed);
    }

    public B lengthBucketBatchSampler(int[] lengths, int batchSize, long seed) {
        batchSize(batchSize);
        return lengthBucketBatchSampler(lengths, seed);
    }

    public B lengthBucketBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        batchSize(batchSize);
        return lengthBucketBatchSampler(lengthExtractor, seed);
    }

    public B lengthBucketBatchSampler(
            int[] lengths,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            long seed) {
        return sampler(DataLoader.lengthBucketBatchSampler(
                lengths,
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed));
    }

    public B lengthBucketBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            long seed) {
        return lengthBucketBatchSampler(
                DataLoader.sequenceLengths(dataset, lengthExtractor),
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                seed);
    }

    public B lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        batchSize(batchSize);
        dropLast(dropLast);
        return sampler(DataLoader.lengthBucketBatchSampler(
                lengths,
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed));
    }

    public B lengthBucketBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(
                DataLoader.sequenceLengths(dataset, lengthExtractor),
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    public B tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, seed));
    }

    public B tokenBudgetBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return tokenBudgetBatchSampler(DataLoader.sequenceLengths(dataset, lengthExtractor), maxTokens, seed);
    }

    public B tokenBudgetBatchSampler(int[] lengths, int maxTokens, int maxExamples, long seed) {
        return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed));
    }

    public B tokenBudgetBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return tokenBudgetBatchSampler(
                DataLoader.sequenceLengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                seed);
    }

    public B tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return batchSampler(DataLoader.tokenBudgetBatchSampler(
                lengths,
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed));
    }

    public B tokenBudgetBatchSampler(
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return tokenBudgetBatchSampler(
                DataLoader.sequenceLengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public B sequentialSampler() {
        return sampler(DataLoader.sequentialSampler());
    }

    public B randomSampler(long seed) {
        return sampler(DataLoader.randomSampler(seed));
    }

    public B randomSampler(int numSamples, boolean replacement, long seed) {
        return sampler(DataLoader.randomSampler(numSamples, replacement, seed));
    }

    public B subsetSampler(int... indices) {
        return sampler(DataLoader.subsetSampler(indices));
    }

    public B distributedSampler(int numReplicas, int rank) {
        return sampler(DataLoader.distributedSampler(numReplicas, rank));
    }

    public B distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return sampler(DataLoader.distributedSampler(numReplicas, rank, shuffle, dropLast, seed));
    }

    public B distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return sampler(DataLoader.distributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch));
    }

    protected abstract B self();

    final GenericDataLoaderRuntime<T> runtime() {
        return new GenericDataLoaderRuntime<>(
                dataset,
                batchSize,
                shuffle,
                dropLast,
                shuffleSeed,
                sampler,
                batchSampler,
                reshuffleEachEpoch,
                initialEpoch);
    }
}
