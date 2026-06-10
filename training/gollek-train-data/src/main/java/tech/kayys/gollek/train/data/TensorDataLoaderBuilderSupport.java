package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Objects;
import java.util.function.ToIntFunction;

abstract class TensorDataLoaderBuilderSupport<B extends TensorDataLoaderBuilderSupport<B>> {
    final DataLoader.TensorDatasetAdapter dataset;
    int batchSize = 32;
    boolean shuffle = false;
    boolean dropLast = false;
    Long shuffleSeed;
    boolean reshuffleEachEpoch;
    long initialEpoch;
    TensorSamplerPlan samplerPlan = TensorSamplerPlan.none();
    BatchSampler batchSampler;
    DataLoader.CollateFn collateFn;

    TensorDataLoaderBuilderSupport(DataLoader.TensorDatasetAdapter dataset) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
    }

    protected abstract B self();

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
        this.samplerPlan = TensorSamplerPlan.explicit(DataLoaderSamplerSelection.requireSampler(sampler));
        this.batchSampler = null;
        return self();
    }

    public B batchSampler(BatchSampler batchSampler) {
        this.batchSampler = DataLoaderSamplerSelection.requireBatchSampler(batchSampler);
        this.samplerPlan = TensorSamplerPlan.none();
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

    public B lengthBucketBatchSampler(ToIntFunction<GradTensor[]> lengthExtractor, long seed) {
        return lengthBucketBatchSampler(tensorDatasetLengths(lengthExtractor), seed);
    }

    public B lengthBucketBatchSampler(int[] lengths, int batchSize, long seed) {
        batchSize(batchSize);
        return lengthBucketBatchSampler(lengths, seed);
    }

    public B lengthBucketBatchSampler(
            ToIntFunction<GradTensor[]> lengthExtractor,
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
            ToIntFunction<GradTensor[]> lengthExtractor,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            long seed) {
        return lengthBucketBatchSampler(
                tensorDatasetLengths(lengthExtractor),
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
            ToIntFunction<GradTensor[]> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(
                tensorDatasetLengths(lengthExtractor),
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    public B inputLengthBucketBatchSampler(long seed) {
        return lengthBucketBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), seed);
    }

    public B inputLengthBucketBatchSampler(int batchSize, long seed) {
        batchSize(batchSize);
        return inputLengthBucketBatchSampler(seed);
    }

    public B tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, seed));
    }

    public B tokenBudgetBatchSampler(ToIntFunction<GradTensor[]> lengthExtractor, int maxTokens, long seed) {
        return tokenBudgetBatchSampler(tensorDatasetLengths(lengthExtractor), maxTokens, seed);
    }

    public B tokenBudgetBatchSampler(int[] lengths, int maxTokens, int maxExamples, long seed) {
        return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed));
    }

    public B tokenBudgetBatchSampler(
            ToIntFunction<GradTensor[]> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return tokenBudgetBatchSampler(tensorDatasetLengths(lengthExtractor), maxTokens, maxExamples, seed);
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
            ToIntFunction<GradTensor[]> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return tokenBudgetBatchSampler(
                tensorDatasetLengths(lengthExtractor),
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public B inputTokenBudgetBatchSampler(int maxTokens, long seed) {
        return tokenBudgetBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), maxTokens, seed);
    }

    public B inputTokenBudgetBatchSampler(int maxTokens, int maxExamples, long seed) {
        return tokenBudgetBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), maxTokens, maxExamples, seed);
    }

    public B classBalancedBatchSampler(int[] labels, int batchesPerEpoch, long seed) {
        return classBalancedBatchSampler(labels, batchesPerEpoch, true, seed);
    }

    public B classBalancedBatchSampler(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this.samplerPlan = TensorSamplerPlan.classBalanced(labels, batchesPerEpoch, replacement, seed);
        this.batchSampler = null;
        return self();
    }

    public B classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return classBalancedBatchSampler(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public B classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return classBalancedBatchSampler(labels, batchesPerEpoch, replacement, seed);
    }

    public B stratifiedBatchSampler(int[] labels, int batchesPerEpoch, long seed) {
        return stratifiedBatchSampler(labels, batchesPerEpoch, true, seed);
    }

    public B stratifiedBatchSampler(
            int[] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this.samplerPlan = TensorSamplerPlan.stratified(labels, batchesPerEpoch, replacement, seed);
        this.batchSampler = null;
        return self();
    }

    public B stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return stratifiedBatchSampler(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public B stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return stratifiedBatchSampler(labels, batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBatchSampler(BinaryLabelMatrix.from(labels), batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBatchSampler(BinaryLabelMatrix.from(labels), batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return multiLabelBalancedBatchSampler(BinaryLabelMatrix.from(labels), batchesPerEpoch, replacement, seed);
    }

    public B multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return multiLabelBalancedBatchSampler(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public B multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return multiLabelBalancedBatchSampler(labels, batchesPerEpoch, replacement, seed);
    }

    B multiLabelBalancedBatchSampler(
            BinaryLabelMatrix matrix,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        batchSize(batchSize);
        return multiLabelBalancedBatchSampler(matrix, batchesPerEpoch, replacement, seed);
    }

    private B multiLabelBalancedBatchSampler(
            BinaryLabelMatrix matrix,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this.samplerPlan = TensorSamplerPlan.multiLabelBalanced(matrix, batchesPerEpoch, replacement, seed);
        this.batchSampler = null;
        return self();
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

    public B collateFn(DataLoader.CollateFn collateFn) {
        this.collateFn = collateFn;
        return self();
    }

    public DataLoader.TensorDataLoader build() {
        return new DataLoader.TensorDataLoader(this);
    }

    private int[] tensorDatasetLengths(ToIntFunction<GradTensor[]> lengthExtractor) {
        Objects.requireNonNull(lengthExtractor, "lengthExtractor must not be null");
        int[] lengths = new int[dataset.size()];
        for (int i = 0; i < lengths.length; i++) {
            GradTensor[] sample = Objects.requireNonNull(dataset.get(i), "dataset sample must not be null");
            int length = lengthExtractor.applyAsInt(sample);
            if (length < 0) {
                throw new IllegalArgumentException("sequence lengths must be non-negative, got: " + length);
            }
            lengths[i] = length;
        }
        return lengths;
    }
}
