package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * DataLoader — batches, shuffles, and yields data in mini-batches.
 *
 * <p>Supports:
 * <ul>
 *   <li>Batching with configurable batch size</li>
 *   <li>Shuffling (on/off)</li>
 *   <li>Drop last incomplete batch</li>
 *   <li>Custom collate function</li>
 *   <li>Element mapping via {@link #map(Function)}</li>
 * </ul>
 *
 * <h3>Generic usage</h3>
 * <pre>{@code
 * Dataset<Integer> dataset = ...;
 * var loader = new DataLoader<>(dataset, 32);
 * for (List<Integer> batch : loader) {
 *     // process batch
 * }
 * }</pre>
 *
 * <h3>Tensor (training) usage</h3>
 * <pre>{@code
 * var tensorDataset = new TensorDataset(inputs, targets);
 * var loader = DataLoader.tensorBuilder(tensorDataset)
 *     .batchSize(32)
 *     .shuffle(true)
 *     .dropLast(true)
 *     .build();
 * for (List<GradTensor> batch : loader) {
 *     var input = batch.get(0);
 *     var target = batch.get(1);
 * }
 * }</pre>
 *
 * @param <T> the type of elements produced by this data loader
 * @author Gollek Team
 * @version 0.2.0
 */
public class DataLoader<T> implements Iterable<List<T>> {
    public static final float DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX = -100.0f;

    private final GenericDataLoaderRuntime<T> runtime;

    // ── Constructors ─────────────────────────────────────────────────────

    /**
     * Create a DataLoader with the given batch size (no shuffle, no drop-last).
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize) {
        this.runtime = new GenericDataLoaderRuntime<>(dataset, batchSize, false, false, null);
    }

    /**
     * Create a DataLoader with full configuration.
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     * @param shuffle   whether to shuffle indices each epoch
     * @param dropLast  whether to drop the last incomplete batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this.runtime = new GenericDataLoaderRuntime<>(dataset, batchSize, shuffle, dropLast, null);
    }

    /**
     * Create a DataLoader with a deterministic shuffle seed.
     *
     * <p>The seed is only used when {@code shuffle} is {@code true}. Reusing the
     * same loader or creating two loaders with the same seed produces the same
     * sample order, which is useful for reproducible examples and tests.</p>
     *
     * @param dataset     the dataset to load from
     * @param batchSize   number of samples per batch
     * @param shuffle     whether to shuffle indices each epoch
     * @param dropLast    whether to drop the last incomplete batch
     * @param shuffleSeed seed used for deterministic shuffling
     */
    public DataLoader(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            long shuffleSeed) {
        this.runtime = new GenericDataLoaderRuntime<>(dataset, batchSize, shuffle, dropLast, shuffleSeed);
    }

    private DataLoader(GenericDataLoaderRuntime<T> runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime must not be null");
    }

    // ── Iteration ────────────────────────────────────────────────────────

    @Override
    public Iterator<List<T>> iterator() {
        return runtime.iterator();
    }

    public Iterable<List<T>> epoch(long epoch) {
        return runtime.epoch(epoch);
    }

    /** Get number of batches per epoch. */
    public int numBatches() {
        return runtime.numBatches();
    }

    public int size() {
        return runtime.size();
    }

    public int sampleCount() {
        return runtime.sampleCount();
    }

    public boolean sampled() {
        return runtime.sampled();
    }

    public int batchSize() {
        return runtime.batchSize();
    }

    public boolean shuffle() {
        return runtime.shuffle();
    }

    public boolean dropLast() {
        return runtime.dropLast();
    }

    public OptionalLong shuffleSeed() {
        return runtime.shuffleSeed();
    }

    public boolean reshuffleEachEpoch() {
        return runtime.reshuffleEachEpoch();
    }

    public long initialEpoch() {
        return runtime.initialEpoch();
    }

    public DataLoaderPlan plan() {
        return runtime.plan("generic");
    }

    // ── Mapping ──────────────────────────────────────────────────────────

    /**
     * Create a new DataLoader that applies a mapping function to each element.
     *
     * <pre>{@code
     * DataLoader<String> mapped = new DataLoader<>(intDataset, 5)
     *     .map(i -> "val_" + i);
     * }</pre>
     *
     * @param mapper function to apply to each element
     * @param <R>    the output type
     * @return a new DataLoader producing mapped elements
     */
    public <R> DataLoader<R> map(Function<? super T, ? extends R> mapper) {
        return new DataLoader<>(runtime.map(mapper));
    }

    public <B> CollatingDataLoader<T, B> collate(Function<? super List<T>, ? extends B> collateFn) {
        return new CollatingDataLoader<>(runtime, collateFn);
    }

    /**
     * Wrap this loader with bounded asynchronous prefetching.
     *
     * <p>Use try-with-resources when a loop may exit early so the worker thread can
     * be interrupted immediately.</p>
     */
    public PrefetchingIterable<List<T>> prefetch() {
        return prefetch(this);
    }

    /**
     * Wrap this loader with bounded asynchronous prefetching.
     *
     * <p>Use try-with-resources when a loop may exit early so the worker thread can
     * be interrupted immediately.</p>
     */
    public PrefetchingIterable<List<T>> prefetch(int bufferSize) {
        return prefetch(this, bufferSize);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TensorDataLoader — GradTensor-specific builder with collation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a fluent builder for a plain generic {@link DataLoader}.
     *
     * <p>This keeps constructor-based usage intact while giving training and
     * data-preparation code the same readable configuration style as tensor
     * loaders.</p>
     *
     * @param dataset the dataset to load from
     * @param <T>     dataset element type
     * @return a new generic DataLoader builder
     */
    public static <T> Builder<T> builder(Dataset<? extends T> dataset) {
        return new Builder<>(dataset);
    }

    /**
     * Create a builder for a tensor-specific DataLoader that supports
     * custom collation and stacking of {@link GradTensor} batches.
     *
     * @param dataset the tensor dataset
     * @return a new builder
     */
    public static TensorBuilder tensorBuilder(TensorDatasetAdapter dataset) {
        return new TensorBuilder(dataset);
    }

    public static <B> PrefetchingIterable<B> prefetch(Iterable<? extends B> source, int bufferSize) {
        return PrefetchingIterable.of(source, bufferSize);
    }

    public static <B> PrefetchingIterable<B> prefetch(Iterable<? extends B> source) {
        return PrefetchingIterable.of(source);
    }

    public static CollateFn defaultTensorCollate() {
        return TensorCollators.defaultPairCollate();
    }

    /**
     * Collate generic {@link Dataset.Sample} records into a tensor training batch.
     *
     * <p>This is useful for datasets such as tokenized text or image datasets that
     * already expose one input tensor and one label tensor per sample.</p>
     */
    public static Function<List<Dataset.Sample>, Batch> sampleBatchCollate() {
        return TensorCollators.sampleBatchCollate();
    }

    /**
     * Collate generic tensor pairs into a tensor training batch.
     *
     * <p>The pair's left tensor becomes {@link Batch#inputs()} and the right tensor
     * becomes {@link Batch#labels()}.</p>
     */
    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, Batch> tensorPairBatchCollate() {
        return TensorCollators.tensorPairBatchCollate();
    }

    /**
     * Collate variable-length {@link Dataset.Sample} records by padding inputs and
     * labels along their first dimension with zero values.
     */
    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate() {
        return paddedSampleBatchCollate(0.0f, 0.0f);
    }

    /**
     * Collate variable-length {@link Dataset.Sample} records by padding inputs and
     * labels along their first dimension with the same pad value.
     */
    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate(float padValue) {
        return paddedSampleBatchCollate(padValue, padValue);
    }

    /**
     * Collate variable-length {@link Dataset.Sample} records by padding inputs and
     * labels along their first dimension.
     */
    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate(
            float inputPadValue,
            float labelPadValue) {
        return TensorCollators.paddedSampleBatchCollate(inputPadValue, labelPadValue);
    }

    /**
     * Collate causal language-modeling samples by padding inputs with the tokenizer pad
     * token id and padding labels with {@link #DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX}.
     *
     * <p>The label ignore index prevents padded label positions from contributing to
     * next-token loss functions that honor ignore labels.</p>
     */
    public static Function<List<Dataset.Sample>, PaddedBatch> causalLanguageModelingBatchCollate(int padTokenId) {
        return causalLanguageModelingBatchCollate((float) padTokenId, DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX);
    }

    /**
     * Collate causal language-modeling samples with explicit input pad and label ignore values.
     */
    public static Function<List<Dataset.Sample>, PaddedBatch> causalLanguageModelingBatchCollate(
            float inputPadValue,
            float labelIgnoreIndex) {
        return paddedSampleBatchCollate(inputPadValue, labelIgnoreIndex);
    }

    /**
     * Collate variable-length tensor pairs by padding inputs and labels along their
     * first dimension with zero values.
     */
    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, PaddedBatch> paddedTensorPairBatchCollate() {
        return paddedTensorPairBatchCollate(0.0f, 0.0f);
    }

    /**
     * Collate variable-length tensor pairs by padding inputs and labels along their
     * first dimension with the same pad value.
     */
    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, PaddedBatch> paddedTensorPairBatchCollate(
            float padValue) {
        return paddedTensorPairBatchCollate(padValue, padValue);
    }

    /**
     * Collate variable-length tensor pairs by padding inputs and labels along their
     * first dimension.
     */
    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, PaddedBatch> paddedTensorPairBatchCollate(
            float inputPadValue,
            float labelPadValue) {
        return TensorCollators.paddedTensorPairBatchCollate(inputPadValue, labelPadValue);
    }

    public static PaddingEfficiencyReport paddingEfficiency(Iterable<PaddedBatch> batches) {
        return PaddingDiagnostics.paddingEfficiency(batches);
    }

    public static Dataset<Dataset.Sample> causalLanguageModelingDataset(int[] tokenIds, int sequenceLength) {
        return LanguageModelingDatasets.causalNextToken(tokenIds, sequenceLength);
    }

    public static Dataset<Dataset.Sample> causalLanguageModelingDataset(
            int[] tokenIds,
            int sequenceLength,
            int stride) {
        return LanguageModelingDatasets.causalNextToken(tokenIds, sequenceLength, stride);
    }

    public static Dataset<Dataset.Sample> causalLanguageModelingDataset(long[] tokenIds, int sequenceLength) {
        return LanguageModelingDatasets.causalNextToken(tokenIds, sequenceLength);
    }

    public static Dataset<Dataset.Sample> causalLanguageModelingDataset(
            long[] tokenIds,
            int sequenceLength,
            int stride) {
        return LanguageModelingDatasets.causalNextToken(tokenIds, sequenceLength, stride);
    }

    public static Dataset<Dataset.Sample> packedCausalLanguageModelingDataset(
            int[][] tokenDocuments,
            int eosTokenId,
            int sequenceLength) {
        return LanguageModelingDatasets.packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength);
    }

    public static Dataset<Dataset.Sample> packedCausalLanguageModelingDataset(
            int[][] tokenDocuments,
            int eosTokenId,
            int sequenceLength,
            int stride) {
        return LanguageModelingDatasets.packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength, stride);
    }

    public static Dataset<Dataset.Sample> packedCausalLanguageModelingDataset(
            long[][] tokenDocuments,
            long eosTokenId,
            int sequenceLength) {
        return LanguageModelingDatasets.packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength);
    }

    public static Dataset<Dataset.Sample> packedCausalLanguageModelingDataset(
            long[][] tokenDocuments,
            long eosTokenId,
            int sequenceLength,
            int stride) {
        return LanguageModelingDatasets.packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength, stride);
    }

    public static int sequenceLength(GradTensor tensor) {
        return SequenceLengthSupport.sequenceLength(tensor);
    }

    public static <T> int[] sequenceLengths(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor) {
        return SequenceLengthSupport.lengths(dataset, lengthExtractor);
    }

    public static ToIntFunction<Dataset.Sample> sampleInputLength() {
        return SequenceLengthSupport.sampleInputLength();
    }

    public static ToIntFunction<Dataset.Sample> sampleLabelLength() {
        return SequenceLengthSupport.sampleLabelLength();
    }

    public static ToIntFunction<Dataset.Pair<GradTensor, GradTensor>> tensorPairInputLength() {
        return SequenceLengthSupport.tensorPairInputLength();
    }

    public static ToIntFunction<Dataset.Pair<GradTensor, GradTensor>> tensorPairLabelLength() {
        return SequenceLengthSupport.tensorPairLabelLength();
    }

    public static int[] sampleInputLengths(Dataset<? extends Dataset.Sample> dataset) {
        return sequenceLengths(dataset, sampleInputLength());
    }

    public static int[] sampleLabelLengths(Dataset<? extends Dataset.Sample> dataset) {
        return sequenceLengths(dataset, sampleLabelLength());
    }

    public static int[] tensorPairInputLengths(
            Dataset<? extends Dataset.Pair<GradTensor, GradTensor>> dataset) {
        return sequenceLengths(dataset, tensorPairInputLength());
    }

    public static int[] tensorPairLabelLengths(
            Dataset<? extends Dataset.Pair<GradTensor, GradTensor>> dataset) {
        return sequenceLengths(dataset, tensorPairLabelLength());
    }

    public static WeightedRandomSampler weightedRandomSampler(
            float[] sampleWeights,
            int numSamples,
            boolean replacement,
            long seed) {
        return SamplerFactories.weightedRandom(sampleWeights, numSamples, replacement, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            long seed) {
        return SamplerFactories.lengthBucket(lengths, batchSize, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        return SamplerFactories.lengthBucket(dataset, lengthExtractor, batchSize, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucket(lengths, batchSize, dropLast, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucket(dataset, lengthExtractor, batchSize, dropLast, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucket(
                lengths,
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucket(
                dataset,
                lengthExtractor,
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        return SamplerFactories.tokenBudget(lengths, maxTokens, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return SamplerFactories.tokenBudget(dataset, lengthExtractor, maxTokens, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            long seed) {
        return SamplerFactories.tokenBudget(lengths, maxTokens, maxExamples, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return SamplerFactories.tokenBudget(dataset, lengthExtractor, maxTokens, maxExamples, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return SamplerFactories.tokenBudget(
                lengths,
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return SamplerFactories.tokenBudget(
                dataset,
                lengthExtractor,
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public static ClassBalancedBatchSampler classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return SamplerFactories.classBalanced(labels, batchSize, batchesPerEpoch, seed);
    }

    public static ClassBalancedBatchSampler classBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return SamplerFactories.classBalanced(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static StratifiedBatchSampler stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return SamplerFactories.stratified(labels, batchSize, batchesPerEpoch, seed);
    }

    public static StratifiedBatchSampler stratifiedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return SamplerFactories.stratified(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, seed);
    }

    public static MultiLabelBalancedBatchSampler multiLabelBalancedBatchSampler(
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return SamplerFactories.multiLabelBalanced(labels, batchSize, batchesPerEpoch, replacement, seed);
    }

    public static SequentialSampler sequentialSampler() {
        return SamplerFactories.sequential();
    }

    public static RandomSampler randomSampler(long seed) {
        return SamplerFactories.random(seed);
    }

    public static RandomSampler randomSampler(int numSamples, boolean replacement, long seed) {
        return SamplerFactories.random(numSamples, replacement, seed);
    }

    public static SubsetSampler subsetSampler(int... indices) {
        return SamplerFactories.subset(indices);
    }

    public static DistributedSampler distributedSampler(int numReplicas, int rank) {
        return SamplerFactories.distributed(numReplicas, rank);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return SamplerFactories.distributed(numReplicas, rank, shuffle, dropLast, seed);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return SamplerFactories.distributed(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }

    public static TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return TensorLabelEncoding.tensorDataset(inputs, labels);
    }

    public static GradTensor classLabels(int... labels) {
        return TensorLabelEncoding.classLabels(labels);
    }

    public static float[] classWeights(int... labels) {
        return LabelWeighting.classWeights(labels);
    }

    public static float[] classWeightsFor(int numClasses, int... labels) {
        return LabelWeighting.classWeightsFor(numClasses, labels);
    }

    public static float[] classBalancedSampleWeights(int... labels) {
        return LabelWeighting.classBalancedSampleWeights(labels);
    }

    public static float[] binaryBalancedSampleWeights(int... labels) {
        return LabelWeighting.binaryBalancedSampleWeights(labels);
    }

    public static float[] multiLabelBalancedSampleWeights(int[][] labels) {
        return LabelWeighting.multiLabelBalancedSampleWeights(labels);
    }

    public static float[] multiLabelBalancedSampleWeights(boolean[][] labels) {
        return LabelWeighting.multiLabelBalancedSampleWeights(labels);
    }

    public static float[] multiLabelBalancedSampleWeights(float[][] labels) {
        return LabelWeighting.multiLabelBalancedSampleWeights(labels);
    }

    public static GradTensor binaryLabels(int... labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static GradTensor binaryLabels(boolean... labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static GradTensor binaryLabels(float... labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static GradTensor binaryLabels(int[][] labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static GradTensor binaryLabels(boolean[][] labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static GradTensor binaryLabels(float[][] labels) {
        return TensorLabelEncoding.binaryLabels(labels);
    }

    public static float binaryPositiveWeight(int... labels) {
        return LabelWeighting.binaryPositiveWeight(labels);
    }

    public static float binaryPositiveWeight(boolean... labels) {
        return LabelWeighting.binaryPositiveWeight(labels);
    }

    public static float binaryPositiveWeight(float... labels) {
        return LabelWeighting.binaryPositiveWeight(labels);
    }

    public static float[] multiLabelPositiveWeights(int[][] labels) {
        return LabelWeighting.positiveWeights(labels);
    }

    public static float[] multiLabelPositiveWeights(boolean[][] labels) {
        return LabelWeighting.positiveWeights(labels);
    }

    public static float[] multiLabelPositiveWeights(float[][] labels) {
        return LabelWeighting.positiveWeights(labels);
    }

    public static TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
        return TensorLabelEncoding.classificationDataset(inputs, labels);
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
        return TensorLabelEncoding.binaryDataset(inputs, labels);
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
        return TensorLabelEncoding.binaryDataset(inputs, labels);
    }

    public static TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
        return TensorLabelEncoding.binaryDataset(inputs, labels);
    }

    public static TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
        return TensorLabelEncoding.binaryDataset(inputs, labels);
    }

    public static TensorDataLoader tensors(GradTensor inputs, GradTensor labels, int batchSize) {
        return TensorLoaderFactories.tensors(inputs, labels, batchSize);
    }

    public static TensorDataLoader classification(GradTensor inputs, int[] labels, int batchSize) {
        return TensorLoaderFactories.classification(inputs, labels, batchSize);
    }

    public static TensorDataLoader classBalancedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return TensorLoaderFactories.classBalancedClassification(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static TensorDataLoader classBalancedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.classBalancedClassification(inputs, labels, batchSize, batchesPerEpoch,
                replacement, seed);
    }

    public static TensorDataLoader stratifiedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return TensorLoaderFactories.stratifiedClassification(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static TensorDataLoader stratifiedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.stratifiedClassification(inputs, labels, batchSize, batchesPerEpoch,
                replacement, seed);
    }

    public static TensorDataLoader classWeightedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return TensorLoaderFactories.classWeightedClassification(inputs, labels, batchSize, numSamples, seed);
    }

    public static TensorDataLoader classWeightedClassification(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.classWeightedClassification(inputs, labels, batchSize, numSamples,
                replacement, seed);
    }

    public static TensorDataLoader binary(GradTensor inputs, int[] labels, int batchSize) {
        return TensorLoaderFactories.binary(inputs, labels, batchSize);
    }

    public static TensorDataLoader weightedBinary(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return TensorLoaderFactories.weightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static TensorDataLoader weightedBinary(
            GradTensor inputs,
            int[] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.weightedBinary(inputs, labels, batchSize, numSamples, replacement, seed);
    }

    public static TensorDataLoader binary(GradTensor inputs, int[][] labels, int batchSize) {
        return TensorLoaderFactories.binary(inputs, labels, batchSize);
    }

    public static TensorDataLoader binary(GradTensor inputs, boolean[][] labels, int batchSize) {
        return TensorLoaderFactories.binary(inputs, labels, batchSize);
    }

    public static TensorDataLoader binary(GradTensor inputs, float[][] labels, int batchSize) {
        return TensorLoaderFactories.binary(inputs, labels, batchSize);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples,
                replacement, seed);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples,
                replacement, seed);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples, seed);
    }

    public static TensorDataLoader multiLabelWeightedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int numSamples,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelWeightedBinary(inputs, labels, batchSize, numSamples,
                replacement, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            int[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch,
                replacement, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            boolean[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch,
                replacement, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch, seed);
    }

    public static TensorDataLoader multiLabelBalancedBinary(
            GradTensor inputs,
            float[][] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        return TensorLoaderFactories.multiLabelBalancedBinary(inputs, labels, batchSize, batchesPerEpoch,
                replacement, seed);
    }

    public static ClassificationDistributionReport classificationDistribution(
            TensorDataLoader loader,
            int numClasses) {
        return DistributionDiagnostics.classificationDistribution(loader, numClasses);
    }

    public static ClassificationDistributionReport classificationDistribution(
            Iterable<Batch> loader,
            int numClasses) {
        return DistributionDiagnostics.classificationDistribution(loader, numClasses);
    }

    public static ClassificationDistributionReport classificationDistribution(TensorDataLoader loader) {
        return DistributionDiagnostics.classificationDistribution(loader);
    }

    public static ClassificationDistributionReport classificationDistribution(Iterable<Batch> loader) {
        return DistributionDiagnostics.classificationDistribution(loader);
    }

    public static ClassificationDistributionReport binaryDistribution(TensorDataLoader loader) {
        return classificationDistribution(loader, 2);
    }

    public static ClassificationDistributionReport binaryDistribution(Iterable<Batch> loader) {
        return classificationDistribution(loader, 2);
    }

    public static MultiLabelDistributionReport multiLabelDistribution(
            TensorDataLoader loader,
            int labelCount) {
        return DistributionDiagnostics.multiLabelDistribution(loader, labelCount);
    }

    public static MultiLabelDistributionReport multiLabelDistribution(
            Iterable<Batch> loader,
            int labelCount) {
        return DistributionDiagnostics.multiLabelDistribution(loader, labelCount);
    }

    public static MultiLabelDistributionReport multiLabelDistribution(TensorDataLoader loader) {
        return DistributionDiagnostics.multiLabelDistribution(loader);
    }

    public static MultiLabelDistributionReport multiLabelDistribution(Iterable<Batch> loader) {
        return DistributionDiagnostics.multiLabelDistribution(loader);
    }

    public static ClassificationDistributionDriftReport classificationDistributionDrift(
            ClassificationDistributionReport reference,
            ClassificationDistributionReport candidate) {
        return DistributionDiagnostics.classificationDistributionDrift(reference, candidate);
    }

    public static MultiLabelDistributionDriftReport multiLabelDistributionDrift(
            MultiLabelDistributionReport reference,
            MultiLabelDistributionReport candidate) {
        return DistributionDiagnostics.multiLabelDistributionDrift(reference, candidate);
    }

    public static TensorDatasetSplit split(GradTensor inputs, GradTensor labels, double trainFraction, long seed) {
        return TensorSplitFactories.split(inputs, labels, trainFraction, seed);
    }

    public static List<TensorDatasetFold> kFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            long seed) {
        return TensorSplitFactories.kFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> repeatedKFold(
            GradTensor inputs,
            GradTensor labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.repeatedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<TensorDatasetFold> repeatedKFold(
            TensorDataset dataset,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.repeatedKFold(dataset, folds, repeats, seed);
    }

    public static List<TensorDatasetFold> groupKFold(
            GradTensor inputs,
            GradTensor labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.groupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<TensorDatasetFold> groupKFold(
            TensorDataset dataset,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.groupKFold(dataset, groups, folds, seed);
    }

    public static List<TensorDatasetFold> stratifiedGroupKFold(
            TensorDataset dataset,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.stratifiedGroupKFold(dataset, labels, groups, folds, seed);
    }

    public static List<TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits) {
        return TensorSplitFactories.timeSeriesSplit(inputs, labels, splits);
    }

    public static List<TensorDatasetFold> timeSeriesSplit(
            GradTensor inputs,
            GradTensor labels,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return TensorSplitFactories.timeSeriesSplit(inputs, labels, splits, validationSize, gap, maxTrainSize);
    }

    public static List<TensorDatasetFold> timeSeriesSplit(
            TensorDataset dataset,
            int splits) {
        return TensorSplitFactories.timeSeriesSplit(dataset, splits);
    }

    public static List<TensorDatasetFold> timeSeriesSplit(
            TensorDataset dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        return TensorSplitFactories.timeSeriesSplit(dataset, splits, validationSize, gap, maxTrainSize);
    }

    public static TensorDatasetThreeWaySplit split(
            GradTensor inputs,
            GradTensor labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.split(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.classificationSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.classificationSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.classificationStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.classificationStratifiedSplit(
                inputs,
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    public static List<TensorDatasetFold> classificationStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return TensorSplitFactories.classificationStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> classificationRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.classificationRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<TensorDatasetFold> classificationStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.classificationStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.binaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.binaryStratifiedSplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static List<TensorDatasetFold> binaryStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            long seed) {
        return TensorSplitFactories.binaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> binaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[] labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.binaryRepeatedStratifiedKFold(inputs, labels, folds, repeats, seed);
    }

    public static List<TensorDatasetFold> binaryStratifiedGroupKFold(
            GradTensor inputs,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.binaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(
                inputs,
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            int[][] labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedKFold(
                inputs,
                labels,
                folds,
                repeats,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            int[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedGroupKFold(
                inputs,
                labels,
                groups,
                folds,
                repeats,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(
                inputs,
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            boolean[][] labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedKFold(
                inputs,
                labels,
                folds,
                repeats,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            boolean[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedGroupKFold(
                inputs,
                labels,
                groups,
                folds,
                repeats,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.binarySplit(inputs, labels, trainFraction, validationFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(inputs, labels, trainFraction, seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedSplit(
                inputs,
                labels,
                trainFraction,
                validationFraction,
                seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                seed);
    }

    public static TensorDatasetThreeWaySplit multiLabelBinaryStratifiedGroupSplit(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupSplit(
                inputs,
                labels,
                groups,
                trainFraction,
                validationFraction,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedKFold(inputs, labels, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedKFold(
            GradTensor inputs,
            float[][] labels,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedKFold(
                inputs,
                labels,
                folds,
                repeats,
                seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryStratifiedGroupKFold(inputs, labels, groups, folds, seed);
    }

    public static List<TensorDatasetFold> multiLabelBinaryRepeatedStratifiedGroupKFold(
            GradTensor inputs,
            float[][] labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return TensorSplitFactories.multiLabelBinaryRepeatedStratifiedGroupKFold(
                inputs,
                labels,
                groups,
                folds,
                repeats,
                seed);
    }

    /**
     * Adapter interface for tensor datasets that return arrays of GradTensors.
     * This is the specialized interface for training workflows where each
     * sample is a tuple of tensors (e.g., [input, target]).
     */
    public interface TensorDatasetAdapter {
        /** Total number of samples. */
        int size();

        /** Get number of samples for given indices. */
        default int size(List<Integer> indices) {
            return indices.size();
        }

        /** Get sample at given index. Returns array of tensors (e.g., [input, target]). */
        GradTensor[] get(int index);
    }

    /**
     * Simple dataset from arrays of GradTensor.
     */
    public static class TensorDataset extends TensorDatasetSupport {
        public TensorDataset(GradTensor[]... samples) {
            super(samples);
        }

        TensorDataset(List<GradTensor[]> samples) {
            super(samples);
        }

        /**
         * Create from separate input and target tensors.
         * Each input[i] and target[i] form a sample pair.
         */
        public TensorDataset(GradTensor inputs, GradTensor targets) {
            super(inputs, targets);
        }

        @Override
        TensorDataset self() {
            return this;
        }
    }

    public record TensorDatasetSplit(TensorDataset train, TensorDataset validation)
            implements TrainValidationTensorDatasetView {
        public TensorDatasetSplit {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }
    }

    public record TensorDatasetFold(int foldIndex, int foldCount, TensorDataset train, TensorDataset validation)
            implements TrainValidationTensorDatasetView {
        public TensorDatasetFold {
            if (foldIndex < 0) {
                throw new IllegalArgumentException("foldIndex must be non-negative, got: " + foldIndex);
            }
            if (foldCount <= foldIndex) {
                throw new IllegalArgumentException(
                        "foldCount must be greater than foldIndex, got foldIndex="
                                + foldIndex + ", foldCount=" + foldCount);
            }
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }
    }

    public record TensorDatasetThreeWaySplit(TensorDataset train, TensorDataset validation, TensorDataset test)
            implements ThreeWayTensorDatasetView {
        public TensorDatasetThreeWaySplit {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
            Objects.requireNonNull(test, "test must not be null");
        }
    }

    /**
     * Builder for plain generic DataLoader instances.
     */
    public static class Builder<T> extends GenericDataLoaderBuilderSupport<T, Builder<T>> {
        Builder(Dataset<? extends T> dataset) {
            super(dataset);
        }

        @Override
        protected Builder<T> self() {
            return this;
        }

        public DataLoader<T> build() {
            return new DataLoader<>(runtime());
        }

        public <B> CollatingDataLoader<T, B> collate(Function<? super List<T>, ? extends B> collateFn) {
            return new CollatingDataLoader<>(runtime(), collateFn);
        }
    }

    /**
     * Generic DataLoader variant that converts each list batch into a user-defined
     * batch object, such as a padded tensor batch or a domain-specific record.
     */
    public static class CollatingDataLoader<T, B> implements Iterable<B> {
        private final CollatingDataLoaderRuntime<T, B> runtime;

        private CollatingDataLoader(
                GenericDataLoaderRuntime<T> source,
                Function<? super List<T>, ? extends B> collateFn) {
            this.runtime = new CollatingDataLoaderRuntime<>(source, collateFn);
        }

        @Override
        public Iterator<B> iterator() {
            return runtime.iterator();
        }

        public Iterable<B> epoch(long epoch) {
            return runtime.epoch(epoch);
        }

        public int numBatches() {
            return runtime.numBatches();
        }

        public int size() {
            return runtime.size();
        }

        public int sampleCount() {
            return runtime.sampleCount();
        }

        public boolean sampled() {
            return runtime.sampled();
        }

        public int batchSize() {
            return runtime.batchSize();
        }

        public boolean shuffle() {
            return runtime.shuffle();
        }

        public boolean dropLast() {
            return runtime.dropLast();
        }

        public OptionalLong shuffleSeed() {
            return runtime.shuffleSeed();
        }

        public boolean reshuffleEachEpoch() {
            return runtime.reshuffleEachEpoch();
        }

        public long initialEpoch() {
            return runtime.initialEpoch();
        }

        public DataLoaderPlan plan() {
            return runtime.plan();
        }

        public PrefetchingIterable<B> prefetch(int bufferSize) {
            return DataLoader.prefetch(this, bufferSize);
        }

        public PrefetchingIterable<B> prefetch() {
            return DataLoader.prefetch(this);
        }
    }

    /**
     * Builder for tensor-specific DataLoader with collation support.
     */
    public static class TensorBuilder extends TensorDataLoaderBuilderSupport<TensorBuilder> {
        TensorBuilder(TensorDatasetAdapter dataset) {
            super(dataset);
        }

        @Override
        protected TensorBuilder self() {
            return this;
        }
    }

    /**
     * Tensor-specific DataLoader that handles collation and stacking
     * of {@link GradTensor} batches.
     */
    public static class TensorDataLoader implements Iterable<Batch> {
        private final TensorDataLoaderRuntime runtime;

        TensorDataLoader(TensorDataLoaderBuilderSupport<?> builder) {
            this.runtime = new TensorDataLoaderRuntime(builder);
        }

        @Override
        public Iterator<Batch> iterator() {
            return runtime.iterator();
        }

        public Iterable<Batch> epoch(long epoch) {
            return runtime.epoch(epoch);
        }

        /** Get number of batches per epoch. */
        public int numBatches() {
            return runtime.numBatches();
        }

        public int size() {
            return runtime.size();
        }

        public int batchSize() {
            return runtime.batchSize();
        }

        public int sampleCount() {
            return runtime.sampleCount();
        }

        public boolean sampled() {
            return runtime.sampled();
        }

        public boolean shuffle() {
            return runtime.shuffle();
        }

        public boolean dropLast() {
            return runtime.dropLast();
        }

        public boolean reshuffleEachEpoch() {
            return runtime.reshuffleEachEpoch();
        }

        public long initialEpoch() {
            return runtime.initialEpoch();
        }

        public DataLoaderPlan plan() {
            return runtime.plan("tensor");
        }

        public PrefetchingIterable<Batch> prefetch(int bufferSize) {
            return DataLoader.prefetch(this, bufferSize);
        }

        public PrefetchingIterable<Batch> prefetch() {
            return DataLoader.prefetch(this);
        }
    }

    /**
     * Collate function interface for tensor datasets.
     */
    @FunctionalInterface
    public interface CollateFn {
        Batch collate(List<Integer> indices, TensorDatasetAdapter dataset);
    }

    public record ClassificationDistributionReport(
            int sampleCount,
            int batchCount,
            int numClasses,
            int[] classCounts,
            List<int[]> batchClassCounts) implements ClassificationDistributionView {
        public ClassificationDistributionReport {
            var state = DistributionReportStates.classificationReport(
                    sampleCount,
                    batchCount,
                    numClasses,
                    classCounts,
                    batchClassCounts);
            classCounts = state.classCounts();
            batchClassCounts = state.batchClassCounts();
        }

        @Override
        public int[] classCounts() {
            return Arrays.copyOf(classCounts, classCounts.length);
        }

        @Override
        public List<int[]> batchClassCounts() {
            return DistributionDiagnostics.copyCountVectors(
                    batchClassCounts,
                    "batchClassCounts",
                    numClasses);
        }
    }

    public record MultiLabelDistributionReport(
            int sampleCount,
            int batchCount,
            int labelCount,
            int[] positiveCounts,
            int[] negativeCounts,
            List<int[]> batchPositiveCounts) implements MultiLabelDistributionView {
        public MultiLabelDistributionReport {
            var state = DistributionReportStates.multiLabelReport(
                    sampleCount,
                    batchCount,
                    labelCount,
                    positiveCounts,
                    negativeCounts,
                    batchPositiveCounts);
            positiveCounts = state.positiveCounts();
            negativeCounts = state.negativeCounts();
            batchPositiveCounts = state.batchPositiveCounts();
        }

        @Override
        public int[] positiveCounts() {
            return Arrays.copyOf(positiveCounts, positiveCounts.length);
        }

        @Override
        public int[] negativeCounts() {
            return Arrays.copyOf(negativeCounts, negativeCounts.length);
        }

        @Override
        public List<int[]> batchPositiveCounts() {
            return DistributionDiagnostics.copyCountVectors(batchPositiveCounts, "batchPositiveCounts", labelCount);
        }
    }

    public record ClassificationDistributionDriftReport(
            int numClasses,
            double[] referenceFractions,
            double[] candidateFractions,
            double[] fractionDeltas,
            List<Integer> referenceMissingClasses,
            List<Integer> candidateMissingClasses) implements ClassificationDistributionDriftView {
        public ClassificationDistributionDriftReport {
            var state = DistributionReportStates.classificationDrift(
                    numClasses,
                    referenceFractions,
                    candidateFractions,
                    fractionDeltas,
                    referenceMissingClasses,
                    candidateMissingClasses);
            referenceFractions = state.referenceFractions();
            candidateFractions = state.candidateFractions();
            fractionDeltas = state.fractionDeltas();
            referenceMissingClasses = state.referenceMissingClasses();
            candidateMissingClasses = state.candidateMissingClasses();
        }

        @Override
        public double[] referenceFractions() {
            return Arrays.copyOf(referenceFractions, referenceFractions.length);
        }

        @Override
        public double[] candidateFractions() {
            return Arrays.copyOf(candidateFractions, candidateFractions.length);
        }

        @Override
        public double[] fractionDeltas() {
            return Arrays.copyOf(fractionDeltas, fractionDeltas.length);
        }
    }

    public record MultiLabelDistributionDriftReport(
            int labelCount,
            double[] referencePositiveFractions,
            double[] candidatePositiveFractions,
            double[] positiveFractionDeltas,
            double referenceLabelCardinality,
            double candidateLabelCardinality,
            List<Integer> referenceZeroPositiveLabels,
            List<Integer> candidateZeroPositiveLabels) implements MultiLabelDistributionDriftView {
        public MultiLabelDistributionDriftReport {
            var state = DistributionReportStates.multiLabelDrift(
                    labelCount,
                    referencePositiveFractions,
                    candidatePositiveFractions,
                    positiveFractionDeltas,
                    referenceLabelCardinality,
                    candidateLabelCardinality,
                    referenceZeroPositiveLabels,
                    candidateZeroPositiveLabels);
            referencePositiveFractions = state.referencePositiveFractions();
            candidatePositiveFractions = state.candidatePositiveFractions();
            positiveFractionDeltas = state.positiveFractionDeltas();
            referenceZeroPositiveLabels = state.referenceZeroPositiveLabels();
            candidateZeroPositiveLabels = state.candidateZeroPositiveLabels();
        }

        @Override
        public double[] referencePositiveFractions() {
            return Arrays.copyOf(referencePositiveFractions, referencePositiveFractions.length);
        }

        @Override
        public double[] candidatePositiveFractions() {
            return Arrays.copyOf(candidatePositiveFractions, candidatePositiveFractions.length);
        }

        @Override
        public double[] positiveFractionDeltas() {
            return Arrays.copyOf(positiveFractionDeltas, positiveFractionDeltas.length);
        }
    }

    /**
     * A record representing a mini-batch of data.
     *
     * @param inputs input tensor (usually [N, ...])
     * @param labels target/label tensor (usually [N, ...])
     */
    public record Batch(GradTensor inputs, GradTensor labels) {
        public Batch {
            inputs = BatchSupport.requireTensor("inputs", inputs);
            labels = BatchSupport.requireTensor("labels", labels);
            BatchSupport.requireCompatibleBatchDimensions(inputs, labels);
        }
    }

    /**
     * A padded mini-batch for variable-length sequence-style tensors.
     *
     * @param inputs       input tensor shaped {@code [batch, maxInputLength, ...]}
     * @param labels       label tensor shaped {@code [batch, maxLabelLength, ...]}
     * @param inputMask    mask shaped {@code [batch, maxInputLength]} with 1 for real tokens
     * @param labelMask    mask shaped {@code [batch, maxLabelLength]} with 1 for real labels
     * @param inputLengths original first-dimension lengths for each input sample
     * @param labelLengths original first-dimension lengths for each label sample
     */
    public record PaddedBatch(
            GradTensor inputs,
            GradTensor labels,
            GradTensor inputMask,
            GradTensor labelMask,
            int[] inputLengths,
            int[] labelLengths) {
        public PaddedBatch {
            inputs = PaddedBatchSupport.requireTensor("inputs", inputs);
            labels = PaddedBatchSupport.requireTensor("labels", labels);
            inputMask = PaddedBatchSupport.requireTensor("inputMask", inputMask);
            labelMask = PaddedBatchSupport.requireTensor("labelMask", labelMask);
            inputLengths = PaddedBatchSupport.copyLengths("inputLengths", inputLengths);
            labelLengths = PaddedBatchSupport.copyLengths("labelLengths", labelLengths);
            PaddedBatchSupport.validate("inputs", inputs, inputMask, inputLengths);
            PaddedBatchSupport.validate("labels", labels, labelMask, labelLengths);
            PaddedBatchSupport.validateAlignedBatchSize(inputs, labels);
        }

        public Batch batch() {
            return new Batch(inputs, labels);
        }

        public PaddingStats inputPaddingStats() {
            return PaddedBatchSupport.stats("inputs", inputs, inputLengths);
        }

        public PaddingStats labelPaddingStats() {
            return PaddedBatchSupport.stats("labels", labels, labelLengths);
        }

        public PaddingEfficiencyReport paddingEfficiency() {
            return new PaddingEfficiencyReport(inputPaddingStats(), labelPaddingStats());
        }

        @Override
        public int[] inputLengths() {
            return inputLengths.clone();
        }

        @Override
        public int[] labelLengths() {
            return labelLengths.clone();
        }
    }
}
