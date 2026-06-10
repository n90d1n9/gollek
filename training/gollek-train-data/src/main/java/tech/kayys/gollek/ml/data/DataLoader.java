package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * DataLoader that batches, shuffles, and yields data in mini-batches.
 *
 * @param <T> type of elements produced by this loader
 */
public class DataLoader<T> implements Iterable<List<T>> {
    public static final float DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX = -100.0f;

    private final GenericDataLoaderRuntime<T> runtime;

    public DataLoader(Dataset<? extends T> dataset, int batchSize) {
        this(dataset, batchSize, false, false);
    }

    public DataLoader(Dataset<? extends T> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this(dataset, batchSize, shuffle, dropLast, null);
    }

    public DataLoader(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            IndexSampler sampler) {
        this(dataset, batchSize, shuffle, dropLast, sampler, null);
    }

    public DataLoader(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            IndexSampler sampler,
            BatchSampler batchSampler) {
        this(new GenericDataLoaderRuntime<>(dataset, batchSize, shuffle, dropLast, sampler, batchSampler));
    }

    private DataLoader(GenericDataLoaderRuntime<T> runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime must not be null");
    }

    @Override
    public Iterator<List<T>> iterator() {
        return runtime.iterator();
    }

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

    public <R> DataLoader<R> map(Function<? super T, ? extends R> mapper) {
        return new DataLoader<>(runtime.map(mapper));
    }

    public <B> CollatingDataLoader<T, B> collate(Function<? super List<T>, ? extends B> collateFn) {
        return new CollatingDataLoader<>(runtime, collateFn);
    }

    public PrefetchingIterable<List<T>> prefetch() {
        return prefetch(this);
    }

    public PrefetchingIterable<List<T>> prefetch(int bufferSize) {
        return prefetch(this, bufferSize);
    }

    public static TensorBuilder tensorBuilder(TensorDatasetAdapter dataset) {
        return new TensorBuilder(dataset);
    }

    public static <B> PrefetchingIterable<B> prefetch(Iterable<? extends B> source) {
        return PrefetchingIterable.of(source);
    }

    public static <B> PrefetchingIterable<B> prefetch(Iterable<? extends B> source, int bufferSize) {
        return PrefetchingIterable.of(source, bufferSize);
    }

    public static CollateFn defaultTensorCollate() {
        return TensorCollators.defaultPairCollate();
    }

    public static Function<List<Dataset.Sample>, Batch> sampleBatchCollate() {
        return TensorCollators.sampleBatchCollate();
    }

    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, Batch> tensorPairBatchCollate() {
        return TensorCollators.tensorPairBatchCollate();
    }

    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate() {
        return paddedSampleBatchCollate(0.0f, 0.0f);
    }

    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate(float padValue) {
        return paddedSampleBatchCollate(padValue, padValue);
    }

    public static Function<List<Dataset.Sample>, PaddedBatch> paddedSampleBatchCollate(
            float inputPadValue,
            float labelPadValue) {
        return TensorCollators.paddedSampleBatchCollate(inputPadValue, labelPadValue);
    }

    public static Function<List<Dataset.Sample>, PaddedBatch> causalLanguageModelingBatchCollate(int padTokenId) {
        return causalLanguageModelingBatchCollate((float) padTokenId, DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX);
    }

    public static Function<List<Dataset.Sample>, PaddedBatch> causalLanguageModelingBatchCollate(
            float inputPadValue,
            float labelIgnoreIndex) {
        return paddedSampleBatchCollate(inputPadValue, labelIgnoreIndex);
    }

    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, PaddedBatch> paddedTensorPairBatchCollate() {
        return paddedTensorPairBatchCollate(0.0f, 0.0f);
    }

    public static Function<List<Dataset.Pair<GradTensor, GradTensor>>, PaddedBatch> paddedTensorPairBatchCollate(
            float padValue) {
        return paddedTensorPairBatchCollate(padValue, padValue);
    }

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
        return SamplerFactories.weightedRandomSampler(sampleWeights, numSamples, replacement, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            long seed) {
        return SamplerFactories.lengthBucketBatchSampler(lengths, batchSize, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        return SamplerFactories.lengthBucketBatchSampler(dataset, lengthExtractor, batchSize, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucketBatchSampler(lengths, batchSize, dropLast, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucketBatchSampler(dataset, lengthExtractor, batchSize, dropLast, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return SamplerFactories.lengthBucketBatchSampler(
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
        return SamplerFactories.lengthBucketBatchSampler(
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
        return SamplerFactories.tokenBudgetBatchSampler(lengths, maxTokens, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return SamplerFactories.tokenBudgetBatchSampler(dataset, lengthExtractor, maxTokens, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            long seed) {
        return SamplerFactories.tokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return SamplerFactories.tokenBudgetBatchSampler(dataset, lengthExtractor, maxTokens, maxExamples, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return SamplerFactories.tokenBudgetBatchSampler(
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
        return SamplerFactories.tokenBudgetBatchSampler(
                dataset,
                lengthExtractor,
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public static SequentialSampler sequentialSampler() {
        return SamplerFactories.sequentialSampler();
    }

    public static RandomSampler randomSampler(long seed) {
        return SamplerFactories.randomSampler(seed);
    }

    public static RandomSampler randomSampler(int numSamples, boolean replacement, long seed) {
        return SamplerFactories.randomSampler(numSamples, replacement, seed);
    }

    public static SubsetSampler subsetSampler(int... indices) {
        return SamplerFactories.subsetSampler(indices);
    }

    public static DistributedSampler distributedSampler(int numReplicas, int rank) {
        return SamplerFactories.distributedSampler(numReplicas, rank);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return SamplerFactories.distributedSampler(numReplicas, rank, shuffle, dropLast, seed);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return SamplerFactories.distributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }

    public static TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return TensorLabelEncoding.tensorDataset(inputs, labels);
    }

    public static GradTensor classLabels(int... labels) {
        return TensorLabelEncoding.classLabels(labels);
    }

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

        public PrefetchingIterable<B> prefetch() {
            return DataLoader.prefetch(this);
        }

        public PrefetchingIterable<B> prefetch(int bufferSize) {
            return DataLoader.prefetch(this, bufferSize);
        }
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
        return LabelWeighting.positiveWeights(BinaryLabelMatrix.from(labels));
    }

    public static float[] multiLabelPositiveWeights(boolean[][] labels) {
        return LabelWeighting.positiveWeights(BinaryLabelMatrix.from(labels));
    }

    public static float[] multiLabelPositiveWeights(float[][] labels) {
        return LabelWeighting.positiveWeights(BinaryLabelMatrix.from(labels));
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
        return tensorBuilder(tensorDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader classification(GradTensor inputs, int[] labels, int batchSize) {
        return tensorBuilder(classificationDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, int[] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, int[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, boolean[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDataLoader binary(GradTensor inputs, float[][] labels, int batchSize) {
        return tensorBuilder(binaryDataset(inputs, labels)).batchSize(batchSize).build();
    }

    public static TensorDatasetSplit split(GradTensor inputs, GradTensor labels, double trainFraction, long seed) {
        return tensorDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit classificationSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return classificationDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit classificationStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorDatasetSplits.stratifiedSplit(classificationDataset(inputs, labels), labels, trainFraction, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit binaryStratifiedSplit(
            GradTensor inputs,
            int[] labels,
            double trainFraction,
            long seed) {
        return TensorDatasetSplits.stratifiedSplit(binaryDataset(inputs, labels), labels, trainFraction, seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            int[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = BinaryLabelMatrix.from(labels);
        return TensorDatasetSplits.multiLabelStratifiedSplit(TensorLabelEncoding.binaryDataset(inputs, matrix),
                matrix,
                trainFraction,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            boolean[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = BinaryLabelMatrix.from(labels);
        return TensorDatasetSplits.multiLabelStratifiedSplit(TensorLabelEncoding.binaryDataset(inputs, matrix),
                matrix,
                trainFraction,
                seed);
    }

    public static TensorDatasetSplit binarySplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        return binaryDataset(inputs, labels).split(trainFraction, seed);
    }

    public static TensorDatasetSplit multiLabelBinaryStratifiedSplit(
            GradTensor inputs,
            float[][] labels,
            double trainFraction,
            long seed) {
        BinaryLabelMatrix matrix = BinaryLabelMatrix.from(labels);
        return TensorDatasetSplits.multiLabelStratifiedSplit(TensorLabelEncoding.binaryDataset(inputs, matrix),
                matrix,
                trainFraction,
                seed);
    }

    public interface TensorDatasetAdapter {
        int size();

        default int size(List<Integer> indices) {
            return indices.size();
        }

        GradTensor[] get(int index);
    }

    public static class TensorDataset implements TensorDatasetAdapter {
        private final TensorDatasetSamples samples;

        public TensorDataset(GradTensor[]... samples) {
            this.samples = TensorDatasetSamples.of(samples);
        }

        public TensorDataset(GradTensor inputs, GradTensor targets) {
            this.samples = TensorDatasetSamples.fromBatchedTensors(inputs, targets);
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public GradTensor[] get(int index) {
            return samples.get(index);
        }

        public TensorDatasetSplit split(double trainFraction, long seed) {
            return TensorDatasetSplits.randomSplit(this, trainFraction, seed);
        }
    }

    public record TensorDatasetSplit(TensorDataset train, TensorDataset validation) {
        public TensorDatasetSplit {
            Objects.requireNonNull(train, "train must not be null");
            Objects.requireNonNull(validation, "validation must not be null");
        }

        public TensorBuilder trainLoader() {
            return tensorBuilder(train);
        }

        public TensorBuilder validationLoader() {
            return tensorBuilder(validation);
        }

        public TensorDataLoader trainLoader(int batchSize) {
            return trainLoader().batchSize(batchSize).build();
        }

        public TensorDataLoader trainLoader(int batchSize, boolean shuffle, long seed) {
            return trainLoader().batchSize(batchSize).shuffle(shuffle).seed(seed).build();
        }

        public TensorDataLoader validationLoader(int batchSize) {
            return validationLoader().batchSize(batchSize).build();
        }
    }

    public static class TensorBuilder {
        private final TensorDataLoaderBuilderState state;

        TensorBuilder(TensorDatasetAdapter dataset) {
            this.state = new TensorDataLoaderBuilderState(dataset);
        }

        public TensorBuilder batchSize(int batchSize) {
            state.batchSize(batchSize);
            return this;
        }

        public TensorBuilder shuffle(boolean shuffle) {
            state.shuffle(shuffle);
            return this;
        }

        public TensorBuilder dropLast(boolean dropLast) {
            state.dropLast(dropLast);
            return this;
        }

        public TensorBuilder seed(long seed) {
            state.seed(seed);
            return this;
        }

        public TensorBuilder randomSeed(long seed) {
            return seed(seed);
        }

        public TensorBuilder reshuffleEachEpoch() {
            return reshuffleEachEpoch(true);
        }

        public TensorBuilder reshuffleEachEpoch(boolean reshuffleEachEpoch) {
            state.reshuffleEachEpoch(reshuffleEachEpoch);
            return this;
        }

        public TensorBuilder initialEpoch(long initialEpoch) {
            state.initialEpoch(initialEpoch);
            return this;
        }

        public TensorBuilder startEpoch(long initialEpoch) {
            return initialEpoch(initialEpoch);
        }

        public TensorBuilder sampler(IndexSampler sampler) {
            state.sampler(sampler);
            return this;
        }

        public TensorBuilder batchSampler(BatchSampler batchSampler) {
            state.batchSampler(batchSampler);
            return this;
        }

        public TensorBuilder weightedRandomSampler(
                float[] sampleWeights,
                int numSamples,
                boolean replacement,
                long seed) {
            return sampler(DataLoader.weightedRandomSampler(sampleWeights, numSamples, replacement, seed));
        }

        public TensorBuilder lengthBucketBatchSampler(int[] lengths, long seed) {
            return sampler(DataLoader.lengthBucketBatchSampler(lengths, state.batchSize(), state.dropLast(), seed));
        }

        public TensorBuilder lengthBucketBatchSampler(ToIntFunction<GradTensor[]> lengthExtractor, long seed) {
            return lengthBucketBatchSampler(state.tensorDatasetLengths(lengthExtractor), seed);
        }

        public TensorBuilder lengthBucketBatchSampler(int[] lengths, int batchSize, long seed) {
            batchSize(batchSize);
            return lengthBucketBatchSampler(lengths, seed);
        }

        public TensorBuilder lengthBucketBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int batchSize,
                long seed) {
            batchSize(batchSize);
            return lengthBucketBatchSampler(lengthExtractor, seed);
        }

        public TensorBuilder lengthBucketBatchSampler(
                int[] lengths,
                int bucketSizeMultiplier,
                boolean shuffleBatches,
                boolean shuffleWithinBuckets,
                long seed) {
            return sampler(DataLoader.lengthBucketBatchSampler(
                    lengths,
                    state.batchSize(),
                    bucketSizeMultiplier,
                    shuffleBatches,
                    shuffleWithinBuckets,
                    state.dropLast(),
                    seed));
        }

        public TensorBuilder lengthBucketBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int bucketSizeMultiplier,
                boolean shuffleBatches,
                boolean shuffleWithinBuckets,
                long seed) {
            return lengthBucketBatchSampler(
                    state.tensorDatasetLengths(lengthExtractor),
                    bucketSizeMultiplier,
                    shuffleBatches,
                    shuffleWithinBuckets,
                    seed);
        }

        public TensorBuilder lengthBucketBatchSampler(
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

        public TensorBuilder lengthBucketBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int batchSize,
                int bucketSizeMultiplier,
                boolean shuffleBatches,
                boolean shuffleWithinBuckets,
                boolean dropLast,
                long seed) {
            return lengthBucketBatchSampler(
                    state.tensorDatasetLengths(lengthExtractor),
                    batchSize,
                    bucketSizeMultiplier,
                    shuffleBatches,
                    shuffleWithinBuckets,
                    dropLast,
                    seed);
        }

        public TensorBuilder inputLengthBucketBatchSampler(long seed) {
            return lengthBucketBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), seed);
        }

        public TensorBuilder inputLengthBucketBatchSampler(int batchSize, long seed) {
            batchSize(batchSize);
            return inputLengthBucketBatchSampler(seed);
        }

        public TensorBuilder tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
            return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, seed));
        }

        public TensorBuilder tokenBudgetBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int maxTokens,
                long seed) {
            return tokenBudgetBatchSampler(state.tensorDatasetLengths(lengthExtractor), maxTokens, seed);
        }

        public TensorBuilder tokenBudgetBatchSampler(int[] lengths, int maxTokens, int maxExamples, long seed) {
            return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed));
        }

        public TensorBuilder tokenBudgetBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int maxTokens,
                int maxExamples,
                long seed) {
            return tokenBudgetBatchSampler(state.tensorDatasetLengths(lengthExtractor), maxTokens, maxExamples, seed);
        }

        public TensorBuilder tokenBudgetBatchSampler(
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

        public TensorBuilder tokenBudgetBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int maxTokens,
                int maxExamples,
                boolean shuffleBatches,
                boolean shuffleWithinBatches,
                long seed) {
            return tokenBudgetBatchSampler(
                    state.tensorDatasetLengths(lengthExtractor),
                    maxTokens,
                    maxExamples,
                    shuffleBatches,
                    shuffleWithinBatches,
                    seed);
        }

        public TensorBuilder inputTokenBudgetBatchSampler(int maxTokens, long seed) {
            return tokenBudgetBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), maxTokens, seed);
        }

        public TensorBuilder inputTokenBudgetBatchSampler(int maxTokens, int maxExamples, long seed) {
            return tokenBudgetBatchSampler(sample -> DataLoader.sequenceLength(sample[0]), maxTokens, maxExamples, seed);
        }

        public TensorBuilder sequentialSampler() {
            return sampler(DataLoader.sequentialSampler());
        }

        public TensorBuilder randomSampler(long seed) {
            return sampler(DataLoader.randomSampler(seed));
        }

        public TensorBuilder randomSampler(int numSamples, boolean replacement, long seed) {
            return sampler(DataLoader.randomSampler(numSamples, replacement, seed));
        }

        public TensorBuilder subsetSampler(int... indices) {
            return sampler(DataLoader.subsetSampler(indices));
        }

        public TensorBuilder distributedSampler(int numReplicas, int rank) {
            return sampler(DataLoader.distributedSampler(numReplicas, rank));
        }

        public TensorBuilder distributedSampler(
                int numReplicas,
                int rank,
                boolean shuffle,
                boolean dropLast,
                long seed) {
            return sampler(DataLoader.distributedSampler(numReplicas, rank, shuffle, dropLast, seed));
        }

        public TensorBuilder distributedSampler(
                int numReplicas,
                int rank,
                boolean shuffle,
                boolean dropLast,
                long seed,
                long epoch) {
            return sampler(DataLoader.distributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch));
        }

        public TensorBuilder collateFn(CollateFn collateFn) {
            state.collateFn(collateFn);
            return this;
        }

        public TensorDataLoader build() {
            return new TensorDataLoader(this);
        }
    }

    public static class TensorDataLoader implements Iterable<Batch> {
        private final TensorDataLoaderRuntime runtime;

        private TensorDataLoader(TensorBuilder builder) {
            this.runtime = new TensorDataLoaderRuntime(builder.state.config());
        }

        @Override
        public Iterator<Batch> iterator() {
            return runtime.iterator();
        }

        public Iterable<Batch> epoch(long epoch) {
            return runtime.epoch(epoch);
        }

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

        public PrefetchingIterable<Batch> prefetch() {
            return DataLoader.prefetch(this);
        }

        public PrefetchingIterable<Batch> prefetch(int bufferSize) {
            return DataLoader.prefetch(this, bufferSize);
        }
    }

    @FunctionalInterface
    public interface CollateFn {
        Batch collate(List<Integer> indices, TensorDatasetAdapter dataset);
    }

    public record Batch(GradTensor inputs, GradTensor labels) {
        public Batch {
            inputs = BatchSupport.requireTensor("inputs", inputs);
            labels = BatchSupport.requireTensor("labels", labels);
            BatchSupport.requireCompatibleBatchDimensions(inputs, labels);
        }
    }

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
