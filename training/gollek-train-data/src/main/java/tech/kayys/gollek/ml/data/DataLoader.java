package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;

/**
 * DataLoader that batches, shuffles, and yields data in mini-batches.
 *
 * @param <T> type of elements produced by this loader
 */
public class DataLoader<T> implements Iterable<List<T>> {
    public static final float DEFAULT_CAUSAL_LM_LABEL_IGNORE_INDEX = -100.0f;

    private final Dataset<? extends T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final IndexSampler sampler;
    private final BatchSampler batchSampler;

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
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        if (sampler != null && batchSampler != null) {
            throw new IllegalArgumentException("sampler and batchSampler cannot both be configured");
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.sampler = sampler;
        this.batchSampler = batchSampler;
    }

    @Override
    public Iterator<List<T>> iterator() {
        if (batchSampler != null) {
            return batchSamplerIterator();
        }
        List<Integer> indices = sampler == null
                ? new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList())
                : new ArrayList<>(sampler.sample(dataset.size()));

        if (sampler == null && shuffle) {
            Collections.shuffle(indices, ThreadLocalRandom.current());
        }

        int numBatches = numBatches();

        return new Iterator<>() {
            private int current = 0;

            @Override
            public boolean hasNext() {
                return current < numBatches;
            }

            @Override
            public List<T> next() {
                if (current >= numBatches) {
                    throw new NoSuchElementException();
                }

                int start = current * batchSize;
                int end = Math.min(start + batchSize, indices.size());

                List<T> batch = new ArrayList<>(end - start);
                for (int i = start; i < end; i++) {
                    batch.add(dataset.get(indices.get(i)));
                }

                current++;
                return batch;
            }
        };
    }

    public int numBatches() {
        if (batchSampler != null) {
            return batchSampler.batchCount(dataset.size());
        }
        int n = sampleCount();
        return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
    }

    public int size() {
        return dataset.size();
    }

    public int batchSize() {
        return batchSize;
    }

    public int sampleCount() {
        if (batchSampler != null) {
            return batchSampler.sampleCount(dataset.size());
        }
        return sampler == null ? dataset.size() : sampler.sampleCount(dataset.size());
    }

    public boolean sampled() {
        return sampler != null || batchSampler != null;
    }

    public boolean shuffle() {
        return shuffle;
    }

    public boolean dropLast() {
        return dropLast;
    }

    public <R> DataLoader<R> map(Function<? super T, ? extends R> mapper) {
        Dataset<? extends T> srcDataset = this.dataset;
        Dataset<R> mappedDataset = new Dataset<>() {
            @Override
            public R get(int index) {
                return mapper.apply(srcDataset.get(index));
            }

            @Override
            public int size() {
                return srcDataset.size();
            }
        };
        return new DataLoader<>(
                mappedDataset,
                this.batchSize,
                this.shuffle,
                this.dropLast,
                this.sampler,
                this.batchSampler);
    }

    private Iterator<List<T>> batchSamplerIterator() {
        List<List<Integer>> batches = batchSampler.sampleBatches(dataset.size());
        return new Iterator<>() {
            private int current = 0;

            @Override
            public boolean hasNext() {
                return current < batches.size();
            }

            @Override
            public List<T> next() {
                if (current >= batches.size()) {
                    throw new NoSuchElementException();
                }
                List<Integer> batchIndices = batches.get(current);
                List<T> batch = new ArrayList<>(batchIndices.size());
                for (int index : batchIndices) {
                    batch.add(dataset.get(index));
                }
                current++;
                return batch;
            }
        };
    }

    public <B> CollatingDataLoader<T, B> collate(Function<? super List<T>, ? extends B> collateFn) {
        return new CollatingDataLoader<>(this, collateFn);
    }

    public static TensorBuilder tensorBuilder(TensorDatasetAdapter dataset) {
        return new TensorBuilder(dataset);
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
        return new WeightedRandomSampler(sampleWeights, numSamples, replacement, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            long seed) {
        return lengthBucketBatchSampler(sequenceLengths(dataset, lengthExtractor), batchSize, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            boolean dropLast,
            long seed) {
        return new LengthBucketBatchSampler(lengths, batchSize, dropLast, seed);
    }

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(sequenceLengths(dataset, lengthExtractor), batchSize, dropLast, seed);
    }

    public static LengthBucketBatchSampler lengthBucketBatchSampler(
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

    public static <T> LengthBucketBatchSampler lengthBucketBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        return lengthBucketBatchSampler(
                sequenceLengths(dataset, lengthExtractor),
                batchSize,
                bucketSizeMultiplier,
                shuffleBatches,
                shuffleWithinBuckets,
                dropLast,
                seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            long seed) {
        return tokenBudgetBatchSampler(sequenceLengths(dataset, lengthExtractor), maxTokens, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            long seed) {
        return new TokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed);
    }

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            long seed) {
        return tokenBudgetBatchSampler(sequenceLengths(dataset, lengthExtractor), maxTokens, maxExamples, seed);
    }

    public static TokenBudgetBatchSampler tokenBudgetBatchSampler(
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

    public static <T> TokenBudgetBatchSampler tokenBudgetBatchSampler(
            Dataset<? extends T> dataset,
            ToIntFunction<? super T> lengthExtractor,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        return tokenBudgetBatchSampler(
                sequenceLengths(dataset, lengthExtractor),
                maxTokens,
                maxExamples,
                shuffleBatches,
                shuffleWithinBatches,
                seed);
    }

    public static SequentialSampler sequentialSampler() {
        return new SequentialSampler();
    }

    public static RandomSampler randomSampler(long seed) {
        return new RandomSampler(seed);
    }

    public static RandomSampler randomSampler(int numSamples, boolean replacement, long seed) {
        return new RandomSampler(numSamples, replacement, seed);
    }

    public static SubsetSampler subsetSampler(int... indices) {
        return new SubsetSampler(indices);
    }

    public static DistributedSampler distributedSampler(int numReplicas, int rank) {
        return new DistributedSampler(numReplicas, rank);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed);
    }

    public static DistributedSampler distributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }

    public static TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return new TensorDataset(inputs, labels);
    }

    public static GradTensor classLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length);
    }

    public static class CollatingDataLoader<T, B> implements Iterable<B> {
        private final DataLoader<T> loader;
        private final Function<? super List<T>, ? extends B> collateFn;

        private CollatingDataLoader(DataLoader<T> loader, Function<? super List<T>, ? extends B> collateFn) {
            this.loader = Objects.requireNonNull(loader, "loader must not be null");
            this.collateFn = Objects.requireNonNull(collateFn, "collateFn must not be null");
        }

        @Override
        public Iterator<B> iterator() {
            Iterator<List<T>> batches = loader.iterator();
            return new Iterator<>() {
                @Override
                public boolean hasNext() {
                    return batches.hasNext();
                }

                @Override
                public B next() {
                    return collateFn.apply(batches.next());
                }
            };
        }

        public int numBatches() {
            return loader.numBatches();
        }

        public int size() {
            return loader.size();
        }

        public int sampleCount() {
            return loader.sampleCount();
        }

        public boolean sampled() {
            return loader.sampled();
        }

        public int batchSize() {
            return loader.batchSize();
        }

        public boolean shuffle() {
            return loader.shuffle();
        }

        public boolean dropLast() {
            return loader.dropLast();
        }
    }

    public static float[] classWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int maxLabel = -1;
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            maxLabel = Math.max(maxLabel, label);
        }
        return classWeightsFor(maxLabel + 1, labels);
    }

    public static float[] classWeightsFor(int numClasses, int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (numClasses <= 0) {
            throw new IllegalArgumentException("numClasses must be positive, got: " + numClasses);
        }
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int[] counts = new int[numClasses];
        for (int label : labels) {
            if (label < 0 || label >= numClasses) {
                throw new IllegalArgumentException(
                        "class label " + label + " out of range [0, " + (numClasses - 1) + "]");
            }
            counts[label]++;
        }
        return balancedClassWeights(counts, labels.length);
    }

    public static float[] classBalancedSampleWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int maxLabel = -1;
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            maxLabel = Math.max(maxLabel, label);
        }
        float[] classWeights = classWeightsFor(maxLabel + 1, labels);
        float[] sampleWeights = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            sampleWeights[i] = classWeights[labels[i]];
        }
        return sampleWeights;
    }

    public static float[] binaryBalancedSampleWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        for (int label : labels) {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + label);
            }
        }
        return classBalancedSampleWeights(labels);
    }

    public static GradTensor binaryLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] != 0 && labels[i] != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            values[i] = labels[i] ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            float value = labels[i];
            if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
            }
            values[i] = value >= 0.5f ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    public static GradTensor binaryLabels(int[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static GradTensor binaryLabels(boolean[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static GradTensor binaryLabels(float[][] labels) {
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    public static float binaryPositiveWeight(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (int label : labels) {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + label);
            }
            positives += label;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float binaryPositiveWeight(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (boolean label : labels) {
            positives += label ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float binaryPositiveWeight(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (float label : labels) {
            if (Math.abs(label) > 1e-6f && Math.abs(label - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + label);
            }
            positives += label >= 0.5f ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    public static float[] multiLabelPositiveWeights(int[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static float[] multiLabelPositiveWeights(boolean[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static float[] multiLabelPositiveWeights(float[][] labels) {
        return positiveWeights(binaryLabelMatrix(labels));
    }

    public static TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.length) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.length);
        }
        return tensorDataset(inputs, classLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.length) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.length);
        }
        return tensorDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    public static TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
        return binaryDataset(inputs, binaryLabels(labels));
    }

    private static TensorDataset binaryDataset(GradTensor inputs, GradTensor labels) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != labels.shape()[0]) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + labels.shape()[0]);
        }
        return tensorDataset(inputs, labels);
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
        return stratifiedSplit(classificationDataset(inputs, labels), labels, trainFraction, seed);
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
        return stratifiedSplit(binaryDataset(inputs, labels), labels, trainFraction, seed);
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
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
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
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
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
        BinaryLabelMatrix matrix = binaryLabelMatrix(labels);
        return multiLabelStratifiedSplit(binaryDataset(inputs, GradTensor.of(matrix.values(), matrix.rows(), matrix.columns())),
                matrix,
                trainFraction,
                seed);
    }

    private static TensorDatasetSplit stratifiedSplit(
            TensorDataset dataset,
            int[] labels,
            double trainFraction,
            long seed) {
        requireTrainFraction(trainFraction);
        if (labels.length != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.length + " vs " + dataset.size());
        }
        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            groups.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        }

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, entry.getKey())));
            int trainCount = stratifiedTrainCount(group.size(), trainFraction);
            trainIndices.addAll(group.subList(0, trainCount));
            validationIndices.addAll(group.subList(trainCount, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x6A09E667F3BCC909L));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xBB67AE8584CAA73BL));
        return splitByIndices(dataset, trainIndices, validationIndices);
    }

    private static TensorDatasetSplit multiLabelStratifiedSplit(
            TensorDataset dataset,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        requireTrainFraction(trainFraction);
        if (labels.rows() != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.rows() + " vs " + dataset.size());
        }

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        int[] trainTargets = new int[columns];
        for (int column = 0; column < columns; column++) {
            trainTargets[column] = stratifiedTrainCount(totalPositives[column], trainFraction);
        }

        int trainTargetSize = stratifiedTrainCount(rows, trainFraction);
        int validationTargetSize = rows - trainTargetSize;
        int[] trainCounts = new int[columns];
        List<Integer> order = multiLabelStratificationOrder(flatLabels, rows, columns, totalPositives, seed);
        List<Integer> trainIndices = new ArrayList<>(trainTargetSize);
        List<Integer> validationIndices = new ArrayList<>(validationTargetSize);

        for (int row : order) {
            if (trainIndices.size() >= trainTargetSize) {
                validationIndices.add(row);
                continue;
            }
            if (validationIndices.size() >= validationTargetSize) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
                continue;
            }

            double trainPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size() + 1,
                    trainTargetSize, true);
            double validationPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size(),
                    trainTargetSize, false);
            if (trainPenalty <= validationPenalty) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
            } else {
                validationIndices.add(row);
            }
        }

        improveMultiLabelAssignment(
                flatLabels, columns, totalPositives, trainTargets, trainIndices, validationIndices, trainCounts);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x3C6EF372FE94F82AL));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xA54FF53A5F1D36F1L));
        return splitByIndices(dataset, trainIndices, validationIndices);
    }

    private static int[] binaryLabelValues(BinaryLabelMatrix labels) {
        int[] values = new int[labels.values().length];
        for (int i = 0; i < labels.values().length; i++) {
            values[i] = labels.values()[i] >= 0.5f ? 1 : 0;
        }
        return values;
    }

    private static int[] columnPositiveCounts(int[] flatLabels, int rows, int columns) {
        int[] counts = new int[columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                counts[column] += flatLabels[row * columns + column];
            }
        }
        return counts;
    }

    private static List<Integer> multiLabelStratificationOrder(
            int[] flatLabels,
            int rows,
            int columns,
            int[] totalPositives,
            long seed) {
        List<Integer> order = new ArrayList<>(IntStream.range(0, rows).boxed().toList());
        Map<Integer, Integer> tieBreakers = new HashMap<>();
        Collections.shuffle(order, new Random(seed));
        for (int i = 0; i < order.size(); i++) {
            tieBreakers.put(order.get(i), i);
        }
        order.sort((left, right) -> {
            int rarity = Double.compare(
                    rowRarityScore(flatLabels, right, columns, totalPositives),
                    rowRarityScore(flatLabels, left, columns, totalPositives));
            if (rarity != 0) {
                return rarity;
            }
            int cardinality = Integer.compare(
                    rowCardinality(flatLabels, right, columns),
                    rowCardinality(flatLabels, left, columns));
            if (cardinality != 0) {
                return cardinality;
            }
            return Integer.compare(tieBreakers.get(left), tieBreakers.get(right));
        });
        return order;
    }

    private static double rowRarityScore(int[] flatLabels, int row, int columns, int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < columns; column++) {
            if (flatLabels[row * columns + column] == 1) {
                score += totalPositives[column] == 0 ? 0.0 : 1.0 / totalPositives[column];
            }
        }
        return score;
    }

    private static int rowCardinality(int[] flatLabels, int row, int columns) {
        int cardinality = 0;
        for (int column = 0; column < columns; column++) {
            cardinality += flatLabels[row * columns + column];
        }
        return cardinality;
    }

    private static double multiLabelSplitPenalty(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[] trainCounts,
            int[] trainTargets,
            int trainSize,
            int trainTargetSize,
            boolean assignToTrain) {
        double penalty = Math.pow(trainSize - trainTargetSize, 2) * columns;
        for (int column = 0; column < columns; column++) {
            int after = trainCounts[column];
            if (assignToTrain) {
                after += flatLabels[row * columns + column];
            }
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            penalty += Math.pow(after - trainTargets[column], 2) * labelWeight;
        }
        return penalty;
    }

    private static void addRowCounts(int[] flatLabels, int row, int columns, int[] counts, int delta) {
        for (int column = 0; column < columns; column++) {
            counts[column] += flatLabels[row * columns + column] * delta;
        }
    }

    private static void improveMultiLabelAssignment(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[] trainTargets,
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            int[] trainCounts) {
        double bestScore = multiLabelAssignmentScore(trainCounts, trainTargets, totalPositives);
        boolean improved;
        do {
            improved = false;
            int bestTrainPosition = -1;
            int bestValidationPosition = -1;
            int[] bestCounts = null;

            for (int trainPosition = 0; trainPosition < trainIndices.size(); trainPosition++) {
                int trainRow = trainIndices.get(trainPosition);
                for (int validationPosition = 0; validationPosition < validationIndices.size(); validationPosition++) {
                    int validationRow = validationIndices.get(validationPosition);
                    int[] candidateCounts = swappedTrainCounts(
                            flatLabels, columns, trainCounts, trainRow, validationRow);
                    double candidateScore = multiLabelAssignmentScore(candidateCounts, trainTargets, totalPositives);
                    if (candidateScore + 1e-9 < bestScore) {
                        bestScore = candidateScore;
                        bestTrainPosition = trainPosition;
                        bestValidationPosition = validationPosition;
                        bestCounts = candidateCounts;
                        improved = true;
                    }
                }
            }

            if (improved) {
                int trainRow = trainIndices.get(bestTrainPosition);
                trainIndices.set(bestTrainPosition, validationIndices.get(bestValidationPosition));
                validationIndices.set(bestValidationPosition, trainRow);
                System.arraycopy(bestCounts, 0, trainCounts, 0, trainCounts.length);
            }
        } while (improved);
    }

    private static int[] swappedTrainCounts(
            int[] flatLabels,
            int columns,
            int[] trainCounts,
            int trainRow,
            int validationRow) {
        int[] candidate = trainCounts.clone();
        addRowCounts(flatLabels, trainRow, columns, candidate, -1);
        addRowCounts(flatLabels, validationRow, columns, candidate, 1);
        return candidate;
    }

    private static double multiLabelAssignmentScore(
            int[] trainCounts,
            int[] trainTargets,
            int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < trainTargets.length; column++) {
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            score += Math.pow(trainCounts[column] - trainTargets[column], 2) * labelWeight;
        }
        return score;
    }

    private static int stratifiedTrainCount(int groupSize, double trainFraction) {
        if (groupSize <= 0) {
            return 0;
        }
        if (groupSize == 1) {
            return 1;
        }
        int trainCount = (int) Math.round(groupSize * trainFraction);
        return Math.max(1, Math.min(groupSize - 1, trainCount));
    }

    private static void rebalanceNonEmptySplit(List<Integer> trainIndices, List<Integer> validationIndices) {
        if (trainIndices.isEmpty() && !validationIndices.isEmpty()) {
            trainIndices.add(validationIndices.remove(validationIndices.size() - 1));
        }
        if (validationIndices.isEmpty() && trainIndices.size() > 1) {
            validationIndices.add(trainIndices.remove(trainIndices.size() - 1));
        }
    }

    private static TensorDatasetSplit splitByIndices(
            TensorDataset dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        List<GradTensor[]> train = new ArrayList<>(trainIndices.size());
        List<GradTensor[]> validation = new ArrayList<>(validationIndices.size());
        for (int index : trainIndices) {
            train.add(dataset.get(index));
        }
        for (int index : validationIndices) {
            validation.add(dataset.get(index));
        }
        return new TensorDatasetSplit(new TensorDataset(train), new TensorDataset(validation));
    }

    private static void requireTrainFraction(double trainFraction) {
        if (trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be between 0 and 1");
        }
    }

    private static long mixSeed(long seed, int label) {
        long value = seed ^ (0x9E3779B97F4A7C15L * (label + 0x632BE5ABL));
        value ^= (value >>> 33);
        value *= 0xff51afd7ed558ccdL;
        value ^= (value >>> 33);
        return value;
    }

    private static float[] positiveWeights(BinaryLabelMatrix matrix) {
        int[] positives = new int[matrix.columns()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {
                if (matrix.values()[row * matrix.columns() + column] >= 0.5f) {
                    positives[column]++;
                }
            }
        }
        float[] weights = new float[matrix.columns()];
        for (int column = 0; column < matrix.columns(); column++) {
            weights[column] = positiveWeight(positives[column], matrix.rows());
        }
        return weights;
    }

    private static float positiveWeight(int positives, int total) {
        if (total <= 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int negatives = total - positives;
        if (positives == 0 || negatives == 0) {
            return 1.0f;
        }
        return negatives / (float) positives;
    }

    private static float[] balancedClassWeights(int[] counts, int total) {
        float[] weights = new float[counts.length];
        for (int i = 0; i < counts.length; i++) {
            weights[i] = counts[i] == 0 ? 1.0f : total / (float) (counts.length * counts[i]);
        }
        return weights;
    }

    private static BinaryLabelMatrix binaryLabelMatrix(int[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            int[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                int value = labelRow[column];
                if (value != 0 && value != 1) {
                    throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + value);
                }
                values[row * columns + column] = value;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static BinaryLabelMatrix binaryLabelMatrix(boolean[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            boolean[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                values[row * columns + column] = labelRow[column] ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static BinaryLabelMatrix binaryLabelMatrix(float[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            float[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                float value = labelRow[column];
                if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                    throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
                }
                values[row * columns + column] = value >= 0.5f ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static int matrixColumns(int rows, int columns) {
        if (rows == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one row");
        }
        if (columns == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one column");
        }
        return columns;
    }

    private static int rowLength(int[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(boolean[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(float[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static void requireRowWidth(int row, int actual, int expected) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "binary label matrix must be rectangular; row "
                            + row + " has " + actual + " columns, expected " + expected);
        }
    }

    private record BinaryLabelMatrix(float[] values, int rows, int columns) {
    }

    public interface TensorDatasetAdapter {
        int size();

        default int size(List<Integer> indices) {
            return indices.size();
        }

        GradTensor[] get(int index);
    }

    public static class TensorDataset implements TensorDatasetAdapter {
        private final List<GradTensor[]> samples;

        public TensorDataset(GradTensor[]... samples) {
            this.samples = copySamples(samples);
        }

        private TensorDataset(List<GradTensor[]> samples) {
            this.samples = copySamples(samples);
        }

        public TensorDataset(GradTensor inputs, GradTensor targets) {
            if (inputs.shape().length == 0 || targets.shape().length == 0) {
                throw new IllegalArgumentException("inputs and targets must include a batch dimension");
            }
            if (inputs.shape()[0] != targets.shape()[0]) {
                throw new IllegalArgumentException("inputs and targets must have same batch dimension");
            }
            int n = (int) inputs.shape()[0];
            long[] inputShape = inputs.shape();
            long[] targetShape = targets.shape();
            samples = new ArrayList<>(n);

            for (int i = 0; i < n; i++) {
                int inputStride = 1;
                for (int d = 1; d < inputShape.length; d++) {
                    inputStride *= inputShape[d];
                }
                int targetStride = 1;
                for (int d = 1; d < targetShape.length; d++) {
                    targetStride *= targetShape[d];
                }

                float[] inputData = inputs.data();
                float[] targetData = targets.data();
                int inputOffset = i * inputStride;
                int targetOffset = i * targetStride;

                long[] inputSampleShape = Arrays.copyOfRange(inputShape, 1, inputShape.length);
                long[] targetSampleShape = Arrays.copyOfRange(targetShape, 1, targetShape.length);

                float[] inputSample = Arrays.copyOfRange(inputData, inputOffset, inputOffset + inputStride);
                float[] targetSample = Arrays.copyOfRange(targetData, targetOffset, targetOffset + targetStride);

                samples.add(new GradTensor[] {
                    GradTensor.of(inputSample, inputSampleShape),
                    GradTensor.of(targetSample, targetSampleShape)
                });
            }
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public GradTensor[] get(int index) {
            GradTensor[] sample = samples.get(index);
            return Arrays.copyOf(sample, sample.length);
        }

        public TensorDatasetSplit split(double trainFraction, long seed) {
            if (trainFraction <= 0.0 || trainFraction >= 1.0) {
                throw new IllegalArgumentException("trainFraction must be between 0 and 1");
            }
            List<Integer> indices = new ArrayList<>(IntStream.range(0, size()).boxed().toList());
            Collections.shuffle(indices, new Random(seed));
            int trainSize = (int) Math.round(size() * trainFraction);
            trainSize = Math.max(1, Math.min(size() - 1, trainSize));
            List<GradTensor[]> train = new ArrayList<>(trainSize);
            List<GradTensor[]> validation = new ArrayList<>(size() - trainSize);
            for (int i = 0; i < indices.size(); i++) {
                GradTensor[] sample = samples.get(indices.get(i));
                if (i < trainSize) {
                    train.add(sample);
                } else {
                    validation.add(sample);
                }
            }
            return new TensorDatasetSplit(new TensorDataset(train), new TensorDataset(validation));
        }

        private static List<GradTensor[]> copySamples(GradTensor[]... samples) {
            Objects.requireNonNull(samples, "samples must not be null");
            List<GradTensor[]> copies = new ArrayList<>(samples.length);
            for (int i = 0; i < samples.length; i++) {
                copies.add(copySample(samples[i], "samples[" + i + "]"));
            }
            return List.copyOf(copies);
        }

        private static List<GradTensor[]> copySamples(List<GradTensor[]> samples) {
            Objects.requireNonNull(samples, "samples must not be null");
            List<GradTensor[]> copies = new ArrayList<>(samples.size());
            for (int i = 0; i < samples.size(); i++) {
                copies.add(copySample(samples.get(i), "samples[" + i + "]"));
            }
            return List.copyOf(copies);
        }

        private static GradTensor[] copySample(GradTensor[] sample, String name) {
            Objects.requireNonNull(sample, name + " must not be null");
            GradTensor[] copy = Arrays.copyOf(sample, sample.length);
            for (int i = 0; i < copy.length; i++) {
                Objects.requireNonNull(copy[i], name + "[" + i + "] must not be null");
            }
            return copy;
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
        private final TensorDatasetAdapter dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private boolean dropLast = false;
        private Long shuffleSeed;
        private boolean reshuffleEachEpoch;
        private long initialEpoch;
        private IndexSampler sampler;
        private BatchSampler batchSampler;
        private CollateFn collateFn;

        TensorBuilder(TensorDatasetAdapter dataset) {
            this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        }

        public TensorBuilder batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
            }
            this.batchSize = batchSize;
            return this;
        }

        public TensorBuilder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            if (!shuffle) {
                this.reshuffleEachEpoch = false;
            }
            return this;
        }

        public TensorBuilder dropLast(boolean dropLast) {
            this.dropLast = dropLast;
            return this;
        }

        public TensorBuilder seed(long seed) {
            this.shuffleSeed = seed;
            return this;
        }

        public TensorBuilder randomSeed(long seed) {
            return seed(seed);
        }

        public TensorBuilder reshuffleEachEpoch() {
            return reshuffleEachEpoch(true);
        }

        public TensorBuilder reshuffleEachEpoch(boolean reshuffleEachEpoch) {
            this.reshuffleEachEpoch = reshuffleEachEpoch;
            if (reshuffleEachEpoch) {
                this.shuffle = true;
            }
            return this;
        }

        public TensorBuilder initialEpoch(long initialEpoch) {
            if (initialEpoch < 0L) {
                throw new IllegalArgumentException("initialEpoch must be non-negative, got: " + initialEpoch);
            }
            this.initialEpoch = initialEpoch;
            return this;
        }

        public TensorBuilder startEpoch(long initialEpoch) {
            return initialEpoch(initialEpoch);
        }

        public TensorBuilder sampler(IndexSampler sampler) {
            this.sampler = Objects.requireNonNull(sampler, "sampler must not be null");
            this.batchSampler = null;
            return this;
        }

        public TensorBuilder batchSampler(BatchSampler batchSampler) {
            this.batchSampler = Objects.requireNonNull(batchSampler, "batchSampler must not be null");
            this.sampler = null;
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
            return sampler(DataLoader.lengthBucketBatchSampler(lengths, batchSize, dropLast, seed));
        }

        public TensorBuilder lengthBucketBatchSampler(ToIntFunction<GradTensor[]> lengthExtractor, long seed) {
            return lengthBucketBatchSampler(tensorDatasetLengths(lengthExtractor), seed);
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
                    batchSize,
                    bucketSizeMultiplier,
                    shuffleBatches,
                    shuffleWithinBuckets,
                    dropLast,
                    seed));
        }

        public TensorBuilder lengthBucketBatchSampler(
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
                    tensorDatasetLengths(lengthExtractor),
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
            return tokenBudgetBatchSampler(tensorDatasetLengths(lengthExtractor), maxTokens, seed);
        }

        public TensorBuilder tokenBudgetBatchSampler(int[] lengths, int maxTokens, int maxExamples, long seed) {
            return batchSampler(DataLoader.tokenBudgetBatchSampler(lengths, maxTokens, maxExamples, seed));
        }

        public TensorBuilder tokenBudgetBatchSampler(
                ToIntFunction<GradTensor[]> lengthExtractor,
                int maxTokens,
                int maxExamples,
                long seed) {
            return tokenBudgetBatchSampler(tensorDatasetLengths(lengthExtractor), maxTokens, maxExamples, seed);
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
                    tensorDatasetLengths(lengthExtractor),
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
            this.collateFn = collateFn;
            return this;
        }

        public TensorDataLoader build() {
            return new TensorDataLoader(this);
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

    public static class TensorDataLoader implements Iterable<Batch> {
        private final TensorDatasetAdapter dataset;
        private final int batchSize;
        private final boolean shuffle;
        private final boolean dropLast;
        private final Long shuffleSeed;
        private final boolean reshuffleEachEpoch;
        private final long initialEpoch;
        private final IndexSampler sampler;
        private final BatchSampler batchSampler;
        private final CollateFn collateFn;
        private final AtomicLong epochCounter;

        private TensorDataLoader(TensorBuilder builder) {
            this.dataset = Objects.requireNonNull(builder.dataset, "dataset must not be null");
            this.batchSize = builder.batchSize;
            this.shuffle = builder.shuffle;
            this.dropLast = builder.dropLast;
            this.shuffleSeed = builder.shuffleSeed;
            this.reshuffleEachEpoch = builder.reshuffleEachEpoch;
            if (builder.initialEpoch < 0L) {
                throw new IllegalArgumentException("initialEpoch must be non-negative, got: " + builder.initialEpoch);
            }
            this.initialEpoch = builder.initialEpoch;
            this.epochCounter = new AtomicLong(initialEpoch);
            this.sampler = builder.sampler;
            this.batchSampler = builder.batchSampler;
            this.collateFn = builder.collateFn != null ? builder.collateFn : defaultCollate();
        }

        @Override
        public Iterator<Batch> iterator() {
            return iterator(nextEpoch());
        }

        public Iterable<Batch> epoch(long epoch) {
            requireEpoch(epoch);
            return () -> iterator(epoch);
        }

        private Iterator<Batch> iterator(long epoch) {
            requireEpoch(epoch);
            if (batchSampler != null) {
                return batchSamplerIterator();
            }
            List<Integer> indices = sampler == null
                    ? new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList())
                    : new ArrayList<>(sampler.sample(dataset.size()));

            if (sampler == null && shuffle) {
                Long epochSeed = DataLoaderShuffleSeeds.forEpoch(
                        shuffleSeed,
                        reshuffleEachEpoch,
                        epoch);
                if (epochSeed == null) {
                    Collections.shuffle(indices, ThreadLocalRandom.current());
                } else {
                    Collections.shuffle(indices, new Random(epochSeed));
                }
            }

            int numBatches = numBatches();

            return new Iterator<>() {
                private int current = 0;

                @Override
                public boolean hasNext() {
                    return current < numBatches;
                }

                @Override
                public Batch next() {
                    if (current >= numBatches) {
                        throw new NoSuchElementException();
                    }

                    int start = current * batchSize;
                    int end = Math.min(start + batchSize, indices.size());
                    List<Integer> batchIndices = indices.subList(start, end);

                    current++;
                    return collateFn.collate(batchIndices, dataset);
                }
            };
        }

        public int numBatches() {
            if (batchSampler != null) {
                return batchSampler.batchCount(dataset.size());
            }
            int n = sampleCount();
            return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
        }

        public int size() {
            return dataset.size();
        }

        public int batchSize() {
            return batchSize;
        }

        public int sampleCount() {
            if (batchSampler != null) {
                return batchSampler.sampleCount(dataset.size());
            }
            return sampler == null ? dataset.size() : sampler.sampleCount(dataset.size());
        }

        public boolean sampled() {
            return sampler != null || batchSampler != null;
        }

        public boolean shuffle() {
            return shuffle;
        }

        public boolean dropLast() {
            return dropLast;
        }

        public boolean reshuffleEachEpoch() {
            return reshuffleEachEpoch;
        }

        public long initialEpoch() {
            return initialEpoch;
        }

        private long nextEpoch() {
            return reshuffleEachEpoch ? epochCounter.getAndIncrement() : initialEpoch;
        }

        private static void requireEpoch(long epoch) {
            if (epoch < 0L) {
                throw new IllegalArgumentException("epoch must be non-negative, got: " + epoch);
            }
        }

        private static CollateFn defaultCollate() {
            return TensorCollators.defaultPairCollate();
        }

        private Iterator<Batch> batchSamplerIterator() {
            List<List<Integer>> batches = batchSampler.sampleBatches(dataset.size());
            return new Iterator<>() {
                private int current = 0;

                @Override
                public boolean hasNext() {
                    return current < batches.size();
                }

                @Override
                public Batch next() {
                    if (current >= batches.size()) {
                        throw new NoSuchElementException();
                    }
                    return collateFn.collate(batches.get(current++), dataset);
                }
            };
        }
    }

    @FunctionalInterface
    public interface CollateFn {
        Batch collate(List<Integer> indices, TensorDatasetAdapter dataset);
    }

    public record Batch(GradTensor inputs, GradTensor labels) {
        @Override
        public GradTensor inputs() {
            return inputs;
        }

        @Override
        public GradTensor labels() {
            return labels;
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
            Objects.requireNonNull(inputs, "inputs must not be null");
            Objects.requireNonNull(labels, "labels must not be null");
            Objects.requireNonNull(inputMask, "inputMask must not be null");
            Objects.requireNonNull(labelMask, "labelMask must not be null");
            inputLengths = Objects.requireNonNull(inputLengths, "inputLengths must not be null").clone();
            labelLengths = Objects.requireNonNull(labelLengths, "labelLengths must not be null").clone();
            validatePaddedTensor("inputs", inputs, inputMask, inputLengths);
            validatePaddedTensor("labels", labels, labelMask, labelLengths);
        }

        public Batch batch() {
            return new Batch(inputs, labels);
        }

        public PaddingStats inputPaddingStats() {
            return PaddingStats.fromLengths(paddedLength("inputs", inputs), inputLengths);
        }

        public PaddingStats labelPaddingStats() {
            return PaddingStats.fromLengths(paddedLength("labels", labels), labelLengths);
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

        private static void validatePaddedTensor(
                String name,
                GradTensor values,
                GradTensor mask,
                int[] lengths) {
            long[] valueShape = values.shape();
            if (valueShape.length < 2) {
                throw new IllegalArgumentException(name + " must include batch and padded sequence dimensions");
            }
            if (valueShape[0] != lengths.length) {
                throw new IllegalArgumentException(name + " batch dimension must match lengths");
            }
            long[] maskShape = mask.shape();
            if (maskShape.length != 2 || maskShape[0] != valueShape[0]) {
                throw new IllegalArgumentException(name + " mask must be shaped [batch, maxLength]");
            }
            long maxLength = valueShape[1];
            if (maskShape[1] != maxLength) {
                throw new IllegalArgumentException(name + " mask length must match padded sequence length");
            }
            for (int length : lengths) {
                if (length < 0 || length > maxLength) {
                    throw new IllegalArgumentException(name + " lengths must be within the padded sequence length");
                }
            }
        }

        private static int paddedLength(String name, GradTensor values) {
            try {
                return Math.toIntExact(values.shape()[1]);
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException(name + " padded sequence length is too large", e);
            }
        }
    }
}
