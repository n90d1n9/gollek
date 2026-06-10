package tech.kayys.gollek.train.data;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

final class GenericDataLoaderRuntime<T> implements Iterable<List<T>> {
    private final Dataset<? extends T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final Long shuffleSeed;
    private final IndexSampler sampler;
    private final BatchSampler batchSampler;
    private final boolean reshuffleEachEpoch;
    private final long initialEpoch;
    private final AtomicLong epochCounter;

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            Long shuffleSeed) {
        this(dataset, batchSize, shuffle, dropLast, shuffleSeed, null);
    }

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            Long shuffleSeed,
            IndexSampler sampler) {
        this(dataset, batchSize, shuffle, dropLast, shuffleSeed, sampler, null);
    }

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            Long shuffleSeed,
            IndexSampler sampler,
            BatchSampler batchSampler) {
        this(dataset, batchSize, shuffle, dropLast, shuffleSeed, sampler, batchSampler, false);
    }

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            Long shuffleSeed,
            IndexSampler sampler,
            BatchSampler batchSampler,
            boolean reshuffleEachEpoch) {
        this(dataset, batchSize, shuffle, dropLast, shuffleSeed, sampler, batchSampler, reshuffleEachEpoch, 0L);
    }

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            Long shuffleSeed,
            IndexSampler sampler,
        BatchSampler batchSampler,
        boolean reshuffleEachEpoch,
        long initialEpoch) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        DataLoaderBatchSizes.requirePositive(batchSize);
        DataLoaderSamplerSelection.requireExclusive(sampler, batchSampler);
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.shuffleSeed = shuffleSeed;
        this.sampler = sampler;
        this.batchSampler = batchSampler;
        this.reshuffleEachEpoch = reshuffleEachEpoch;
        this.initialEpoch = initialEpoch;
        this.epochCounter = DataLoaderEpochs.counter(initialEpoch);
    }

    @Override
    public Iterator<List<T>> iterator() {
        return iterator(nextEpoch());
    }

    Iterator<List<T>> iterator(long epoch) {
        DataLoaderEpochs.requireEpoch(epoch);
        List<List<Integer>> batches = epochBatches(epoch);

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

                List<T> batch = GenericBatchMaterializer.materialize(dataset, batches.get(current));

                current++;
                return batch;
            }
        };
    }

    Iterable<List<T>> epoch(long epoch) {
        DataLoaderEpochs.requireEpoch(epoch);
        return () -> iterator(epoch);
    }

    int numBatches() {
        if (batchSampler != null) {
            return batchSampler.batchCount(dataset.size());
        }
        return TensorBatching.batchCount(sampleCount(), batchSize, dropLast);
    }

    int size() {
        return dataset.size();
    }

    int sampleCount() {
        if (batchSampler != null) {
            return batchSampler.sampleCount(dataset.size());
        }
        return sampler == null ? dataset.size() : sampler.sampleCount(dataset.size());
    }

    boolean sampled() {
        return sampler != null || batchSampler != null;
    }

    int batchSize() {
        return batchSize;
    }

    boolean shuffle() {
        return shuffle;
    }

    boolean dropLast() {
        return dropLast;
    }

    OptionalLong shuffleSeed() {
        return shuffleSeed == null ? OptionalLong.empty() : OptionalLong.of(shuffleSeed);
    }

    boolean reshuffleEachEpoch() {
        return reshuffleEachEpoch;
    }

    long initialEpoch() {
        return initialEpoch;
    }

    DataLoaderPlan plan(String kind) {
        return new DataLoaderPlan(
                kind,
                size(),
                sampleCount(),
                batchSize(),
                numBatches(),
                sampled(),
                shuffle(),
                dropLast(),
                shuffleSeed,
                reshuffleEachEpoch(),
                initialEpoch());
    }

    <R> GenericDataLoaderRuntime<R> map(Function<? super T, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        Dataset<? extends T> source = dataset;
        Dataset<R> mappedDataset = new Dataset<>() {
            @Override
            public R get(int index) {
                return mapper.apply(source.get(index));
            }

            @Override
            public int size() {
                return source.size();
            }
        };
        return new GenericDataLoaderRuntime<>(
                mappedDataset,
                batchSize,
                shuffle,
                dropLast,
                shuffleSeed,
                sampler,
                batchSampler,
                reshuffleEachEpoch,
                initialEpoch);
    }

    private List<List<Integer>> epochBatches(long epoch) {
        if (batchSampler != null) {
            return batchSampler.sampleBatches(dataset.size());
        }
        return TensorBatching.fixedBatches(
                TensorBatching.epochIndices(dataset.size(), sampler, shuffle, shuffleSeed, reshuffleEachEpoch, epoch),
                batchSize,
                dropLast);
    }

    private long nextEpoch() {
        return DataLoaderEpochs.next(epochCounter, reshuffleEachEpoch, initialEpoch);
    }
}
