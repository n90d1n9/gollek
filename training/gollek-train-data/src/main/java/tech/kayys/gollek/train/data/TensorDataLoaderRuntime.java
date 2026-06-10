package tech.kayys.gollek.train.data;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

final class TensorDataLoaderRuntime implements Iterable<DataLoader.Batch> {
    private final DataLoader.TensorDatasetAdapter dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final Long shuffleSeed;
    private final IndexSampler sampler;
    private final BatchSampler batchSampler;
    private final DataLoader.CollateFn collateFn;
    private final boolean reshuffleEachEpoch;
    private final long initialEpoch;
    private final AtomicLong epochCounter;

    TensorDataLoaderRuntime(TensorDataLoaderBuilderSupport<?> builder) {
        this.dataset = Objects.requireNonNull(builder.dataset, "dataset must not be null");
        this.batchSize = builder.batchSize;
        this.shuffle = builder.shuffle;
        this.dropLast = builder.dropLast;
        this.shuffleSeed = builder.shuffleSeed;
        this.sampler = builder.samplerPlan.resolve(builder.batchSize);
        this.batchSampler = builder.batchSampler;
        this.collateFn = builder.collateFn != null ? builder.collateFn : DataLoader.defaultTensorCollate();
        this.reshuffleEachEpoch = builder.reshuffleEachEpoch;
        this.initialEpoch = builder.initialEpoch;
        this.epochCounter = DataLoaderEpochs.counter(initialEpoch);
    }

    @Override
    public Iterator<DataLoader.Batch> iterator() {
        return iterator(nextEpoch());
    }

    Iterator<DataLoader.Batch> iterator(long epoch) {
        DataLoaderEpochs.requireEpoch(epoch);
        List<List<Integer>> batches = epochBatches(epoch);

        return new Iterator<>() {
            private int current = 0;

            @Override
            public boolean hasNext() {
                return current < batches.size();
            }

            @Override
            public DataLoader.Batch next() {
                if (current >= batches.size()) {
                    throw new NoSuchElementException();
                }

                List<Integer> batchIndices = batches.get(current);

                current++;
                return TensorBatchCollator.collate(collateFn, batchIndices, dataset);
            }
        };
    }

    Iterable<DataLoader.Batch> epoch(long epoch) {
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

    int batchSize() {
        return batchSize;
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

    boolean shuffle() {
        return shuffle;
    }

    boolean dropLast() {
        return dropLast;
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

    private List<List<Integer>> epochBatches(long epoch) {
        if (batchSampler != null) {
            return batchSampler.sampleBatches(dataset.size());
        }
        return TensorBatching.fixedBatches(
                TensorBatching.epochIndices(dataset, sampler, shuffle, shuffleSeed, reshuffleEachEpoch, epoch),
                batchSize,
                dropLast);
    }

    private long nextEpoch() {
        return DataLoaderEpochs.next(epochCounter, reshuffleEachEpoch, initialEpoch);
    }
}
