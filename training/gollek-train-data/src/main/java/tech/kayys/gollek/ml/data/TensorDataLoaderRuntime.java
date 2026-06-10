package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicLong;

final class TensorDataLoaderRuntime implements Iterable<DataLoader.Batch> {
    private final DataLoader.TensorDatasetAdapter dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final Long shuffleSeed;
    private final boolean reshuffleEachEpoch;
    private final long initialEpoch;
    private final IndexSampler sampler;
    private final BatchSampler batchSampler;
    private final DataLoader.CollateFn collateFn;
    private final AtomicLong epochCounter;

    TensorDataLoaderRuntime(TensorDataLoaderConfig config) {
        this.dataset = config.dataset();
        this.batchSize = config.batchSize();
        this.shuffle = config.shuffle();
        this.dropLast = config.dropLast();
        this.shuffleSeed = config.shuffleSeed();
        this.reshuffleEachEpoch = config.reshuffleEachEpoch();
        this.initialEpoch = config.initialEpoch();
        this.sampler = config.sampler();
        this.batchSampler = config.batchSampler();
        this.collateFn = config.collateFn();
        this.epochCounter = DataLoaderEpochs.counter(initialEpoch);
    }

    @Override
    public Iterator<DataLoader.Batch> iterator() {
        return iterator(nextEpoch());
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

    private Iterator<DataLoader.Batch> iterator(long epoch) {
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

                return TensorBatchCollator.collate(collateFn, batches.get(current++), dataset);
            }
        };
    }

    private long nextEpoch() {
        return DataLoaderEpochs.next(epochCounter, reshuffleEachEpoch, initialEpoch);
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
}
