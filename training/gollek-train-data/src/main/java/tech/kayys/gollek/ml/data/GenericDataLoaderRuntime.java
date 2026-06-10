package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Function;

final class GenericDataLoaderRuntime<T> implements Iterable<List<T>> {
    private final Dataset<? extends T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final IndexSampler sampler;
    private final BatchSampler batchSampler;

    GenericDataLoaderRuntime(
            Dataset<? extends T> dataset,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            IndexSampler sampler,
            BatchSampler batchSampler) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        DataLoaderBatchSizes.requirePositive(batchSize);
        DataLoaderSamplerSelection.requireExclusive(sampler, batchSampler);
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.sampler = sampler;
        this.batchSampler = batchSampler;
    }

    @Override
    public Iterator<List<T>> iterator() {
        List<List<Integer>> batches = epochBatches();

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
                sampler,
                batchSampler);
    }

    private List<List<Integer>> epochBatches() {
        if (batchSampler != null) {
            return batchSampler.sampleBatches(dataset.size());
        }
        return TensorBatching.fixedBatches(
                TensorBatching.epochIndices(dataset.size(), sampler, shuffle, null, false, 0L),
                batchSize,
                dropLast);
    }

}
