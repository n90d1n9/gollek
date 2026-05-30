package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

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
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        if (sampler != null && batchSampler != null) {
            throw new IllegalArgumentException("sampler and batchSampler cannot both be configured");
        }
        if (initialEpoch < 0L) {
            throw new IllegalArgumentException("initialEpoch must be non-negative, got: " + initialEpoch);
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.shuffleSeed = shuffleSeed;
        this.sampler = sampler;
        this.batchSampler = batchSampler;
        this.reshuffleEachEpoch = reshuffleEachEpoch;
        this.initialEpoch = initialEpoch;
        this.epochCounter = new AtomicLong(initialEpoch);
    }

    @Override
    public Iterator<List<T>> iterator() {
        return iterator(nextEpoch());
    }

    Iterator<List<T>> iterator(long epoch) {
        requireEpoch(epoch);
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

    Iterable<List<T>> epoch(long epoch) {
        requireEpoch(epoch);
        return () -> iterator(epoch);
    }

    int numBatches() {
        if (batchSampler != null) {
            return batchSampler.batchCount(dataset.size());
        }
        int n = sampleCount();
        return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
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
        List<Integer> indices = epochIndices(epoch);
        int numBatches = numBatches();
        List<List<Integer>> batches = new ArrayList<>(numBatches);
        for (int current = 0; current < numBatches; current++) {
            int start = current * batchSize;
            int end = Math.min(start + batchSize, indices.size());
            batches.add(indices.subList(start, end));
        }
        return batches;
    }

    private List<Integer> epochIndices(long epoch) {
        List<Integer> indices = sampler == null
                ? new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList())
                : new ArrayList<>(sampler.sample(dataset.size()));
        if (sampler != null || !shuffle) {
            return indices;
        }
        Long epochSeed = DataLoaderShuffleSeeds.forEpoch(shuffleSeed, reshuffleEachEpoch, epoch);
        if (epochSeed == null) {
            Collections.shuffle(indices, ThreadLocalRandom.current());
        } else {
            Collections.shuffle(indices, new Random(epochSeed));
        }
        return indices;
    }

    private long nextEpoch() {
        return reshuffleEachEpoch ? epochCounter.getAndIncrement() : initialEpoch;
    }

    private static void requireEpoch(long epoch) {
        if (epoch < 0L) {
            throw new IllegalArgumentException("epoch must be non-negative, got: " + epoch);
        }
    }
}
