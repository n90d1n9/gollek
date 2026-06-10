package tech.kayys.gollek.train.data;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.Function;

final class CollatingDataLoaderRuntime<T, B> implements Iterable<B> {
    private final GenericDataLoaderRuntime<T> source;
    private final Function<? super List<T>, ? extends B> collateFn;

    CollatingDataLoaderRuntime(
            GenericDataLoaderRuntime<T> source,
            Function<? super List<T>, ? extends B> collateFn) {
        this.source = Objects.requireNonNull(source, "source must not be null");
        this.collateFn = Objects.requireNonNull(collateFn, "collateFn must not be null");
    }

    @Override
    public Iterator<B> iterator() {
        return iterator(source);
    }

    Iterable<B> epoch(long epoch) {
        return () -> iterator(source.epoch(epoch));
    }

    int numBatches() {
        return source.numBatches();
    }

    int size() {
        return source.size();
    }

    int sampleCount() {
        return source.sampleCount();
    }

    boolean sampled() {
        return source.sampled();
    }

    int batchSize() {
        return source.batchSize();
    }

    boolean shuffle() {
        return source.shuffle();
    }

    boolean dropLast() {
        return source.dropLast();
    }

    OptionalLong shuffleSeed() {
        return source.shuffleSeed();
    }

    boolean reshuffleEachEpoch() {
        return source.reshuffleEachEpoch();
    }

    long initialEpoch() {
        return source.initialEpoch();
    }

    DataLoaderPlan plan() {
        return source.plan("collating");
    }

    private Iterator<B> iterator(Iterable<List<T>> batches) {
        Iterator<List<T>> iterator = batches.iterator();
        return new Iterator<>() {
            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public B next() {
                return collateFn.apply(iterator.next());
            }
        };
    }
}
