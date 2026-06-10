package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
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
        Iterator<List<T>> batches = source.iterator();
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
}
