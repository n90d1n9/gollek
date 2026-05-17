package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.stream.Stream;

/**
 * Represents a dataset that streams its samples instead of loading them into memory.
 *
 * @param <T> type of elements in this dataset
 */
public interface StreamingDataset<T> extends Iterable<T> {

    @Override
    Iterator<T> iterator();

    default Stream<T> stream() {
        return java.util.stream.StreamSupport.stream(spliterator(), false);
    }
}
