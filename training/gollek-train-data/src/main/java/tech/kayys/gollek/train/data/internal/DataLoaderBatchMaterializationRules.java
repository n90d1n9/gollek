package tech.kayys.gollek.train.data.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

public final class DataLoaderBatchMaterializationRules {
    private DataLoaderBatchMaterializationRules() {
    }

    public static <T> List<T> materialize(IntFunction<? extends T> rowReader, List<Integer> batchIndices) {
        Objects.requireNonNull(rowReader, "rowReader must not be null");
        Objects.requireNonNull(batchIndices, "batchIndices must not be null");
        List<T> batch = new ArrayList<>(batchIndices.size());
        for (int index : batchIndices) {
            batch.add(rowReader.apply(index));
        }
        return batch;
    }
}
