package tech.kayys.gollek.train.data.internal;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

public final class DataLoaderTensorCollationRules {
    private DataLoaderTensorCollationRules() {
    }

    public static <B, D> B collate(
            BiFunction<List<Integer>, D, B> collateFn,
            List<Integer> batchIndices,
            D dataset) {
        Objects.requireNonNull(collateFn, "collateFn must not be null");
        Objects.requireNonNull(batchIndices, "batchIndices must not be null");
        Objects.requireNonNull(dataset, "dataset must not be null");
        return collateFn.apply(batchIndices, dataset);
    }
}
