package tech.kayys.gollek.ml.data;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.train.data.internal.DataLoaderBatchMaterializationRules;

final class GenericBatchMaterializer {
    private GenericBatchMaterializer() {
    }

    static <T> List<T> materialize(Dataset<? extends T> dataset, List<Integer> batchIndices) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return DataLoaderBatchMaterializationRules.materialize(dataset::get, batchIndices);
    }
}
