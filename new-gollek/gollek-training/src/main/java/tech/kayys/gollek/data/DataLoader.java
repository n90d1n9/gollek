package tech.kayys.gollek.data;

import java.util.*;

public final class DataLoader {
    private final Dataset<Batch> dataset;

    public DataLoader(Dataset<Batch> dataset) {
        this.dataset = dataset;
    }

    public Iterable<Batch> batches() {
        return dataset;
    }
}