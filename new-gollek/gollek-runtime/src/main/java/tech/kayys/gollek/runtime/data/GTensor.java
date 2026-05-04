package tech.kayys.gollek.runtime.data;

import tech.kayys.gollek.ml.tensor.Tensor; // or tech.kayys.gollek.core.tensor.Tensor depending on your module

/**
 * Represents a standard Tensor within the unified runtime execution context.
 */
public final class GTensor implements GData {
    private final String id;
    private final Tensor tensor;

    public GTensor(String id, Tensor tensor) {
        this.id = id;
        this.tensor = tensor;
    }

    public Tensor tensor() {
        return tensor;
    }

    @Override
    public String id() {
        return id;
    }
}
