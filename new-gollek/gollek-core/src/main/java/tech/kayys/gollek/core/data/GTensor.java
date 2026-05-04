package tech.kayys.gollek.core.data;

import tech.kayys.gollek.core.tensor.Tensor;

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
