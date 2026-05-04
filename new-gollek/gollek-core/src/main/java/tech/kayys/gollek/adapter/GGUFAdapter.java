package tech.kayys.gollek.adapter;

import tech.kayys.gollek.model.ModelAdapter;
import tech.kayys.gollek.core.tensor.Tensor;

public final class GGUFAdapter implements ModelAdapter {
    @Override
    public Tensor getWeight(String name) {
        // load quantized block
        // convert OR use special kernel
        throw new UnsupportedOperationException("quantized path");
    }

    @Override
    public int numLayers() {
        return 32;
    }

    @Override
    public int hiddenSize() {
        return 4096;
    }

    @Override
    public int numHeads() {
        return 32;
    }
}