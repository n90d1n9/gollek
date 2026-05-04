package tech.kayys.gollek.model;

import tech.kayys.gollek.core.tensor.Tensor;

public interface ModelAdapter {
    Tensor getWeight(String name);
    int numLayers();
    int hiddenSize();
    int numHeads();
}