package tech.kayys.gollek.model;

import tech.kayys.gollek.ir.GGraph;
import tech.kayys.gollek.runtime.weight.WeightStore;

public final class LoadedModel {
    public final GGraph graph;
    public final WeightStore weights;

    public LoadedModel(GGraph graph, WeightStore weights) {
        this.graph = graph;
        this.weights = weights;
    }
}