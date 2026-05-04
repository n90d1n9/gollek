package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.tensor.Tensor;

public interface OpUnary {
    Tensor apply(ExecutionContext ctx, Tensor input);
}