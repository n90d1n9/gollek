package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.tensor.Tensor;

public interface OpBinary {
    Tensor apply(ExecutionContext ctx, Tensor a, Tensor b);
}
