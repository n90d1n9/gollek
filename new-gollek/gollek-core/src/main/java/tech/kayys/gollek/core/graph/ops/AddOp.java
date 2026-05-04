package tech.kayys.gollek.core.graph.ops;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.tensor.Tensor;

public final class AddOp implements OpBinary {
    @Override
    public Tensor apply(ExecutionContext ctx, Tensor a, Tensor b) {
        return ctx.backend().add(a, b);
    }
}