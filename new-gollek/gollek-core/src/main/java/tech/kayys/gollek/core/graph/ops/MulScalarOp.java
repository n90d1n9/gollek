package tech.kayys.gollek.core.graph.ops;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.tensor.Tensor;

public final class MulScalarOp implements OpUnary {
    private final float scalar;

    public MulScalarOp(float scalar) {
        this.scalar = scalar;
    }

    @Override
    public Tensor apply(ExecutionContext ctx, Tensor input) {
        return ctx.backend().mul(input, scalar);
    }
}