package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.tensor.Tensor;

public abstract class AbstractNode implements Node {
    @Override
    public final Tensor eval(ExecutionContext ctx) {
        Tensor cached = ctx.getCached(this);
        if (cached != null)
            return cached;
        Tensor result = compute(ctx);
        ctx.cache(this, result);
        return result;
    }

    protected abstract Tensor compute(ExecutionContext ctx);
}