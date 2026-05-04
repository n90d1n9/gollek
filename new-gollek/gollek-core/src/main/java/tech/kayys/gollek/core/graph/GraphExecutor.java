package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.tensor.Tensor;

public final class GraphExecutor {
    public Tensor run(Node output, ExecutionContext ctx) {
        return output.eval(ctx);
    }
}