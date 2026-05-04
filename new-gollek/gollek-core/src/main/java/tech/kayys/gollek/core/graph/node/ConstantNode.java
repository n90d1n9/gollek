package tech.kayys.gollek.core.graph.node;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.tensor.Tensor;

public final class ConstantNode extends AbstractNode {
    private final Tensor value;

    public ConstantNode(Tensor value) {
        this.value = value;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        return value;
    }
}