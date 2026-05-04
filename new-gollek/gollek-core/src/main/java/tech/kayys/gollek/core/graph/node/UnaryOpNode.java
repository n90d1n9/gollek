package tech.kayys.gollek.core.graph.node;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.tensor.Tensor;
import java.util.function.BiFunction;

public final class UnaryOpNode extends AbstractNode {
    private final Node input;
    private final OpUnary op;

    public UnaryOpNode(Node input, OpUnary op) {
        this.input = input;
        this.op = op;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor x = input.eval(ctx);
        return op.apply(ctx, x);

    }
}