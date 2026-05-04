package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.graph.node.*;
import tech.kayys.gollek.core.graph.ops.*;
import tech.kayys.gollek.core.tensor.Tensor;

public final class GraphBuilder {
    public Node constant(Tensor t) {
        return new ConstantNode(t);
    }

    public Node add(Node a, Node b) {
        return new BinaryOpNode(a, b, new AddOp());
    }

    public Node mul(Node a, float scalar) {
        return new UnaryOpNode(a, new MulScalarOp(scalar));
    }

    public Node matmul(Node a, Node b) {
        return new BinaryOpNode(a, b, new MatMulOp());
    }
}