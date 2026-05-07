package tech.kayys.gollek.core.graph.optimize;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;
import tech.kayys.gollek.core.graph.ops.*;

public final class FuseAddMulPass implements GraphOptimizer {
    @Override
    public Node optimize(Node root) {
        if (!(root instanceof UnaryOpNode mulNode))
            return root;
            
        Node input = mulNode.inputs()[0];
        if (input instanceof BinaryOpNode addNode) {
            if (addNode.op() instanceof AddOp &&
                    mulNode.op() instanceof MulScalarOp mulOp) {
                return new FusedAddMulNode(
                        addNode.inputs()[0],
                        addNode.inputs()[1],
                        mulOp.scalar());
            }
        }
        return root;
    }
}