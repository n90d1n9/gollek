package tech.kayys.gollek.core.graph.optimize;

import tech.kayys.gollek.core.graph.Node;
import java.util.List;

public final class OptimizerPipeline implements GraphOptimizer {
    private final List<GraphOptimizer> passes;

    public OptimizerPipeline(List<GraphOptimizer> passes) {
        this.passes = passes;
    }

    @Override
    public Node optimize(Node root) {
        Node current = root;
        for (GraphOptimizer pass : passes) {
            current = pass.optimize(current);
        }
        return current;
    }
}