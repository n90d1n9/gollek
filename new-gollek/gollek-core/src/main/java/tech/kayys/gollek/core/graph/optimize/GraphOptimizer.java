package tech.kayys.gollek.core.graph.optimize;

import tech.kayys.gollek.core.graph.Node;

public interface GraphOptimizer {
    Node optimize(Node root);
}