package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.Node;

import tech.kayys.gollek.core.graph.util.GraphUtils;

public final class ExecutionPlanner {
    public ExecutionPlan plan(Node root) {
        return new ExecutionPlan(
                GraphUtils.topoSort(root));
    }
}