package tech.kayys.gollek.compiler.rewrite;

import tech.kayys.gollek.ir.GGraph;

public final class RewriteResult {
    public final GGraph graph;
    public final boolean changed;

    public RewriteResult(GGraph graph, boolean changed) {
        this.graph = graph;
        this.changed = changed;
    }
}
