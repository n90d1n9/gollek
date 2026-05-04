package tech.kayys.gollek.compiler.rewrite;

import tech.kayys.gollek.compiler.*;
import tech.kayys.gollek.ir.*;
import java.util.List;

public final class RewritePass implements Pass {
    private final String name;
    private final List<RewriteRule> rules;

    public RewritePass(String name, List<RewriteRule> rules) {
        this.name = name;
        this.rules = rules;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < graph.ops().size(); i++) {
                for (RewriteRule rule : rules) {
                    if (rule.matches(graph, i)) {
                        RewriteResult res = rule.apply(graph, i);
                        graph = res.graph;
                        changed |= res.changed;
                    }
                }
            }
        } while (changed);
        return graph;
    }
}