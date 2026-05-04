package tech.kayys.gollek.compiler.rules;

import tech.kayys.gollek.compiler.rewrite.*;
import tech.kayys.gollek.ir.*;
import java.util.*;

public final class MatMulAddFusionRule implements RewriteRule {
    @Override
    public boolean matches(GGraph graph, int i) {
        if (i + 1 >= graph.ops().size())
            return false;
        GOp a = graph.ops().get(i);
        GOp b = graph.ops().get(i + 1);
        return a.opType().equals("matmul")
                && b.opType().equals("add")
                && b.inputs().get(0).id().equals(a.outputs().get(0));
    }

    @Override
    public RewriteResult apply(GGraph graph, int i) {
        List<GOp> ops = new ArrayList<>(graph.ops());
        GOp matmul = ops.get(i);
        GOp add = ops.get(i + 1);
        GOp fused = new GOp(
                "matmul_bias",
                matmul.name() + "_fused",
                matmul.inputs(),
                add.outputs(),
                Map.of()
        );
        ops.remove(i + 1);
        ops.set(i, fused);
        return new RewriteResult(
                true,
                new GGraph(ops, graph.inputs(), graph.outputs()),
        );
    }
}