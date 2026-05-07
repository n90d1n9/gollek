package tech.kayys.gollek.compiler.passes;

import tech.kayys.gollek.compiler.*;
import tech.kayys.gollek.ir.*;
import java.util.*;

public final class AttentionLoweringPass implements Pass {
    @Override
    public String name() {
        return "attention_lowering";
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        List<GOp> newOps = new ArrayList<>();
        for (GOp op : graph.ops()) {
            if (!op.opType().equals("attention")) {
                newOps.add(op);
                continue;
            }
            GValueRef q = op.inputs().get(0);
            GValueRef k = op.inputs().get(1);
            GValueRef v = op.inputs().get(2);
            GValueId qk = new GValueId(op.name() + "_qk");
            GValueId sm = new GValueId(op.name() + "_sm");
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("matmul")),
                    op.name() + "_qk",
                    List.of(q, k),
                    List.of(qk),
                    Map.of()));
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("softmax")),
                    op.name() + "_softmax",
                    List.of(new GValueRef(qk)),
                    List.of(sm),
                    Map.of()));
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("matmul")),
                    op.name() + "_out",
                    List.of(new GValueRef(sm), v),
                    op.outputs(),
                    Map.of()));
        }
        return new GGraph(newOps, graph.inputs(), graph.outputs());
    }
}