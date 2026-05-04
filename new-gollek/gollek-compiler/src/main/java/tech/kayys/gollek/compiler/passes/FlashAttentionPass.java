package tech.kayys.gollek.compiler.passes;

public final class FlashAttentionPass implements Pass {
    @Override
    public String name() {
        return "flash_attention";
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        List<GOp> ops = new ArrayList<>();
        for (GOp op : graph.ops()) {
            if (op.opType().equals("attention")) {
                ops.add(new GOp(
                        "flash_attention_v3",
                        op.name(),
                        op.inputs(),
                        op.outputs(),
                        op.attrs()));
            } else {
                ops.add(op);
            }
        }
        return new GGraph(ops, graph.inputs(), graph.outputs());
    }
}