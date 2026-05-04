package tech.kayys.gollek.core.graph.node;

public class GuidanceNode implements Node {
    private final Node cond;
    private final Node uncond;
    private final float scale;

    public Tensor eval(ExecutionContext ctx) {
        Tensor u = uncond.eval(ctx);
        Tensor c = cond.eval(ctx);
        return u.add(c.sub(u).mul(scale));
    }
}