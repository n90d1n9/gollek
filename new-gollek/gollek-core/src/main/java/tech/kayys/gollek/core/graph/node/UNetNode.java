package tech.kayys.gollek.core.graph.node;

public class UNetNode implements Node {
    private final Node latents;
    private final Node embeddings;
    private final float timestep;
    private final DiffusionBackend backend;

    public Tensor eval(ExecutionContext ctx) {
        return backend.runUNet(
                latents.eval(ctx),
                embeddings.eval(ctx),
                timestep);
    }
}