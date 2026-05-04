package tech.kayys.gollek.core.graph.node;

public class SchedulerNode implements Node {
    private final Node latents;
    private final Node noise;
    private final Scheduler scheduler;
    private final float timestep;

    public Tensor eval(ExecutionContext ctx) {
        return scheduler.step(
                latents.eval(ctx),
                noise.eval(ctx),
                timestep);
    }
}