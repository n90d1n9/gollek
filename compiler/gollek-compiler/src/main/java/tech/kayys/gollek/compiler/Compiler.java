package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.*;

public final class Compiler {
    private final GraphOptimizer optimizer = new GraphOptimizer();
    private final MemoryPlanner memory = new MemoryPlanner();
    private final KernelScheduler scheduler = new KernelScheduler();

    public CompiledGraph compile(GGraph graph) {
        // 1. optimize graph
        GGraph optimized = optimizer.optimize(graph);
        // 2. memory plan
        var memoryPlan = memory.plan(optimized);
        // 3. execution schedule
        var execPlan = scheduler.schedule(optimized);
        return new CompiledGraph(optimized, memoryPlan, execPlan);
    }
}