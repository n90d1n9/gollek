package tech.kayys.gollek.compiler;

import java.util.Map;
import tech.kayys.gollek.ir.*;

public final class CompiledGraph {
    public final GGraph graph;
    public final Map<GValueId, MemorySlot> memory;
    public final ExecutionPlan plan;

    public CompiledGraph(GGraph graph,
            Map<GValueId, MemorySlot> memory,
            ExecutionPlan plan) {
        this.graph = graph;
        this.memory = memory;
        this.plan = plan;
    }
}