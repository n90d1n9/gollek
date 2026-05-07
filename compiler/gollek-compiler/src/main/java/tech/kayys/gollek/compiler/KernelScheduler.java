package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class KernelScheduler {
    public ExecutionPlan schedule(GGraph graph) {
        List<ExecutionPlan.Step> steps = new ArrayList<>();
        for (int i = 0; i < graph.ops().size(); i++) {
            GOp op = graph.ops().get(i);
            steps.add(new ExecutionPlan.Step(
                    op.name(),
                    assignDevice(op),
                    dependencies(graph, op)));
        }
        return new ExecutionPlan(steps);
    }

    private int assignDevice(GOp op) {
        return 0; // extend with distributed logic
    }

    private List<Integer> dependencies(GGraph graph, GOp op) {
        return List.of(); // fill with DAG dependencies
    }
}