package tech.kayys.gollek.compiler;

import java.util.*;

public final class ExecutionPlan {
    public static final class Step {
        public final String opName;
        public final int device;
        public final List<Integer> deps;

        public Step(String opName, int device, List<Integer> deps) {
            this.opName = opName;
            this.device = device;
            this.deps = deps;
        }
    }

    private final List<Step> steps;

    public ExecutionPlan(List<Step> steps) {
        this.steps = steps;

    }

    public List<Step> steps() {
        return steps;
    }
}