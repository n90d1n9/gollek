package tech.kayys.gollek.runtime.plan;

import tech.kayys.gollek.ir.*;

/**
 * IR → Compiler → ExecutionPlan
 * ↓
 * Execution Planner
 * ↓
 * 🔥 Cost Model + Placement + Kernel Selection
 * ↓
 * Executable Plan (annotated)
 * ↓
 * ExecutionEngine
 */
public final class PlannedStep {
    public final GOp op;
    public final Device device;
    public final String kernelId;

    public PlannedStep(GOp op, Device device, String kernelId) {
        this.op = op;
        this.device = device;
        this.kernelId = kernelId;
    }
}