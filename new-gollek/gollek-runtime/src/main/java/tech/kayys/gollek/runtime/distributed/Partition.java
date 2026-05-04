package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.runtime.plan.PlannedStep;
import tech.kayys.gollek.runtime.Device;
import java.util.List;

/**
 * 
 * We extend:
 * ExecutablePlan
 * ↓
 * DistributedPlanner
 * ↓
 * PartitionedPlan (multi-node)
 * ↓
 * DistributedExecutor
 * AND integrate:
 * KV Cache → planner-aware → runtime-efficient
 */
public final class Partition {
    public final String id;
    public final Device device;
    public final List<PlannedStep> steps;

    public Partition(String id, Device device, List<PlannedStep> steps) {
        this.id = id;
        this.device = device;
        this.steps = steps;
    }
}