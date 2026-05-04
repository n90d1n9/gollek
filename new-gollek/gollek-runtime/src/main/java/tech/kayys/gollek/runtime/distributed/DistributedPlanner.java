package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.runtime.plan.*;
import tech.kayys.gollek.runtime.Device;

import java.util.*;

public final class DistributedPlanner {
    public PartitionedPlan partition(ExecutablePlan plan) {
        List<Partition> partitions = new ArrayList<>();
        List<PlannedStep> current = new ArrayList<>();
        Device currentDevice = null;
        int id = 0;
        for (PlannedStep step : plan.steps) {
            if (currentDevice == null) {
                currentDevice = step.device;
            }
            // split when device changes
            if (step.device != currentDevice) {
                partitions.add(new Partition(
                        "p" + id++,
                        currentDevice,
                        List.copyOf(current)));
                current.clear();
                currentDevice = step.device;
            }
            current.add(step);
        }

        if (!current.isEmpty()) {
            partitions.add(new Partition(
                    "p" + id,
                    currentDevice,
                    List.copyOf(current)));
        }
        return new PartitionedPlan(partitions);
    }
}