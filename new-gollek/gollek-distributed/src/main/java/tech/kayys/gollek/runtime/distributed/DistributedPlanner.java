package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.graph.exec.ExecutionPlan;
import tech.kayys.gollek.core.tensor.Device;

import java.util.ArrayList;
import java.util.List;

public final class DistributedPlanner {

    public PartitionedPlan partition(ExecutionPlan plan) {
        List<Partition> partitions = new ArrayList<>();
        List<Node> currentSteps = new ArrayList<>();
        Device currentDevice = null;
        int id = 0;

        for (Node node : plan.order()) {
            Device preferred = node.preferredDevice();
            
            if (currentDevice == null) {
                currentDevice = preferred;
            }

            // Split when device changes
            if (!preferred.equals(currentDevice)) {
                partitions.add(new Partition(
                    "p" + id++,
                    currentDevice,
                    new ArrayList<>(currentSteps)
                ));
                currentSteps.clear();
                currentDevice = preferred;
            }
            
            currentSteps.add(node);
        }

        if (!currentSteps.isEmpty()) {
            partitions.add(new Partition(
                "p" + id,
                currentDevice,
                new ArrayList<>(currentSteps)
            ));
        }

        return new PartitionedPlan(partitions);
    }
}
