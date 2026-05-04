package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.core.graph.Node;
import tech.kayys.gollek.core.tensor.Device;
import java.util.List;

public final class Partition {
    private final String id;
    private final Device device;
    private final List<Node> steps;

    public Partition(String id, Device device, List<Node> steps) {
        this.id = id;
        this.device = device;
        this.steps = steps;
    }

    public String id() { return id; }
    public Device device() { return device; }
    public List<Node> steps() { return steps; }
}
