package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.tensor.Device;
import tech.kayys.gollek.core.tensor.Tensor;

public interface Node {
    String id();
    Tensor eval(ExecutionContext ctx);
    
    default Device preferredDevice() {
        return Device.CPU; // Default to CPU
    }
}