package tech.kayys.gollek.spi.spec;
import tech.kayys.aljabr.core.tensor.DeviceType;

/**
 * Represents the resource requirements for running a model.
 */
public record ResourceRequirements(
        MemoryRequirements memory,
        ComputeRequirements compute,
        StorageRequirements storage) {
}