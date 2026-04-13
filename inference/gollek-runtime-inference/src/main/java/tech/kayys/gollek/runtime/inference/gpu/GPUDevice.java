package tech.kayys.gollek.runtime.inference.gpu;

/**
 * GPU device information for CUDA backend.
 * <p>
 * Represents a single GPU device with its capabilities, memory,
 * and compute properties. Used for device selection and kernel dispatch.
 *
 * @since 0.3.0
 */
public record GPUDevice(
    /** Device ID (0, 1, 2, ...) */
    int deviceId,

    /** Device name (e.g., "NVIDIA A100-SXM4-80GB") */
    String name,

    /** Total memory in bytes */
    long totalMemory,

    /** Compute capability major version */
    int computeCapabilityMajor,

    /** Compute capability minor version */
    int computeCapabilityMinor,

    /** Number of streaming multiprocessors */
    int smCount,

    /** Max threads per block */
    int maxThreadsPerBlock,

    /** Max threads per SM */
    int maxThreadsPerSM,

    /** Memory bandwidth (GB/s) */
    double memoryBandwidthGbps,

    /** Whether device supports FP16 tensor cores */
    boolean supportsFP16TensorCores,

    /** Whether device supports BF16 tensor cores */
    boolean supportsBF16TensorCores,

    /** Whether device supports FP8 tensor cores (Hopper+) */
    boolean supportsFP8TensorCores
) {
    /**
     * Returns the compute capability as a double (e.g., 8.0, 9.0).
     */
    public double computeCapability() {
        return computeCapabilityMajor + computeCapabilityMinor * 0.1;
    }

    /**
     * Returns whether device is Hopper architecture (compute 9.0+).
     */
    public boolean isHopperOrNewer() {
        return computeCapabilityMajor >= 9;
    }

    /**
     * Returns whether device is Ampere architecture (compute 8.0).
     */
    public boolean isAmpere() {
        return computeCapabilityMajor == 8;
    }

    /**
     * Returns total memory in GB.
     */
    public double totalMemoryGB() {
        return totalMemory / (1024.0 * 1024 * 1024);
    }

    /**
     * Returns available memory in bytes (estimated).
     */
    public long availableMemory(long usedMemory) {
        return totalMemory - usedMemory;
    }

    @Override
    public String toString() {
        return "GPUDevice[%s, %.1fGB, CC %d.%d, %d SMs]".formatted(
            name, totalMemoryGB(), computeCapabilityMajor, computeCapabilityMinor, smCount);
    }
}
