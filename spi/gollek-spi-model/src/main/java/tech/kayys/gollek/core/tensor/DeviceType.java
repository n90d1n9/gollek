package tech.kayys.gollek.core.tensor;

/**
 * Target device for model inference and memory allocation.
 */
public enum DeviceType {
    AUTO("auto", "Auto-select best available device", 0, false),
    CPU("cpu", "CPU", 1, false),
    CUDA("cuda", "NVIDIA CUDA GPU", 10, true),
    ROCM("rocm", "AMD ROCm GPU", 10, true),
    INTEL_GPU("intel-gpu", "Intel GPU", 8, true),
    METAL("metal", "Apple Metal GPU", 9, true),
    TPU("tpu", "Google Cloud TPU", 15, true),
    NPU("npu", "Neural Processing Unit", 12, true),
    INFERENTIA("inferentia", "AWS Inferentia", 14, true),
    AZURE_MAIA("azure-maia", "Azure Maia", 14, true),
    DIRECTML("directml", "DirectML", 8, true),
    WEBGPU("webgpu", "WebGPU", 5, true),
    REMOTE("remote", "Remote gRPC", 0, false),
    TPU_V4("tpu-v4", "Google Cloud TPU v4", 16, true),
    TPU_V5("tpu-v5", "Google Cloud TPU v5", 17, true),
    OPENVINO("openvino", "Intel OpenVINO", 7, true);

    private final String id;
    private final String displayName;
    private final int performanceScore;
    private final boolean requiresSpecializedHardware;

    DeviceType(String id, String displayName, int performanceScore, boolean requiresSpecializedHardware) {
        this.id = id;
        this.displayName = displayName;
        this.performanceScore = performanceScore;
        this.requiresSpecializedHardware = requiresSpecializedHardware;
    }

    public static DeviceType fromId(String id) {
        for (DeviceType type : values()) {
            if (type.id.equalsIgnoreCase(id)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown device type: " + id);
    }

    public boolean isGpu() {
        return this == CUDA || this == ROCM || this == INTEL_GPU || this == METAL || this == DIRECTML || this == WEBGPU;
    }

    public boolean isCloud() {
        return this == TPU || this == TPU_V4 || this == TPU_V5 || this == INFERENTIA || this == AZURE_MAIA;
    }

    public boolean isEdge() {
        return this == CPU || this == NPU || this == METAL;
    }

    public boolean supportsGpu() {
        return isGpu();
    }

    public boolean supportsTpu() {
        return this == TPU || this == TPU_V4 || this == TPU_V5;
    }

    public String getId() {
        return id;
    }

    public String getDisplayName() {
        return displayName;
    }

    public int getPerformanceScore() {
        return performanceScore;
    }

    public boolean isRequiresSpecializedHardware() {
        return requiresSpecializedHardware;
    }

    public boolean isAccelerator() {
        return this == CUDA || this == METAL || this == ROCM || this == TPU || this == NPU;
    }
}
