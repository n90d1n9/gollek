package tech.kayys.gollek.ml.runner;

import tech.kayys.gollek.core.tensor.DeviceType;

/**
 * User-facing device selector for unified model runners.
 */
public enum RunnerDevice {
    AUTO(DeviceType.AUTO, "Auto"),
    CPU(DeviceType.CPU, "CPU"),
    CUDA(DeviceType.CUDA, "CUDA"),
    METAL(DeviceType.METAL, "Metal"),
    ROCM(DeviceType.ROCM, "ROCm");

    private final DeviceType deviceType;
    private final String deviceName;

    RunnerDevice(DeviceType deviceType, String deviceName) {
        this.deviceType = deviceType;
        this.deviceName = deviceName;
    }

    public DeviceType toDeviceType() {
        return deviceType;
    }

    public String deviceName() {
        return deviceName;
    }

    public boolean isAccelerated() {
        return this != AUTO && this != CPU;
    }

    public static RunnerDevice fromDeviceType(DeviceType deviceType) {
        if (deviceType == null) {
            return AUTO;
        }
        return switch (deviceType) {
            case CPU -> CPU;
            case CUDA -> CUDA;
            case METAL -> METAL;
            case ROCM -> ROCM;
            default -> AUTO;
        };
    }
}
