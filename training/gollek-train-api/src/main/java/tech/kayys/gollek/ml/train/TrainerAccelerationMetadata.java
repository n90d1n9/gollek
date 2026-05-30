package tech.kayys.gollek.ml.train;

import java.util.Map;
import tech.kayys.gollek.ml.autograd.Acceleration;

/**
 * Publishes execution backend metadata for trainer summaries.
 */
public final class TrainerAccelerationMetadata {
    private TrainerAccelerationMetadata() {
    }

    public static void put(
            Map<String, Object> metadata,
            String requestedDevice,
            Acceleration.BackendStatus currentStatus,
            Acceleration.BackendStatus startStatus) {
        String normalizedRequestedDevice = TrainerMetadataSupport.normalizeDevice(requestedDevice);
        metadata.put("requestedDevice", normalizedRequestedDevice);
        metadata.put("executionBackend", currentStatus.id());
        metadata.put("executionDeviceName", currentStatus.deviceName());
        metadata.put("executionAccelerated", currentStatus.accelerated());
        metadata.put("requestedDeviceAvailable", currentStatus.available());
        metadata.put("acceleratedMatmulCalls", currentStatus.acceleratedMatmulCalls());
        metadata.put("executionFallback", executionFallback(normalizedRequestedDevice, currentStatus));
        if (startStatus != null) {
            long startCalls = startStatus.acceleratedMatmulCalls();
            long delta = Math.max(0L, currentStatus.acceleratedMatmulCalls() - startCalls);
            metadata.put("executionBackendAtStart", startStatus.id());
            metadata.put("executionDeviceNameAtStart", startStatus.deviceName());
            metadata.put("executionAcceleratedAtStart", startStatus.accelerated());
            metadata.put("requestedDeviceAvailableAtStart", startStatus.available());
            metadata.put("acceleratedMatmulCallsAtStart", startCalls);
            metadata.put("acceleratedMatmulCallsDelta", delta);
            metadata.put("acceleratedMatmulUsed", delta > 0L);
            metadata.put("executionBackendChanged", !currentStatus.id().equals(startStatus.id()));
        }
    }

    private static boolean executionFallback(
            String requestedDevice,
            Acceleration.BackendStatus currentStatus) {
        return TrainerAccelerationPolicy.fallback(requestedDevice, currentStatus);
    }
}
