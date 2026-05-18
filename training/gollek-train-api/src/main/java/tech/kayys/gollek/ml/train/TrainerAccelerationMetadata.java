package tech.kayys.gollek.ml.train;

import java.util.Map;
import tech.kayys.gollek.ml.autograd.Acceleration;

/**
 * Publishes execution backend metadata for trainer summaries.
 */
final class TrainerAccelerationMetadata {
    private TrainerAccelerationMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            String requestedDevice,
            Acceleration.BackendStatus currentStatus,
            Acceleration.BackendStatus startStatus) {
        metadata.put("requestedDevice", requestedDevice);
        metadata.put("executionBackend", currentStatus.id());
        metadata.put("executionDeviceName", currentStatus.deviceName());
        metadata.put("executionAccelerated", currentStatus.accelerated());
        metadata.put("requestedDeviceAvailable", currentStatus.available());
        metadata.put("acceleratedMatmulCalls", currentStatus.acceleratedMatmulCalls());
        if (startStatus != null) {
            long startCalls = startStatus.acceleratedMatmulCalls();
            metadata.put("acceleratedMatmulCallsAtStart", startCalls);
            metadata.put(
                    "acceleratedMatmulCallsDelta",
                    Math.max(0L, currentStatus.acceleratedMatmulCalls() - startCalls));
        }
    }
}
