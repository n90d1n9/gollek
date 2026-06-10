package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Type-safe view over trainer accelerator placement and accelerated-operation counters.
 */
public record TrainingReportAcceleration(
        String requestedDevice,
        String executionBackend,
        String executionDeviceName,
        boolean executionAccelerated,
        boolean requestedDeviceAvailable,
        boolean executionFallback,
        OptionalLong acceleratedMatmulCalls,
        String executionBackendAtStart,
        boolean executionAcceleratedAtStart,
        boolean requestedDeviceAvailableAtStart,
        OptionalLong acceleratedMatmulCallsAtStart,
        OptionalLong acceleratedMatmulCallsDelta,
        boolean acceleratedMatmulUsed,
        boolean executionBackendChanged) {
    public TrainingReportAcceleration {
        requestedDevice = normalizeText(requestedDevice, "auto");
        executionBackend = normalizeText(executionBackend, "unknown");
        executionDeviceName = normalizeText(executionDeviceName, "");
        acceleratedMatmulCalls = acceleratedMatmulCalls == null ? OptionalLong.empty() : acceleratedMatmulCalls;
        executionBackendAtStart = normalizeText(executionBackendAtStart, "");
        acceleratedMatmulCallsAtStart = acceleratedMatmulCallsAtStart == null
                ? OptionalLong.empty()
                : acceleratedMatmulCallsAtStart;
        acceleratedMatmulCallsDelta = acceleratedMatmulCallsDelta == null
                ? OptionalLong.empty()
                : acceleratedMatmulCallsDelta;
    }

    public static TrainingReportAcceleration fromMetadata(Map<String, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new TrainingReportAcceleration(
                stringValue(metadata.get("requestedDevice"), "auto"),
                stringValue(metadata.get("executionBackend"), "unknown"),
                stringValue(metadata.get("executionDeviceName"), ""),
                booleanValue(metadata.get("executionAccelerated")),
                booleanValue(metadata.get("requestedDeviceAvailable")),
                booleanValue(metadata.get("executionFallback")),
                optionalLong(metadata.get("acceleratedMatmulCalls")),
                stringValue(metadata.get("executionBackendAtStart"), ""),
                booleanValue(metadata.get("executionAcceleratedAtStart")),
                booleanValue(metadata.get("requestedDeviceAvailableAtStart")),
                optionalLong(metadata.get("acceleratedMatmulCallsAtStart")),
                optionalLong(metadata.get("acceleratedMatmulCallsDelta")),
                booleanValue(metadata.get("acceleratedMatmulUsed")),
                booleanValue(metadata.get("executionBackendChanged")));
    }

    public boolean available() {
        return !"unknown".equals(executionBackend)
                || !"auto".equals(requestedDevice)
                || acceleratedMatmulCalls.isPresent()
                || acceleratedMatmulCallsDelta.isPresent();
    }

    public boolean explicitAcceleratorRequested() {
        return !requestedDevice.equals("auto") && !requestedDevice.equals("cpu");
    }

    public boolean requestedAcceleratorUnavailable() {
        return explicitAcceleratorRequested() && !requestedDeviceAvailable;
    }

    public boolean requestedAcceleratorFellBack() {
        return explicitAcceleratorRequested() && executionFallback;
    }

    public boolean acceleratedWorkMissing() {
        return executionAccelerated && acceleratedMatmulCallsDelta.isPresent() && !acceleratedMatmulUsed;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available());
        map.put("requestedDevice", requestedDevice);
        map.put("executionBackend", executionBackend);
        map.put("executionDeviceName", executionDeviceName);
        map.put("executionAccelerated", executionAccelerated);
        map.put("requestedDeviceAvailable", requestedDeviceAvailable);
        map.put("executionFallback", executionFallback);
        acceleratedMatmulCalls.ifPresent(value -> map.put("acceleratedMatmulCalls", value));
        if (!executionBackendAtStart.isBlank()) {
            map.put("executionBackendAtStart", executionBackendAtStart);
            map.put("executionAcceleratedAtStart", executionAcceleratedAtStart);
            map.put("requestedDeviceAvailableAtStart", requestedDeviceAvailableAtStart);
        }
        acceleratedMatmulCallsAtStart.ifPresent(value -> map.put("acceleratedMatmulCallsAtStart", value));
        acceleratedMatmulCallsDelta.ifPresent(value -> map.put("acceleratedMatmulCallsDelta", value));
        map.put("acceleratedMatmulUsed", acceleratedMatmulUsed);
        map.put("executionBackendChanged", executionBackendChanged);
        return Map.copyOf(map);
    }

    private static String normalizeText(String value, String fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return value.trim();
    }
}
