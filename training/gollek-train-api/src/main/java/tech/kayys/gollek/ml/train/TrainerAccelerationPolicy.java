package tech.kayys.gollek.ml.train;

import tech.kayys.gollek.ml.autograd.Acceleration;

/**
 * Shared trainer policy for optional accelerator fallback.
 */
public final class TrainerAccelerationPolicy {
    private TrainerAccelerationPolicy() {
    }

    public static String normalizeDevice(String requestedDevice) {
        return TrainerMetadataSupport.normalizeDevice(requestedDevice);
    }

    public static boolean fallback(String requestedDevice, Acceleration.BackendStatus status) {
        String normalized = normalizeDevice(requestedDevice);
        if ("auto".equals(normalized) || "cpu".equals(normalized)) {
            return false;
        }
        return !status.available() || !normalized.equals(status.id());
    }

    public static void requireNoFallback(
            String requestedDevice,
            Acceleration.BackendStatus status,
            boolean failOnAcceleratorFallback) {
        if (!failOnAcceleratorFallback) {
            return;
        }
        String normalized = normalizeDevice(requestedDevice);
        if ("auto".equals(normalized) && status.accelerated()) {
            return;
        }
        if (!"auto".equals(normalized) && !fallback(requestedDevice, status)) {
            return;
        }
        throw new IllegalStateException(
                "Requested accelerator '" + normalized + "' is unavailable; execution backend resolved to '"
                        + status.id() + "' (" + status.deviceName() + "). "
                        + "Disable strict accelerator fallback or install/enable the requested backend.");
    }
}
