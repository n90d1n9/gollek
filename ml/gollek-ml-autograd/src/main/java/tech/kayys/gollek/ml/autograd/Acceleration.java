package tech.kayys.gollek.ml.autograd;

/**
 * Runtime acceleration controls for the CPU-backed GradTensor compatibility
 * layer.
 *
 * <p>The implementation uses optional reflection bridges so the training stack
 * can use Metal/CUDA modules when they are present, while remaining buildable as
 * a pure Java module. CPU fallback is reported honestly and is never counted as
 * GPU acceleration.</p>
 */
public final class Acceleration {

    private Acceleration() {
    }

    public static Scope prefer(String deviceId) {
        return AcceleratedOps.prefer(deviceId);
    }

    public static BackendStatus status() {
        return status(null);
    }

    public static BackendStatus status(String deviceId) {
        return AcceleratedOps.status(deviceId);
    }

    public static long acceleratedMatmulCalls() {
        return AcceleratedOps.acceleratedMatmulCalls();
    }

    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }

    public record BackendStatus(
            String id,
            String deviceName,
            boolean accelerated,
            boolean available,
            long acceleratedMatmulCalls) {
    }

}
