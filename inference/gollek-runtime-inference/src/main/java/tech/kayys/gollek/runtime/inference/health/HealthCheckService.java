package tech.kayys.gollek.runtime.inference.health;

import tech.kayys.gollek.runtime.inference.batch.ContinuousBatchScheduler;
import tech.kayys.gollek.runtime.inference.kv.KVCacheManager;
import tech.kayys.gollek.runtime.inference.kv.PagedKVCache;
import tech.kayys.gollek.runtime.inference.kv.GPUMemoryManager;

import io.smallrye.mutiny.Uni;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Health check service for production inference serving.
 * <p>
 * Provides three types of health probes following Kubernetes conventions:
 * <ul>
 *   <li><b>Startup Probe:</b> Is the service still initializing? (model loading, warming)</li>
 *   <li><b>Readiness Probe:</b> Is the service ready to accept traffic? (models loaded, schedulers running)</li>
 *   <li><b>Liveness Probe:</b> Is the service alive? (not deadlocked, responsive)</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * HealthCheckService health = HealthCheckService.builder()
 *     .addModel("llama-3-70b")
 *     .addScheduler("llama-3-70b", continuousBatchScheduler)
 *     .addKVCacheManager(kvCacheManager)
 *     .startupTimeout(Duration.ofMinutes(5))
 *     .build();
 *
 * // Check startup
 * HealthResult startup = health.checkStartup();
 * if (!startup.isHealthy()) {
 *     // Service still starting up, don't send traffic yet
 * }
 *
 * // Check readiness (for load balancer)
 * HealthResult readiness = health.checkReadiness();
 * if (!readiness.isHealthy()) {
 *     // Remove from load balancer pool
 * }
 *
 * // Check liveness (for container orchestrator)
 * HealthResult liveness = health.checkLiveness();
 * if (!liveness.isHealthy()) {
 *     // Restart container
 * }
 * }</pre>
 *
 * @since 0.2.0
 */
public interface HealthCheckService {

    /**
     * Checks if the service has completed startup initialization.
     * <p>
     * Returns unhealthy if:
     * <ul>
     *   <li>Models are still loading</li>
     *   <li>Schedulers are not started</li>
     *   <li>Startup timeout exceeded</li>
     * </ul>
     *
     * @return health result with details
     */
    HealthResult checkStartup();

    /**
     * Checks if the service is ready to accept traffic.
     * <p>
     * Returns unhealthy if:
     * <ul>
     *   <li>No models are loaded</li>
     *   <li>Any registered scheduler is not running</li>
     *   <li>KV cache manager is closed</li>
     * </ul>
     *
     * @return health result with details
     */
    HealthResult checkReadiness();

    /**
     * Checks if the service is alive.
     * <p>
     * Returns unhealthy if:
     * <ul>
     *   <li>Service is deadlocked</li>
     *   <li>Memory manager is closed</li>
     *   <li>Critical component failed</li>
     * </ul>
     *
     * @return health result with details
     */
    HealthResult checkLiveness();

    /**
     * Gets comprehensive health status.
     *
     * @return detailed health information
     */
    HealthDetails getHealthDetails();

    // ── Model Registration ─────────────────────────────────────────────

    /**
     * Registers a model for health tracking.
     */
    void registerModel(String modelId);

    /**
     * Marks a model as loaded.
     */
    void markModelLoaded(String modelId);

    /**
     * Marks a model as warmed (ready for inference).
     */
    void markModelWarmed(String modelId);

    /**
     * Creates a builder for configuring this service.
     */
    static Builder builder() {
        return new Builder();
    }

    // ── Nested Types ───────────────────────────────────────────────────

    /**
     * Result of a health check.
     */
    record HealthResult(
        boolean healthy,
        String component,
        String status,
        String message,
        LocalDateTime timestamp,
        Map<String, Object> details
    ) {
        public static HealthResult healthy(String component, String message) {
            return new HealthResult(true, component, "healthy", message,
                LocalDateTime.now(), Map.of());
        }

        public static HealthResult unhealthy(String component, String message) {
            return new HealthResult(false, component, "unhealthy", message,
                LocalDateTime.now(), Map.of());
        }

        public static HealthResult unhealthy(String component, String message,
                                            Map<String, Object> details) {
            return new HealthResult(false, component, "unhealthy", message,
                LocalDateTime.now(), details);
        }
    }

    /**
     * Comprehensive health details.
     */
    record HealthDetails(
        HealthResult startup,
        HealthResult readiness,
        HealthResult liveness,
        Map<String, ModelHealth> modelHealth,
        Map<String, SchedulerHealth> schedulerHealth,
        MemoryHealth memoryHealth,
        LocalDateTime timestamp
    ) {
    }

    /**
     * Per-model health status.
     */
    record ModelHealth(
        String modelId,
        boolean loaded,
        boolean warmed,
        int loadedVersions,
        LocalDateTime loadedAt,
        long inferenceCount,
        double avgLatencyMs,
        String status
    ) {
    }

    /**
     * Per-scheduler health status.
     */
    record SchedulerHealth(
        String modelId,
        boolean running,
        int activeRequests,
        int pendingRequests,
        long totalProcessed,
        long totalRejected,
        double avgBatchUtilization,
        String status
    ) {
    }

    /**
     * Memory health status.
     */
    record MemoryHealth(
        long totalMemory,
        long allocatedMemory,
        long availableMemory,
        double utilization,
        int kvCacheBlocksUsed,
        int kvCacheBlocksTotal,
        double kvCacheUtilization,
        boolean oomRisk,
        String status
    ) {
    }

    /**
     * Builder for HealthCheckService.
     */
    class Builder {
        private final DefaultHealthCheckService.Builder delegate = DefaultHealthCheckService.builder();

        public Builder startupTimeout(Duration timeout) {
            delegate.startupTimeout(timeout);
            return this;
        }

        public Builder kvCacheManager(KVCacheManager kvCacheManager) {
            delegate.kvCacheManager(kvCacheManager);
            return this;
        }

        public Builder gpuMemoryManager(GPUMemoryManager gpuMemoryManager) {
            delegate.gpuMemoryManager(gpuMemoryManager);
            return this;
        }

        public HealthCheckService build() {
            return delegate.build();
        }
    }
}
