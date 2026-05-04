package tech.kayys.gollek.multimodal.monitoring;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.metrics.annotation.Counted;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monitoring service for multimodal inference.
 * Collects metrics, manages dashboards, and triggers alerts.
 */
@ApplicationScoped
public class MultimodalMonitoringService {

    private static final Logger log = Logger.getLogger(MultimodalMonitoringService.class);

    // Counters
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong successfulRequests = new AtomicLong(0);
    private final AtomicLong failedRequests = new AtomicLong(0);
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);

    // Latency tracking
    private final AtomicLong totalLatencyMs = new AtomicLong(0);
    private final AtomicLong maxLatencyMs = new AtomicLong(0);
    private final AtomicLong p95LatencyMs = new AtomicLong(0);
    private final AtomicLong p99LatencyMs = new AtomicLong(0);

    // Concurrency tracking
    private final AtomicLong activeRequests = new AtomicLong(0);
    private final AtomicLong activeStreams = new AtomicLong(0);
    private final AtomicLong maxConcurrentRequests = new AtomicLong(0);

    // Resource tracking
    private final AtomicLong memoryUsedBytes = new AtomicLong(0);
    private final AtomicLong gpuMemoryUsedBytes = new AtomicLong(0);

    // Error tracking
    private final Map<String, AtomicLong> errorsByType = new ConcurrentHashMap<>();
    private final AtomicLong timeoutErrors = new AtomicLong(0);
    private final AtomicLong validationErrors = new AtomicLong(0);
    private final AtomicLong systemErrors = new AtomicLong(0);

    // Performance tracking
    private final AtomicLong requestsPerSecond = new AtomicLong(0);
    private final AtomicLong tokensPerSecond = new AtomicLong(0);
    private Instant lastResetTime = Instant.now();

    /**
     * Record a new request.
     */
    @Counted(name = "multimodal_requests_total", description = "Total number of requests")
    public void recordRequest() {
        totalRequests.incrementAndGet();
        activeRequests.incrementAndGet();

        // Update max concurrent
        long current = activeRequests.get();
        maxConcurrentRequests.updateAndGet(max -> Math.max(max, current));

        // Calculate RPS
        updateRequestsPerSecond();
    }

    /**
     * Record a successful request completion.
     */
    public void recordSuccess(long latencyMs, int tokensProcessed) {
        successfulRequests.incrementAndGet();
        activeRequests.decrementAndGet();
        totalLatencyMs.addAndGet(latencyMs);
        totalTokensProcessed.addAndGet(tokensProcessed);

        // Update max latency
        maxLatencyMs.updateAndGet(max -> Math.max(max, latencyMs));

        // Update percentiles
        updateLatencyPercentiles(latencyMs);

        // Calculate TPS
        updateTokensPerSecond(tokensProcessed);
    }

    /**
     * Record a failed request.
     */
    public void recordFailure(String errorType) {
        failedRequests.incrementAndGet();
        activeRequests.decrementAndGet();

        // Track error by type
        errorsByType.computeIfAbsent(errorType, k -> new AtomicLong(0))
                .incrementAndGet();

        // Track specific error types
        switch (errorType.toLowerCase()) {
            case "timeout":
                timeoutErrors.incrementAndGet();
                break;
            case "validation":
                validationErrors.incrementAndGet();
                break;
            default:
                systemErrors.incrementAndGet();
        }

        log.errorf("Request failed: %s", errorType);
    }

    /**
     * Record stream start.
     */
    public void recordStreamStart() {
        activeStreams.incrementAndGet();
    }

    /**
     * Record stream completion.
     */
    public void recordStreamEnd() {
        activeStreams.decrementAndGet();
    }

    /**
     * Record GPU memory usage.
     */
    public void recordGpuMemoryUsage(long bytes) {
        gpuMemoryUsedBytes.set(bytes);
    }

    /**
     * Record system memory usage.
     */
    public void recordMemoryUsage(long bytes) {
        memoryUsedBytes.set(bytes);
    }

    /**
     * Get current metrics snapshot.
     */
    public MonitoringMetrics getMetrics() {
        return new MonitoringMetrics(
                totalRequests.get(),
                successfulRequests.get(),
                failedRequests.get(),
                calculateErrorRate(),
                totalLatencyMs.get(),
                maxLatencyMs.get(),
                p95LatencyMs.get(),
                p99LatencyMs.get(),
                activeRequests.get(),
                activeStreams.get(),
                maxConcurrentRequests.get(),
                totalTokensProcessed.get(),
                requestsPerSecond.get(),
                tokensPerSecond.get(),
                memoryUsedBytes.get(),
                gpuMemoryUsedBytes.get(),
                timeoutErrors.get(),
                validationErrors.get(),
                systemErrors.get(),
                errorsByType.entrySet().stream()
                        .collect(java.util.stream.Collectors.toMap(
                                Map.Entry::getKey,
                                e -> e.getValue().get())),
                lastResetTime);
    }

    /**
     * Get health status.
     */
    public HealthStatus getHealthStatus() {
        double errorRate = calculateErrorRate();
        long currentP95 = p95LatencyMs.get();
        long currentMemory = memoryUsedBytes.get();

        HealthStatus status = new HealthStatus();
        status.timestamp = Instant.now();
        status.overall = "UP";

        // Check error rate
        if (errorRate > 10) {
            status.errorRate = "CRITICAL";
            status.overall = "DOWN";
        } else if (errorRate > 5) {
            status.errorRate = "WARNING";
        } else {
            status.errorRate = "OK";
        }

        // Check latency
        if (currentP95 > 5000) {
            status.latency = "CRITICAL";
            status.overall = "DEGRADED";
        } else if (currentP95 > 2000) {
            status.latency = "WARNING";
        } else {
            status.latency = "OK";
        }

        // Check memory
        double memoryPercent = (double) currentMemory / Runtime.getRuntime().maxMemory() * 100;
        if (memoryPercent > 90) {
            status.memory = "CRITICAL";
            status.overall = "DEGRADED";
        } else if (memoryPercent > 80) {
            status.memory = "WARNING";
        } else {
            status.memory = "OK";
        }

        return status;
    }

    /**
     * Get alert status.
     */
    public java.util.List<Alert> getActiveAlerts() {
        java.util.List<Alert> alerts = new java.util.ArrayList<>();

        double errorRate = calculateErrorRate();
        long currentP95 = p95LatencyMs.get();
        long currentP99 = p99LatencyMs.get();
        double memoryPercent = (double) memoryUsedBytes.get() / Runtime.getRuntime().maxMemory() * 100;

        // Error rate alerts
        if (errorRate > 10) {
            alerts.add(new Alert("ERROR_RATE_CRITICAL", "Error rate > 10%", "CRITICAL"));
        } else if (errorRate > 5) {
            alerts.add(new Alert("ERROR_RATE_WARNING", "Error rate > 5%", "WARNING"));
        }

        // Latency alerts
        if (currentP99 > 5000) {
            alerts.add(new Alert("LATENCY_P99_CRITICAL", "P99 latency > 5s", "CRITICAL"));
        } else if (currentP99 > 3000) {
            alerts.add(new Alert("LATENCY_P99_WARNING", "P99 latency > 3s", "WARNING"));
        }

        if (currentP95 > 3000) {
            alerts.add(new Alert("LATENCY_P95_CRITICAL", "P95 latency > 3s", "CRITICAL"));
        } else if (currentP95 > 2000) {
            alerts.add(new Alert("LATENCY_P95_WARNING", "P95 latency > 2s", "WARNING"));
        }

        // Memory alerts
        if (memoryPercent > 90) {
            alerts.add(new Alert("MEMORY_CRITICAL", "Memory usage > 90%", "CRITICAL"));
        } else if (memoryPercent > 80) {
            alerts.add(new Alert("MEMORY_WARNING", "Memory usage > 80%", "WARNING"));
        }

        // Throughput alerts
        if (requestsPerSecond.get() < 10) {
            alerts.add(new Alert("THROUGHPUT_LOW", "Throughput < 10 req/s", "WARNING"));
        }

        return alerts;
    }

    /**
     * Reset metrics.
     */
    public void resetMetrics() {
        log.info("Resetting metrics");

        totalRequests.set(0);
        successfulRequests.set(0);
        failedRequests.set(0);
        totalTokensProcessed.set(0);
        totalLatencyMs.set(0);
        maxLatencyMs.set(0);
        p95LatencyMs.set(0);
        p99LatencyMs.set(0);
        activeRequests.set(0);
        activeStreams.set(0);
        maxConcurrentRequests.set(0);
        memoryUsedBytes.set(0);
        gpuMemoryUsedBytes.set(0);
        timeoutErrors.set(0);
        validationErrors.set(0);
        systemErrors.set(0);
        errorsByType.clear();
        requestsPerSecond.set(0);
        tokensPerSecond.set(0);
        lastResetTime = Instant.now();
    }

    // Private helper methods

    private double calculateErrorRate() {
        long total = totalRequests.get();
        if (total == 0)
            return 0.0;
        return (double) failedRequests.get() / total * 100.0;
    }

    private void updateRequestsPerSecond() {
        Instant now = Instant.now();
        Duration elapsed = Duration.between(lastResetTime, now);

        if (elapsed.getSeconds() >= 1) {
            long rps = totalRequests.get() / elapsed.getSeconds();
            requestsPerSecond.set(rps);
        }
    }

    private void updateTokensPerSecond(int tokens) {
        Instant now = Instant.now();
        Duration elapsed = Duration.between(lastResetTime, now);

        if (elapsed.getSeconds() >= 1) {
            long tps = totalTokensProcessed.get() / elapsed.getSeconds();
            tokensPerSecond.set(tps);
        }
    }

    private void updateLatencyPercentiles(long latencyMs) {
        // Simplified percentile calculation
        // In production, use proper histogram
        long current = totalRequests.get();
        if (current % 100 == 0) {
            p95LatencyMs.set(maxLatencyMs.get() * 95 / 100);
            p99LatencyMs.set(maxLatencyMs.get() * 99 / 100);
        }
    }

    // Metric classes

    public static class MonitoringMetrics {
        public final long totalRequests;
        public final long successfulRequests;
        public final long failedRequests;
        public final double errorRate;
        public final long totalLatencyMs;
        public final long maxLatencyMs;
        public final long p95LatencyMs;
        public final long p99LatencyMs;
        public final long activeRequests;
        public final long activeStreams;
        public final long maxConcurrentRequests;
        public final long totalTokensProcessed;
        public final long requestsPerSecond;
        public final long tokensPerSecond;
        public final long memoryUsedBytes;
        public final long gpuMemoryUsedBytes;
        public final long timeoutErrors;
        public final long validationErrors;
        public final long systemErrors;
        public final Map<String, Long> errorsByType;
        public final Instant lastResetTime;

        public MonitoringMetrics(long totalRequests, long successfulRequests, long failedRequests,
                double errorRate, long totalLatencyMs, long maxLatencyMs,
                long p95LatencyMs, long p99LatencyMs, long activeRequests,
                long activeStreams, long maxConcurrentRequests, long totalTokensProcessed,
                long requestsPerSecond, long tokensPerSecond, long memoryUsedBytes,
                long gpuMemoryUsedBytes, long timeoutErrors, long validationErrors,
                long systemErrors, Map<String, Long> errorsByType, Instant lastResetTime) {
            this.totalRequests = totalRequests;
            this.successfulRequests = successfulRequests;
            this.failedRequests = failedRequests;
            this.errorRate = errorRate;
            this.totalLatencyMs = totalLatencyMs;
            this.maxLatencyMs = maxLatencyMs;
            this.p95LatencyMs = p95LatencyMs;
            this.p99LatencyMs = p99LatencyMs;
            this.activeRequests = activeRequests;
            this.activeStreams = activeStreams;
            this.maxConcurrentRequests = maxConcurrentRequests;
            this.totalTokensProcessed = totalTokensProcessed;
            this.requestsPerSecond = requestsPerSecond;
            this.tokensPerSecond = tokensPerSecond;
            this.memoryUsedBytes = memoryUsedBytes;
            this.gpuMemoryUsedBytes = gpuMemoryUsedBytes;
            this.timeoutErrors = timeoutErrors;
            this.validationErrors = validationErrors;
            this.systemErrors = systemErrors;
            this.errorsByType = errorsByType;
            this.lastResetTime = lastResetTime;
        }
    }

    public static class HealthStatus {
        public Instant timestamp;
        public String overall;
        public String errorRate;
        public String latency;
        public String memory;

        @Override
        public String toString() {
            return String.format("HealthStatus{overall=%s, errorRate=%s, latency=%s, memory=%s}",
                    overall, errorRate, latency, memory);
        }
    }

    public static class Alert {
        public final String code;
        public final String message;
        public final String severity;
        public final Instant timestamp;

        public Alert(String code, String message, String severity) {
            this.code = code;
            this.message = message;
            this.severity = severity;
            this.timestamp = Instant.now();
        }

        @Override
        public String toString() {
            return String.format("Alert[%s] %s - %s", severity, code, message);
        }
    }
}
