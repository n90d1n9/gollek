package tech.kayys.gollek.runtime.inference.observability;

import org.jboss.logging.Logger;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * LLM inference observability tracer.
 * <p>
 * Provides comprehensive observability for production LLM inference:
 * <ul>
 *   <li><b>Distributed Tracing:</b> Per-request spans with detailed attributes</li>
 *   <li><b>Token-Level Metrics:</b> Input/output tokens, TTFT, throughput</li>
 *   <li><b>KV Cache Tracking:</b> Memory usage, compression ratios</li>
 *   <li><b>Rate Limiting Metrics:</b> Allowed/rejected counts per tenant</li>
 *   <li><b>Real-Time Dashboard:</b> Current state of inference serving</li>
 * </ul>
 *
 * <h2>Integration</h2>
 * <pre>{@code
 * LLMObservability observability = LLMObservability.builder()
 *     .serviceName("gollek-inference")
 *     .enableTracing(true)
 *     .enableMetrics(true)
 *     .build();
 *
 * // Trace an inference request
 * InferenceTrace trace = observability.startTrace(
 *     "llama-3-70b", "tenant-123", "req-456");
 *
 * try {
 *     // Process request...
 *     trace.recordTokens(128, 256);  // input, output
 *     trace.recordTTFT(45);  // 45ms
 *     trace.recordKVCacheUsage(512, 2048);  // blocks, bytes
 *     trace.recordSuccess();
 * } catch (Exception e) {
 *     trace.recordError(e);
 * } finally {
 *     trace.end();
 * }
 *
 * // Get real-time dashboard
 * InferenceDashboard dashboard = observability.getDashboard();
 * }</pre>
 *
 * @since 0.2.0
 */
public final class LLMObservability {

    private static final Logger LOG = Logger.getLogger(LLMObservability.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Service name for traces */
    private final String serviceName;

    /** Whether tracing is enabled */
    private final boolean enableTracing;

    /** Whether metrics are enabled */
    private final boolean enableMetrics;

    /** Sampling rate (0.0 to 1.0) */
    private final double sampleRate;

    // ── Active Traces ─────────────────────────────────────────────────

    /** Currently active traces: requestId → InferenceTrace */
    private final Map<String, InferenceTrace> activeTraces = new ConcurrentHashMap<>();

    // ── Global Metrics ────────────────────────────────────────────────

    /** Total requests processed */
    private final AtomicLong totalRequests = new AtomicLong(0);

    /** Total successful requests */
    private final AtomicLong totalSuccess = new AtomicLong(0);

    /** Total failed requests */
    private final AtomicLong totalErrors = new AtomicLong(0);

    /** Total rate limited requests */
    private final AtomicLong totalRateLimited = new AtomicLong(0);

    /** Total cancelled requests */
    private final AtomicLong totalCancelled = new AtomicLong(0);

    /** Total input tokens */
    private final AtomicLong totalInputTokens = new AtomicLong(0);

    /** Total output tokens */
    private final AtomicLong totalOutputTokens = new AtomicLong(0);

    /** Sum of TTFT (for average calculation) */
    private final DoubleAdder ttftSum = new DoubleAdder();

    /** Count of TTFT measurements */
    private final AtomicLong ttftCount = new AtomicLong(0);

    /** Sum of tokens per second (for throughput calculation) */
    private final DoubleAdder throughputSum = new DoubleAdder();

    /** Count of throughput measurements */
    private final AtomicLong throughputCount = new AtomicLong(0);

    // ── Per-Tenant Metrics ────────────────────────────────────────────

    /** Per-tenant request counts */
    private final Map<String, AtomicLong> tenantRequests = new ConcurrentHashMap<>();

    /** Per-tenant error counts */
    private final Map<String, AtomicLong> tenantErrors = new ConcurrentHashMap<>();

    /** Per-tenant rate limited counts */
    private final Map<String, AtomicLong> tenantRateLimited = new ConcurrentHashMap<>();

    /** Per-tenant token counts */
    private final Map<String, AtomicLong> tenantTokens = new ConcurrentHashMap<>();

    // ── Lifecycle ─────────────────────────────────────────────────────

    /** Service start time */
    private final Instant startTime;

    private LLMObservability(Config config) {
        this.serviceName = config.serviceName;
        this.enableTracing = config.enableTracing;
        this.enableMetrics = config.enableMetrics;
        this.sampleRate = config.sampleRate;
        this.startTime = Instant.now();

        LOG.infof("LLMObservability initialized: service=%s, tracing=%b, metrics=%b, sample=%.2f",
            serviceName, enableTracing, enableMetrics, sampleRate);
    }

    /**
     * Creates a builder for configuring this observability instance.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Tracing ───────────────────────────────────────────────────────

    /**
     * Starts a new inference trace.
     *
     * @param modelId model identifier
     * @param tenantId tenant identifier
     * @param requestId request identifier
     * @return inference trace instance
     */
    public InferenceTrace startTrace(String modelId, String tenantId, String requestId) {
        if (!enableTracing) {
            return InferenceTrace.noop();
        }

        // Check sampling
        if (sampleRate < 1.0 && Math.random() > sampleRate) {
            return InferenceTrace.noop();
        }

        InferenceTrace trace = new InferenceTrace(
            serviceName, modelId, tenantId, requestId, this);
        activeTraces.put(requestId, trace);
        totalRequests.incrementAndGet();
        tenantRequests.computeIfAbsent(tenantId, k -> new AtomicLong())
            .incrementAndGet();

        return trace;
    }

    /**
     * Ends an inference trace.
     */
    void endTrace(String requestId, InferenceTrace trace) {
        activeTraces.remove(requestId);
    }

    // ── Metrics Recording ─────────────────────────────────────────────

    /**
     * Records a successful request with token counts.
     */
    void recordSuccess(String tenantId, int inputTokens, int outputTokens,
                      double ttftMs, double tokensPerSec) {
        if (!enableMetrics) return;

        totalSuccess.incrementAndGet();
        totalInputTokens.addAndGet(inputTokens);
        totalOutputTokens.addAndGet(outputTokens);
        tenantTokens.computeIfAbsent(tenantId, k -> new AtomicLong())
            .addAndGet(inputTokens + outputTokens);

        if (ttftMs > 0) {
            ttftSum.add(ttftMs);
            ttftCount.incrementAndGet();
        }

        if (tokensPerSec > 0) {
            throughputSum.add(tokensPerSec);
            throughputCount.incrementAndGet();
        }
    }

    /**
     * Records a failed request.
     */
    void recordError(String tenantId, String errorType) {
        if (!enableMetrics) return;

        totalErrors.incrementAndGet();
        tenantErrors.computeIfAbsent(tenantId, k -> new AtomicLong())
            .incrementAndGet();
    }

    /**
     * Records a rate-limited request.
     */
    public void recordRateLimited(String tenantId) {
        if (!enableMetrics) return;

        totalRateLimited.incrementAndGet();
        tenantRateLimited.computeIfAbsent(tenantId, k -> new AtomicLong())
            .incrementAndGet();
    }

    /**
     * Records a cancelled request.
     */
    void recordCancelled() {
        if (!enableMetrics) return;
        totalCancelled.incrementAndGet();
    }

    // ── Query Methods ─────────────────────────────────────────────────

    /**
     * Gets average time-to-first-token in milliseconds.
     */
    public double getAvgTTFT() {
        long count = ttftCount.get();
        return count == 0 ? 0.0 : ttftSum.sum() / count;
    }

    /**
     * Gets average throughput in tokens per second.
     */
    public double getAvgThroughput() {
        long count = throughputCount.get();
        return count == 0 ? 0.0 : throughputSum.sum() / count;
    }

    /**
     * Gets total tokens processed (input + output).
     */
    public long getTotalTokens() {
        return totalInputTokens.get() + totalOutputTokens.get();
    }

    /**
     * Gets error rate as a percentage (0.0 to 100.0).
     */
    public double getErrorRate() {
        long total = totalRequests.get();
        return total == 0 ? 0.0 : (double) totalErrors.get() / total * 100.0;
    }

    /**
     * Gets real-time inference dashboard.
     */
    public InferenceDashboard getDashboard() {
        return new InferenceDashboard(
            serviceName,
            startTime,
            Instant.now(),
            totalRequests.get(),
            totalSuccess.get(),
            totalErrors.get(),
            totalRateLimited.get(),
            totalCancelled.get(),
            totalInputTokens.get(),
            totalOutputTokens.get(),
            getAvgTTFT(),
            getAvgThroughput(),
            activeTraces.size(),
            transformMap(tenantRequests),
            transformMap(tenantErrors),
            transformMap(tenantRateLimited)
        );
    }

    private Map<String, Long> transformMap(Map<String, AtomicLong> map) {
        Map<String, Long> result = new HashMap<>();
        map.forEach((k, v) -> result.put(k, v.get()));
        return result;
    }

    /**
     * Resets all metrics counters.
     */
    public void reset() {
        totalRequests.set(0);
        totalSuccess.set(0);
        totalErrors.set(0);
        totalRateLimited.set(0);
        totalCancelled.set(0);
        totalInputTokens.set(0);
        totalOutputTokens.set(0);
        ttftSum.reset();
        ttftCount.set(0);
        throughputSum.reset();
        throughputCount.set(0);
        tenantRequests.clear();
        tenantErrors.clear();
        tenantRateLimited.clear();
        LOG.info("LLMObservability metrics reset");
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Real-time inference dashboard.
     */
    public record InferenceDashboard(
        String serviceName,
        Instant startTime,
        Instant currentTime,
        long totalRequests,
        long totalSuccess,
        long totalErrors,
        long totalRateLimited,
        long totalCancelled,
        long totalInputTokens,
        long totalOutputTokens,
        double avgTTFT,
        double avgThroughput,
        int activeRequests,
        Map<String, Long> tenantRequests,
        Map<String, Long> tenantErrors,
        Map<String, Long> tenantRateLimited
    ) {
        /**
         * Uptime in seconds.
         */
        public long uptimeSeconds() {
            return currentTime.getEpochSecond() - startTime.getEpochSecond();
        }

        /**
         * Requests per second.
         */
        public double requestsPerSecond() {
            long uptime = uptimeSeconds();
            return uptime == 0 ? 0.0 : (double) totalRequests / uptime;
        }

        /**
         * Tokens per second.
         */
        public double tokensPerSecond() {
            long uptime = uptimeSeconds();
            return uptime == 0 ? 0.0 : (double) (totalInputTokens + totalOutputTokens) / uptime;
        }

        /**
         * Error rate percentage.
         */
        public double errorRate() {
            return totalRequests == 0 ? 0.0 : (double) totalErrors / totalRequests * 100.0;
        }

        /**
         * Rate limit rejection rate.
         */
        public double rateLimitRate() {
            return totalRequests == 0 ? 0.0 : (double) totalRateLimited / totalRequests * 100.0;
        }
    }

    /**
     * Configuration for LLMObservability.
     */
    private static final class Config {
        String serviceName = "gollek-inference";
        boolean enableTracing = true;
        boolean enableMetrics = true;
        double sampleRate = 1.0;
    }

    /**
     * Builder for LLMObservability.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        public Builder serviceName(String serviceName) {
            config.serviceName = serviceName;
            return this;
        }

        public Builder enableTracing(boolean enable) {
            config.enableTracing = enable;
            return this;
        }

        public Builder enableMetrics(boolean enable) {
            config.enableMetrics = enable;
            return this;
        }

        /**
         * Sets sampling rate (0.0 to 1.0).
         * <p>
         * 1.0 = sample all requests, 0.1 = sample 10%
         */
        public Builder sampleRate(double rate) {
            config.sampleRate = Math.max(0.0, Math.min(1.0, rate));
            return this;
        }

        public LLMObservability build() {
            return new LLMObservability(config);
        }
    }
}
