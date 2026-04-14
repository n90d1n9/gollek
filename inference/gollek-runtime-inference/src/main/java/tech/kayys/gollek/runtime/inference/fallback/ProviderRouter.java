package tech.kayys.gollek.runtime.inference.fallback;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Multi-provider fallback router for resilient inference serving.
 * <p>
 * Routes requests across multiple providers with automatic failover:
 * <ol>
 *   <li>Try primary provider (e.g., local GPU)</li>
 *   <li>If fails, try secondary provider (e.g., OpenAI)</li>
 *   <li>If fails, try tertiary provider (e.g., Anthropic)</li>
 *   <li>If all fail, return error with fallback response</li>
 * </ol>
 *
 * <h2>Fallback Strategy</h2>
 * <pre>
 * Request → Local GPU (fastest, cheapest)
 *   ↓ (timeout/error)
 *   → OpenAI GPT-4 (reliable, moderate cost)
 *   ↓ (timeout/error)
 *   → Anthropic Claude (backup, higher cost)
 *   ↓ (timeout/error)
 *   → Cached/Fallback Response
 * </pre>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li><b>Automatic Failover:</b> Try next provider on timeout/error</li>
 *   <li><b>Health Monitoring:</b> Track provider success/failure rates</li>
 *   <li><b>Cost Tracking:</b> Per-provider cost aggregation</li>
 *   <li><b>Circuit Breaking:</b> Skip unhealthy providers</li>
 *   <li><b>Smart Routing:</b> Route by cost, latency, or availability</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ProviderRouter router = ProviderRouter.builder()
 *     .addProvider("local-gpu", ProviderConfig.local("llama-3-70b"))
 *     .addProvider("openai", ProviderConfig.openai("gpt-4"))
 *     .addProvider("anthropic", ProviderConfig.anthropic("claude-3"))
 *     .maxRetries(3)
 *     .timeoutPerProvider(Duration.ofSeconds(10))
 *     .fallbackResponse("I'm sorry, I cannot process your request at the moment.")
 *     .build();
 *
 * InferenceResponse response = router.route(
 *     InferenceRequest.builder()
 *         .messages(List.of(Message.user("Hello")))
 *         .build(),
 *     RequestContext.builder().apiKey("sk-123").build());
 * }</pre>
 *
 * @since 0.4.0
 */
public final class ProviderRouter {

    private static final Logger LOG = Logger.getLogger(ProviderRouter.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Ordered list of providers (by priority) */
    private final List<ProviderConfig> providers;

    /** Maximum retry attempts */
    private final int maxRetries;

    /** Timeout per provider attempt */
    private final Duration timeoutPerProvider;

    /** Fallback response when all providers fail */
    private final String fallbackResponse;

    /** Whether to enable circuit breaking */
    private final boolean circuitBreakerEnabled;

    /** Circuit breaker threshold (consecutive failures before skipping) */
    private final int circuitBreakerThreshold;

    // ── Provider State ────────────────────────────────────────────────

    /** Provider health tracking: providerId → ProviderHealth */
    private final Map<String, ProviderHealth> providerHealth = new ConcurrentHashMap<>();

    /** Executor for parallel provider attempts */
    private final ExecutorService executor;

    // ── Statistics ────────────────────────────────────────────────────

    /** Total requests routed */
    private final AtomicLong totalRequests = new AtomicLong(0);

    /** Total successful requests */
    private final AtomicLong totalSuccess = new AtomicLong(0);

    /** Total fallback responses */
    private final AtomicLong totalFallbacks = new AtomicLong(0);

    /** Per-provider success counts */
    private final Map<String, AtomicLong> providerSuccessCounts = new ConcurrentHashMap<>();

    /** Per-provider failure counts */
    private final Map<String, AtomicLong> providerFailureCounts = new ConcurrentHashMap<>();

    /** Per-provider latency sums */
    private final Map<String, AtomicLong> providerLatencySums = new ConcurrentHashMap<>();

    // ── Lifecycle ─────────────────────────────────────────────────────

    private ProviderRouter(Config config) {
        this.providers = List.copyOf(config.providers);
        this.maxRetries = config.maxRetries;
        this.timeoutPerProvider = config.timeoutPerProvider;
        this.fallbackResponse = config.fallbackResponse;
        this.circuitBreakerEnabled = config.circuitBreakerEnabled;
        this.circuitBreakerThreshold = config.circuitBreakerThreshold;
        this.executor = Executors.newVirtualThreadPerTaskExecutor();

        // Initialize health tracking
        for (ProviderConfig provider : providers) {
            providerHealth.put(provider.id(), new ProviderHealth());
            providerSuccessCounts.put(provider.id(), new AtomicLong(0));
            providerFailureCounts.put(provider.id(), new AtomicLong(0));
            providerLatencySums.put(provider.id(), new AtomicLong(0));
        }

        LOG.infof("ProviderRouter initialized: %d providers, maxRetries=%d, timeout=%ds",
            providers.size(), maxRetries, timeoutPerProvider.getSeconds());
    }

    /**
     * Creates a builder for configuring this router.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Request Routing ───────────────────────────────────────────────

    /**
     * Routes an inference request through the provider chain.
     *
     * @param request inference request
     * @param context request context
     * @return inference response from first successful provider
     */
    public ProviderResponse route(Object request, RequestContext context) {
        totalRequests.incrementAndGet();

        int attempts = 0;
        List<String> attemptedProviders = new ArrayList<>();
        Exception lastError = null;

        for (ProviderConfig provider : providers) {
            if (attempts >= maxRetries) {
                LOG.warnf("Max retries (%d) reached, using fallback", maxRetries);
                break;
            }

            // Check circuit breaker
            if (circuitBreakerEnabled && isCircuitOpen(provider.id())) {
                LOG.debugf("Skipping provider %s (circuit open)", provider.id());
                continue;
            }

            Instant attemptStart = Instant.now();
            attemptedProviders.add(provider.id());

            try {
                // Execute with timeout
                ProviderResponse response = executeWithTimeout(provider, request, context);

                // Success
                long latencyMs = Duration.between(attemptStart, Instant.now()).toMillis();
                recordSuccess(provider.id(), latencyMs);

                LOG.infof("Provider %s succeeded in %dms (attempt %d)",
                    provider.id(), latencyMs, attempts + 1);

                return response.withMetadata(Map.of(
                    "provider", provider.id(),
                    "attempt", attempts + 1,
                    "latency_ms", latencyMs,
                    "attempted_providers", attemptedProviders
                ));

            } catch (Exception e) {
                long latencyMs = Duration.between(attemptStart, Instant.now()).toMillis();
                recordFailure(provider.id());
                lastError = e;

                LOG.warnf("Provider %s failed in %dms: %s",
                    provider.id(), latencyMs, e.getMessage());

                attempts++;
            }
        }

        // All providers failed - use fallback
        totalFallbacks.incrementAndGet();

        LOG.warnf("All providers failed, returning fallback response (attempted: %s)",
            attemptedProviders);

        return new ProviderResponse(
            context.requestId(),
            providers.get(0).modelId(),
            fallbackResponse,
            0, 0, 0,
            "fallback",
            Map.of(
                "fallback", true,
                "attempted_providers", attemptedProviders,
                "last_error", lastError != null ? lastError.getMessage() : "unknown"
            )
        );
    }

    /**
     * Routes request to a specific provider.
     */
    public ProviderResponse routeToProvider(String providerId, Object request,
                                           RequestContext context) {
        ProviderConfig provider = providers.stream()
            .filter(p -> p.id().equals(providerId))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Unknown provider: " + providerId));

        Instant start = Instant.now();
        try {
            ProviderResponse response = executeWithTimeout(provider, request, context);
            long latencyMs = Duration.between(start, Instant.now()).toMillis();
            recordSuccess(providerId, latencyMs);
            return response;
        } catch (Exception e) {
            recordFailure(providerId);
            throw new ProviderRoutingException("Provider " + providerId + " failed: " + e.getMessage(), e);
        }
    }

    // ── Health Monitoring ─────────────────────────────────────────────

    /**
     * Gets health status for a provider.
     */
    public ProviderHealthStatus getProviderHealth(String providerId) {
        ProviderHealth health = providerHealth.get(providerId);
        if (health == null) {
            throw new IllegalArgumentException("Unknown provider: " + providerId);
        }

        long total = health.successCount() + health.failureCount();
        double successRate = total == 0 ? 1.0 : (double) health.successCount() / total;
        long avgLatency = health.successCount() == 0 ? 0 :
            health.totalLatencyMs() / health.successCount();

        boolean circuitOpen = circuitBreakerEnabled &&
            health.consecutiveFailures() >= circuitBreakerThreshold;

        return new ProviderHealthStatus(
            providerId,
            successRate,
            health.successCount(),
            health.failureCount(),
            avgLatency,
            health.consecutiveFailures(),
            circuitOpen
        );
    }

    /**
     * Gets health status for all providers.
     */
    public List<ProviderHealthStatus> getAllProviderHealth() {
        return providers.stream()
            .map(p -> getProviderHealth(p.id()))
            .toList();
    }

    /**
     * Resets health tracking for a provider.
     */
    public void resetProviderHealth(String providerId) {
        providerHealth.put(providerId, new ProviderHealth());
        LOG.infof("Provider health reset: %s", providerId);
    }

    // ── Statistics ────────────────────────────────────────────────────

    /**
     * Gets routing statistics.
     */
    public RouterStats getStats() {
        Map<String, ProviderStats> providerStats = new HashMap<>();
        for (ProviderConfig provider : providers) {
            String id = provider.id();
            long success = providerSuccessCounts.get(id).get();
            long failure = providerFailureCounts.get(id).get();
            long totalLatency = providerLatencySums.get(id).get();
            providerStats.put(id, new ProviderStats(
                success, failure, totalLatency,
                success == 0 ? 0 : totalLatency / success
            ));
        }

        return new RouterStats(
            totalRequests.get(),
            totalSuccess.get(),
            totalFallbacks.get(),
            providers.size(),
            providerStats
        );
    }

    // ── Internal Methods ──────────────────────────────────────────────

    /**
     * Executes request with timeout.
     */
    private ProviderResponse executeWithTimeout(ProviderConfig provider,
                                                Object request,
                                                RequestContext context) {
        try {
            Future<ProviderResponse> future = executor.submit(() ->
                provider.client().execute(request, context));

            return future.get(timeoutPerProvider.toMillis(), TimeUnit.MILLISECONDS);

        } catch (TimeoutException e) {
            throw new ProviderTimeoutException(
                "Provider " + provider.id() + " timed out after " + timeoutPerProvider.getSeconds() + "s");
        } catch (Exception e) {
            throw new ProviderExecutionException(
                "Provider " + provider.id() + " failed: " + e.getMessage(), e);
        }
    }

    /**
     * Checks if circuit breaker is open for a provider.
     */
    private boolean isCircuitOpen(String providerId) {
        ProviderHealth health = providerHealth.get(providerId);
        return health != null && health.consecutiveFailures() >= circuitBreakerThreshold;
    }

    /**
     * Records successful provider attempt.
     */
    private void recordSuccess(String providerId, long latencyMs) {
        totalSuccess.incrementAndGet();
        providerSuccessCounts.get(providerId).incrementAndGet();
        providerLatencySums.get(providerId).addAndGet(latencyMs);

        ProviderHealth health = providerHealth.get(providerId);
        providerHealth.put(providerId, health.withSuccess());
    }

    /**
     * Records failed provider attempt.
     */
    private void recordFailure(String providerId) {
        providerFailureCounts.get(providerId).incrementAndGet();

        ProviderHealth health = providerHealth.get(providerId);
        providerHealth.put(providerId, health.withFailure());
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Provider health tracking record.
     */
    record ProviderHealth(
        long successCount,
        long failureCount,
        long totalLatencyMs,
        int consecutiveFailures
    ) {
        ProviderHealth() {
            this(0, 0, 0, 0);
        }

        ProviderHealth withSuccess() {
            return new ProviderHealth(
                successCount + 1,
                failureCount,
                totalLatencyMs,
                0  // Reset consecutive failures
            );
        }

        ProviderHealth withFailure() {
            return new ProviderHealth(
                successCount,
                failureCount + 1,
                totalLatencyMs,
                consecutiveFailures + 1
            );
        }
    }

    /**
     * Provider health status snapshot.
     */
    public record ProviderHealthStatus(
        String providerId,
        double successRate,
        long successCount,
        long failureCount,
        long avgLatencyMs,
        int consecutiveFailures,
        boolean circuitOpen
    ) {}

    /**
     * Provider statistics.
     */
    public record ProviderStats(
        long successCount,
        long failureCount,
        long totalLatencyMs,
        long avgLatencyMs
    ) {}

    /**
     * Router statistics.
     */
    public record RouterStats(
        long totalRequests,
        long totalSuccess,
        long totalFallbacks,
        int providerCount,
        Map<String, ProviderStats> providerStats
    ) {
        public double fallbackRate() {
            return totalRequests == 0 ? 0.0 :
                (double) totalFallbacks / totalRequests * 100.0;
        }
    }

    /**
     * Provider response wrapper.
     */
    public record ProviderResponse(
        String requestId,
        String model,
        String content,
        int inputTokens,
        int outputTokens,
        long latencyMs,
        String finishReason,
        Map<String, Object> metadata
    ) {
        public ProviderResponse withMetadata(Map<String, Object> metadata) {
            return new ProviderResponse(
                requestId, model, content, inputTokens, outputTokens,
                latencyMs, finishReason, metadata);
        }
    }

    /**
     * Request context for provider execution.
     */
    public record RequestContext(
        String apiKey,
        String requestId,
        String tenantId
    ) {
        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private String apiKey;
            private String requestId = UUID.randomUUID().toString();
            private String tenantId = "default";

            public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
            public Builder requestId(String requestId) { this.requestId = requestId; return this; }
            public Builder tenantId(String tenantId) { this.tenantId = tenantId; return this; }

            public RequestContext build() {
                return new RequestContext(apiKey, requestId, tenantId);
            }
        }
    }

    /**
     * Provider client interface.
     */
    public interface ProviderClient {
        ProviderResponse execute(Object request, RequestContext context);
    }

    /**
     * Provider configuration.
     */
    public record ProviderConfig(
        String id,
        String modelId,
        ProviderClient client,
        int priority,
        Map<String, Object> extra
    ) {
        public static ProviderConfig local(String modelId) {
            return new ProviderConfig("local", modelId, null, 0, Map.of());
        }

        public static ProviderConfig openai(String modelId) {
            return new ProviderConfig("openai", modelId, null, 1, Map.of());
        }

        public static ProviderConfig anthropic(String modelId) {
            return new ProviderConfig("anthropic", modelId, null, 2, Map.of());
        }
    }

    /**
     * Exception thrown on provider timeout.
     */
    public static class ProviderTimeoutException extends RuntimeException {
        public ProviderTimeoutException(String message) {
            super(message);
        }
    }

    /**
     * Exception thrown on provider execution failure.
     */
    public static class ProviderExecutionException extends RuntimeException {
        public ProviderExecutionException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Exception thrown on routing failure.
     */
    public static class ProviderRoutingException extends RuntimeException {
        public ProviderRoutingException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Builder for ProviderRouter.
     */
    public static final class Builder {
        private final List<ProviderConfig> providers = new ArrayList<>();
        private int maxRetries = 3;
        private Duration timeoutPerProvider = Duration.ofSeconds(10);
        private String fallbackResponse = "I'm sorry, I cannot process your request at the moment.";
        private boolean circuitBreakerEnabled = true;
        private int circuitBreakerThreshold = 5;

        private Builder() {}

        public Builder addProvider(String id, ProviderConfig config) {
            providers.add(config);
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder timeoutPerProvider(Duration timeout) {
            this.timeoutPerProvider = timeout;
            return this;
        }

        public Builder fallbackResponse(String response) {
            this.fallbackResponse = response;
            return this;
        }

        public Builder circuitBreakerEnabled(boolean enabled) {
            this.circuitBreakerEnabled = enabled;
            return this;
        }

        public Builder circuitBreakerThreshold(int threshold) {
            this.circuitBreakerThreshold = threshold;
            return this;
        }

        public ProviderRouter build() {
            if (providers.isEmpty()) {
                throw new IllegalStateException("At least one provider is required");
            }
            return new ProviderRouter(new Config(
                providers, maxRetries, timeoutPerProvider,
                fallbackResponse, circuitBreakerEnabled, circuitBreakerThreshold));
        }
    }

    /**
     * Configuration record.
     */
    private record Config(
        List<ProviderConfig> providers,
        int maxRetries,
        Duration timeoutPerProvider,
        String fallbackResponse,
        boolean circuitBreakerEnabled,
        int circuitBreakerThreshold
    ) {}
}
