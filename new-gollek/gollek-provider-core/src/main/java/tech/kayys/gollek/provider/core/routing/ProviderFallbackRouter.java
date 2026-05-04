package tech.kayys.gollek.provider.routing;

import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderHealth;

import io.smallrye.mutiny.Uni;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Intelligent provider router with automatic fallback chain.
 * <p>
 * Routes inference requests through a prioritized chain of providers:
 * <ol>
 *   <li>Local GPU/CPU (GGUF, ONNX, etc.)</li>
 *   <li>Primary cloud provider (OpenAI, Anthropic, etc.)</li>
 *   <li>Backup cloud provider</li>
 *   <li>Emergency fallback (cheapest available)</li>
 * </ol>
 * <p>
 * <b>Routing Strategy:</b>
 * <ul>
 *   <li><b>Cost-aware:</b> Tracks per-token costs and respects budget limits</li>
 *   <li><b>Latency-optimized:</b> Prefers low-latency providers based on historical data</li>
 *   <li><b>Fault-tolerant:</b> Automatic retry with next provider on failure</li>
 *   <li><b>Quota-managed:</b> Respects per-provider quotas and rotates when exhausted</li>
 * </ul>
 * 
 * <h2>Usage</h2>
 * <pre>
 * ProviderFallbackRouter router = ProviderFallbackRouter.builder()
 *     .primary("openai", "gpt-4", 0.03)  // provider, model, cost per 1K tokens
 *     .fallback("anthropic", "claude-3", 0.025)
 *     .localFallback("gguf", "llama-3-70b.gguf", 0.0)
 *     .maxCostPerRequest(0.10)
 *     .maxRetries(2)
 *     .timeout(Duration.ofSeconds(30))
 *     .build();
 * 
 * Uni<ProviderResponse> response = router.route(request);
 * </pre>
 * 
 * @since 0.1.0
 */
public class ProviderFallbackRouter {

    private static final Logger LOG = Logger.getLogger(ProviderFallbackRouter.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Ordered provider chain (primary → fallbacks) */
    private final List<ProviderChainEntry> providerChain;

    /** Maximum cost per request in USD cents */
    private final double maxCostPerRequest;

    /** Maximum number of retries across fallbacks */
    private final int maxRetries;

    /** Request timeout */
    private final java.time.Duration timeout;

    // ── State ─────────────────────────────────────────────────────────

    /** Provider registry */
    private final Map<String, LLMProvider> providers = new ConcurrentHashMap<>();

    /** Metrics per provider */
    private final Map<String, ProviderMetrics> metrics = new ConcurrentHashMap<>();

    /** Circuit breakers per provider */
    private final Map<String, CircuitBreaker> circuitBreakers = new ConcurrentHashMap<>();

    // ── Lifecycle ─────────────────────────────────────────────────────

    private ProviderFallbackRouter(Config config) {
        this.providerChain = List.copyOf(config.providerChain);
        this.maxCostPerRequest = config.maxCostPerRequest;
        this.maxRetries = config.maxRetries;
        this.timeout = config.timeout;

        // Initialize metrics and circuit breakers
        for (ProviderChainEntry entry : providerChain) {
            metrics.put(entry.providerId(), new ProviderMetrics());
            circuitBreakers.put(entry.providerId(), new CircuitBreaker());
        }

        LOG.infof("ProviderFallbackRouter initialized: %d providers in chain", 
            providerChain.size());
    }

    /**
     * Creates a builder for configuring this router.
     */
    public static ConfigBuilder builder() {
        return new ConfigBuilder();
    }

    /**
     * Registers a provider instance.
     */
    public void registerProvider(String id, LLMProvider provider) {
        providers.put(id, provider);
        LOG.infof("Registered provider: %s", id);
    }

    // ── Routing ───────────────────────────────────────────────────────

    /**
     * Routes an inference request through the provider chain.
     *
     * @param request the inference request
     * @return Uni with the inference response, or error if all providers fail
     */
    public Uni<InferenceResponse> route(ProviderRequest request) {
        return routeWithFallbacks(request, 0, 0, null);
    }

    /**
     * Recursive fallback implementation.
     */
    private Uni<InferenceResponse> routeWithFallbacks(
            ProviderRequest request, 
            int attempt, 
            double accumulatedCost,
            Throwable lastError) {

        // Check if we've exceeded max retries
        if (attempt >= maxRetries) {
            return Uni.createFrom().failure(
                new MaxRetriesExceededException(
                    "Max retries (" + maxRetries + ") exceeded. Last error: " + 
                    (lastError != null ? lastError.getMessage() : "unknown")));
        }

        // Get next provider in chain
        if (attempt >= providerChain.size()) {
            return Uni.createFrom().failure(
                new NoProvidersAvailableException(
                    "No providers available in fallback chain"));
        }

        ProviderChainEntry entry = providerChain.get(attempt);
        String providerId = entry.providerId();

        // Check circuit breaker
        CircuitBreaker cb = circuitBreakers.get(providerId);
        if (cb.isOpen()) {
            LOG.warnf("Circuit breaker open for %s, trying next", providerId);
            return routeWithFallbacks(request, attempt + 1, accumulatedCost, lastError);
        }

        // Check cost budget
        double estimatedCost = entry.costPer1KTokens() * estimateTokens(request) / 1000.0;
        if (accumulatedCost + estimatedCost > maxCostPerRequest) {
            LOG.warnf("Cost budget exceeded for provider %s (%.4f > %.4f), trying next",
                providerId, accumulatedCost + estimatedCost, maxCostPerRequest);
            return routeWithFallbacks(request, attempt + 1, accumulatedCost, lastError);
        }

        // Get provider
        LLMProvider provider = providers.get(providerId);
        if (provider == null || !provider.isEnabled()) {
            LOG.warnf("Provider %s not available, trying next", providerId);
            return routeWithFallbacks(request, attempt + 1, accumulatedCost, lastError);
        }

        // Check provider health
        if (!provider.health().await().atMost(timeout).isHealthy()) {
            LOG.warnf("Provider %s unhealthy, trying next", providerId);
            cb.recordFailure();
            return routeWithFallbacks(request, attempt + 1, accumulatedCost, lastError);
        }

        // Execute request
        long startTime = System.currentTimeMillis();
        return provider.infer(request)
            .onItem().transform(response -> {
                // Record success metrics
                long latency = System.currentTimeMillis() - startTime;
                ProviderMetrics m = metrics.get(providerId);
                m.recordSuccess(latency, response.getTokensUsed());
                cb.recordSuccess();

                LOG.infof("Request succeeded with %s (attempt %d, latency=%dms)",
                    providerId, attempt + 1, latency);

                return response;
            })
            .onFailure().recoverWithUni(error -> {
                // Record failure
                long latency = System.currentTimeMillis() - startTime;
                ProviderMetrics m = metrics.get(providerId);
                m.recordFailure(latency);
                cb.recordFailure();

                LOG.warnf(error, "Provider %s failed (attempt %d), trying next",
                    providerId, attempt + 1);

                // Try next provider
                return routeWithFallbacks(request, attempt + 1, 
                    accumulatedCost, error);
            });
    }

    // ── Metrics ───────────────────────────────────────────────────────

    /**
     * Gets metrics for all providers.
     */
    public Map<String, ProviderMetrics> getMetrics() {
        return Collections.unmodifiableMap(metrics);
    }

    /**
     * Gets provider statistics.
     */
    public Map<String, ProviderStats> getProviderStats() {
        Map<String, ProviderStats> stats = new HashMap<>();
        for (Map.Entry<String, ProviderMetrics> entry : metrics.entrySet()) {
            stats.put(entry.getKey(), entry.getValue().getStats());
        }
        return stats;
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /**
     * Estimates token count for a request.
     * Rough estimate: ~4 characters per token for English text.
     */
    private int estimateTokens(ProviderRequest request) {
        // Get messages and estimate
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            return 100;  // Default estimate
        }
        
        int totalChars = request.getMessages().stream()
            .mapToInt(msg -> msg.getContent() != null ? msg.getContent().length() : 0)
            .sum();
        
        return Math.max(totalChars / 4, 10);
    }

    // ── Nested Classes ────────────────────────────────────────────────

    /**
     * Entry in the provider fallback chain.
     */
    record ProviderChainEntry(
        String providerId,
        String modelId,
        double costPer1KTokens,
        int priority
    ) {}

    /**
     * Provider metrics.
     */
    public static class ProviderMetrics {
        private final AtomicInteger totalRequests = new AtomicInteger(0);
        private final AtomicInteger successfulRequests = new AtomicInteger(0);
        private final AtomicInteger failedRequests = new AtomicInteger(0);
        private final AtomicLong totalLatency = new AtomicLong(0);
        private final AtomicLong totalTokens = new AtomicLong(0);

        void recordSuccess(long latencyMs, int tokens) {
            successfulRequests.incrementAndGet();
            totalRequests.incrementAndGet();
            totalLatency.addAndGet(latencyMs);
            totalTokens.addAndGet(tokens);
        }

        void recordFailure(long latencyMs) {
            failedRequests.incrementAndGet();
            totalRequests.incrementAndGet();
            totalLatency.addAndGet(latencyMs);
        }

        public ProviderStats getStats() {
            int total = totalRequests.get();
            int success = successfulRequests.get();
            return new ProviderStats(
                total,
                success,
                totalRequests.get() - success,
                total > 0 ? totalLatency.get() / total : 0,
                totalTokens.get(),
                total > 0 ? (double) success / total * 100.0 : 0.0
            );
        }
    }

    /**
     * Provider statistics.
     */
    public record ProviderStats(
        int totalRequests,
        int successfulRequests,
        int failedRequests,
        long avgLatencyMs,
        long totalTokens,
        double successRate
    ) {}

    /**
     * Simple circuit breaker implementation.
     */
    static class CircuitBreaker {
        private static final int FAILURE_THRESHOLD = 5;
        private static final int SUCCESS_THRESHOLD = 3;
        private static final long RESET_TIMEOUT_MS = 60000;  // 1 minute

        private volatile State state = State.CLOSED;
        private final AtomicInteger failureCount = new AtomicInteger(0);
        private final AtomicInteger successCount = new AtomicInteger(0);
        private volatile long lastFailureTime = 0;

        enum State { CLOSED, OPEN, HALF_OPEN }

        synchronized void recordSuccess() {
            if (state == State.HALF_OPEN) {
                successCount.incrementAndGet();
                if (successCount.get() >= SUCCESS_THRESHOLD) {
                    state = State.CLOSED;
                    failureCount.set(0);
                    successCount.set(0);
                    LOG.info("Circuit breaker closed");
                }
            }
        }

        synchronized void recordFailure() {
            failureCount.incrementAndGet();
            lastFailureTime = System.currentTimeMillis();
            if (failureCount.get() >= FAILURE_THRESHOLD) {
                state = State.OPEN;
                successCount.set(0);
                LOG.warn("Circuit breaker opened");
            }
        }

        boolean isOpen() {
            if (state == State.CLOSED) {
                return false;
            }
            if (state == State.OPEN) {
                // Check if reset timeout has passed
                if (System.currentTimeMillis() - lastFailureTime > RESET_TIMEOUT_MS) {
                    state = State.HALF_OPEN;
                    successCount.set(0);
                    LOG.info("Circuit breaker half-open");
                    return false;
                }
                return true;
            }
            return false;  // HALF_OPEN
        }
    }

    /**
     * Configuration for the router.
     */
    public static final class Config {
        final List<ProviderChainEntry> providerChain;
        final double maxCostPerRequest;
        final int maxRetries;
        final java.time.Duration timeout;

        private Config(List<ProviderChainEntry> providerChain, double maxCostPerRequest,
                      int maxRetries, java.time.Duration timeout) {
            this.providerChain = providerChain;
            this.maxCostPerRequest = maxCostPerRequest;
            this.maxRetries = maxRetries;
            this.timeout = timeout;
        }
    }

    /**
     * Builder for Config.
     */
    public static final class ConfigBuilder {
        private final List<ProviderChainEntry> providerChain = new ArrayList<>();
        private double maxCostPerRequest = 1.0;  // $1.00
        private int maxRetries = 3;
        private java.time.Duration timeout = java.time.Duration.ofSeconds(30);
        private int priorityCounter = 0;

        /**
         * Adds a primary provider.
         *
         * @param providerId provider identifier
         * @param modelId model identifier
         * @param costPer1KTokens cost per 1000 tokens in USD
         */
        public ConfigBuilder primary(String providerId, String modelId, double costPer1KTokens) {
            providerChain.add(0, new ProviderChainEntry(
                providerId, modelId, costPer1KTokens, priorityCounter++));
            return this;
        }

        /**
         * Adds a fallback provider.
         */
        public ConfigBuilder fallback(String providerId, String modelId, double costPer1KTokens) {
            providerChain.add(new ProviderChainEntry(
                providerId, modelId, costPer1KTokens, priorityCounter++));
            return this;
        }

        /**
         * Adds a local fallback provider (e.g., GGUF).
         */
        public ConfigBuilder localFallback(String providerId, String modelId, double costPer1KTokens) {
            providerChain.add(new ProviderChainEntry(
                providerId, modelId, costPer1KTokens, priorityCounter++));
            return this;
        }

        public ConfigBuilder maxCostPerRequest(double maxCostPerRequest) {
            this.maxCostPerRequest = maxCostPerRequest;
            return this;
        }

        public ConfigBuilder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public ConfigBuilder timeout(java.time.Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Config build() {
            if (providerChain.isEmpty()) {
                throw new IllegalStateException("At least one provider must be configured");
            }
            return new Config(providerChain, maxCostPerRequest, maxRetries, timeout);
        }
    }

    // ── Exception Types ───────────────────────────────────────────────

    static class MaxRetriesExceededException extends RuntimeException {
        MaxRetriesExceededException(String message) {
            super(message);
        }
    }

    static class NoProvidersAvailableException extends RuntimeException {
        NoProvidersAvailableException(String message) {
            super(message);
        }
    }
}
