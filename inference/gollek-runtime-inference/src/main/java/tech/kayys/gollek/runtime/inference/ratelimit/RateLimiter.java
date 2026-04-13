package tech.kayys.gollek.runtime.inference.ratelimit;

import org.jboss.logging.Logger;

import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Multi-tenant rate limiter using token bucket algorithm.
 * <p>
 * Provides rate limiting at multiple levels:
 * <ul>
 *   <li><b>Requests per minute:</b> Limits API call frequency</li>
 *   <li><b>Tokens per minute:</b> Limits compute consumption</li>
 *   <li><b>Concurrent requests:</b> Limits simultaneous load</li>
 *   <li><b>Context length:</b> Limits per-request resource usage</li>
 * </ul>
 *
 * <h2>Token Bucket Algorithm</h2>
 * <pre>
 * Each tenant has a bucket with capacity N tokens.
 * - Tokens are consumed on each request (1 token per request)
 * - Tokens refill at rate R tokens/minute
 * - If bucket is empty, request is rejected (429 Too Many Requests)
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * RateLimiter rateLimiter = RateLimiter.builder()
 *     .defaultTier(RateLimitTier.PRO)
 *     .refillIntervalSeconds(1)
 *     .build();
 *
 * // Set tenant tier
 * rateLimiter.setTenantTier("tenant-123", RateLimitTier.ENTERPRISE);
 *
 * // Check rate limit before processing request
 * RateLimitResult result = rateLimiter.tryAcquire("tenant-123", 500);
 * if (!result.isAllowed()) {
 *     return Response.status(429).entity(result.toErrorPayload()).build();
 * }
 *
 * // Release after request completes
 * rateLimiter.release("tenant-123");
 * }</pre>
 *
 * @since 0.2.0
 */
public final class RateLimiter {

    private static final Logger LOG = Logger.getLogger(RateLimiter.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Default tier for unregistered tenants */
    private final RateLimitTier defaultTier;

    /** Refill interval in seconds */
    private final int refillIntervalSeconds;

    /** Clock for time tracking (mockable for testing) */
    private final Clock clock;

    // ── Per-Tenant State ──────────────────────────────────────────────

    /** Tenant → tier mapping */
    private final Map<String, RateLimitTier> tenantTiers = new ConcurrentHashMap<>();

    /** Tenant → token bucket for requests */
    private final Map<String, TokenBucket> requestBuckets = new ConcurrentHashMap<>();

    /** Tenant → token bucket for tokens */
    private final Map<String, TokenBucket> tokenBuckets = new ConcurrentHashMap<>();

    /** Tenant → concurrent request counter */
    private final Map<String, AtomicInteger> concurrentRequests = new ConcurrentHashMap<>();

    // ── Global Statistics ─────────────────────────────────────────────

    /** Total allowed requests */
    private final AtomicLong totalAllowed = new AtomicLong(0);

    /** Total rejected requests */
    private final AtomicLong totalRejected = new AtomicLong(0);

    /** Total tenants tracked */
    private final AtomicInteger activeTenants = new AtomicInteger(0);

    /**
     * Creates a new rate limiter.
     */
    private RateLimiter(Config config) {
        this.defaultTier = config.defaultTier;
        this.refillIntervalSeconds = config.refillIntervalSeconds;
        this.clock = config.clock;
    }

    /**
     * Creates a builder for configuring this rate limiter.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Tenant Management ─────────────────────────────────────────────

    /**
     * Sets the rate limit tier for a tenant.
     *
     * @param tenantId tenant identifier
     * @param tier rate limit tier
     */
    public void setTenantTier(String tenantId, RateLimitTier tier) {
        tenantTiers.put(tenantId, tier);
        LOG.infof("Set tenant %s tier to %s", tenantId, tier);
    }

    /**
     * Gets the rate limit tier for a tenant.
     */
    public RateLimitTier getTenantTier(String tenantId) {
        return tenantTiers.getOrDefault(tenantId, defaultTier);
    }

    /**
     * Removes a tenant from rate limiting.
     */
    public void removeTenant(String tenantId) {
        tenantTiers.remove(tenantId);
        requestBuckets.remove(tenantId);
        tokenBuckets.remove(tenantId);
        concurrentRequests.remove(tenantId);
        activeTenants.decrementAndGet();
    }

    // ── Rate Limiting ─────────────────────────────────────────────────

    /**
     * Attempts to acquire permission for a request.
     *
     * @param tenantId tenant identifier
     * @param estimatedTokens estimated tokens for this request (input + output)
     * @return rate limit result (allowed or rejected with reason)
     */
    public RateLimitResult tryAcquire(String tenantId, int estimatedTokens) {
        RateLimitTier tier = getTenantTier(tenantId);

        // Unlimited tier - always allow
        if (tier.isUnlimited()) {
            totalAllowed.incrementAndGet();
            return RateLimitResult.allowed(tier);
        }

        // Initialize buckets if needed
        TokenBucket requestBucket = requestBuckets.computeIfAbsent(tenantId,
            id -> new TokenBucket(tier.getMaxRequestsPerMinute(), tier.getMaxRequestsPerMinute(), refillIntervalSeconds, clock));
        TokenBucket tokenBucket = tokenBuckets.computeIfAbsent(tenantId,
            id -> new TokenBucket(tier.getMaxTokensPerMinute(), tier.getMaxTokensPerMinute(), refillIntervalSeconds, clock));

        AtomicInteger concurrent = concurrentRequests.computeIfAbsent(tenantId,
            id -> new AtomicInteger(0));

        // Check concurrent request limit
        if (concurrent.get() >= tier.getMaxConcurrentRequests()) {
            totalRejected.incrementAndGet();
            return RateLimitResult.rejected(tier, "concurrent_limit",
                "Maximum concurrent requests exceeded (" + tier.getMaxConcurrentRequests() + ")");
        }

        // Check context length limit
        if (estimatedTokens > tier.getMaxContextLength()) {
            totalRejected.incrementAndGet();
            return RateLimitResult.rejected(tier, "context_limit",
                "Context length exceeds maximum (" + tier.getMaxContextLength() + ")");
        }

        // Check request rate limit
        if (!requestBucket.tryConsume(1)) {
            totalRejected.incrementAndGet();
            long retryAfter = requestBucket.getTimeUntilRefill();
            return RateLimitResult.rejected(tier, "request_limit",
                "Request rate limit exceeded (" + tier.getMaxRequestsPerMinute() + "/min)",
                retryAfter);
        }

        // Check token rate limit
        if (!tokenBucket.tryConsume(estimatedTokens)) {
            // Refund request token since we're rejecting
            requestBucket.refund(1);
            totalRejected.incrementAndGet();
            long retryAfter = tokenBucket.getTimeUntilRefill();
            return RateLimitResult.rejected(tier, "token_limit",
                "Token rate limit exceeded (" + tier.getMaxTokensPerMinute() + "/min)",
                retryAfter);
        }

        // All checks passed - increment concurrent counter
        concurrent.incrementAndGet();
        totalAllowed.incrementAndGet();

        return RateLimitResult.allowed(tier);
    }

    /**
     * Releases a concurrent request slot.
     *
     * @param tenantId tenant identifier
     */
    public void release(String tenantId) {
        AtomicInteger concurrent = concurrentRequests.get(tenantId);
        if (concurrent != null) {
            concurrent.decrementAndGet();
        }
    }

    /**
     * Records actual token usage after request completion.
     *
     * @param tenantId tenant identifier
     * @param actualTokens actual tokens consumed
     */
    public void recordTokenUsage(String tenantId, int actualTokens) {
        // Token bucket was already charged with estimate
        // If actual differs significantly, we could adjust here
        // For now, the estimate is good enough
    }

    // ── Query Methods ─────────────────────────────────────────────────

    /**
     * Gets remaining request quota for a tenant.
     */
    public int getRemainingRequests(String tenantId) {
        TokenBucket bucket = requestBuckets.get(tenantId);
        if (bucket == null) return getTenantTier(tenantId).getMaxRequestsPerMinute();
        return (int) bucket.getTokens();
    }

    /**
     * Gets remaining token quota for a tenant.
     */
    public int getRemainingTokens(String tenantId) {
        TokenBucket bucket = tokenBuckets.get(tenantId);
        if (bucket == null) return getTenantTier(tenantId).getMaxTokensPerMinute();
        return (int) bucket.getTokens();
    }

    /**
     * Gets current concurrent request count for a tenant.
     */
    public int getConcurrentRequests(String tenantId) {
        AtomicInteger concurrent = concurrentRequests.get(tenantId);
        return concurrent != null ? concurrent.get() : 0;
    }

    /**
     * Gets rate limiting statistics.
     */
    public RateLimitStats getStats() {
        return new RateLimitStats(
            totalAllowed.get(),
            totalRejected.get(),
            activeTenants.get(),
            tenantTiers.size()
        );
    }

    /**
     * Resets all rate limit counters.
     */
    public void reset() {
        requestBuckets.clear();
        tokenBuckets.clear();
        concurrentRequests.clear();
        totalAllowed.set(0);
        totalRejected.set(0);
        activeTenants.set(0);
        LOG.info("Rate limiter counters reset");
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Result of a rate limit check.
     */
    public record RateLimitResult(
        boolean allowed,
        RateLimitTier tier,
        String rejectionReason,
        String rejectionCode,
        long retryAfterSeconds,
        int remainingRequests,
        int remainingTokens,
        int concurrentRequests
    ) {
        public static RateLimitResult allowed(RateLimitTier tier) {
            return new RateLimitResult(true, tier, null, null, 0, 0, 0, 0);
        }

        public static RateLimitResult rejected(RateLimitTier tier, String code, String reason) {
            return new RateLimitResult(false, tier, reason, code, 0, 0, 0, 0);
        }

        public static RateLimitResult rejected(RateLimitTier tier, String code, String reason,
                                              long retryAfterSeconds) {
            return new RateLimitResult(false, tier, reason, code, retryAfterSeconds, 0, 0, 0);
        }

        /**
         * Whether the client should retry after the specified time.
         */
        public boolean shouldRetryAfter() {
            return !allowed && retryAfterSeconds > 0;
        }
    }

    /**
     * Rate limiter statistics.
     */
    public record RateLimitStats(
        long totalAllowed,
        long totalRejected,
        int activeTenants,
        int configuredTenants
    ) {
        /**
         * Rejection rate as a percentage (0.0 to 100.0).
         */
        public double rejectionRate() {
            long total = totalAllowed + totalRejected;
            return total == 0 ? 0.0 : (double) totalRejected / total * 100.0;
        }
    }

    /**
     * Token bucket implementation.
     * <p>
     * Thread-safe token bucket with configurable capacity and refill rate.
     */
    private static final class TokenBucket {
        /** Maximum tokens (bucket capacity) */
        private final long capacity;

        /** Tokens refilled per interval */
        private final long refillRate;

        /** Refill interval in seconds */
        private final int refillIntervalSeconds;

        /** Current token count */
        private volatile long tokens;

        /** Last refill timestamp */
        private volatile Instant lastRefill;

        /** Clock for time tracking */
        private final Clock clock;

        TokenBucket(long capacity, long refillRate, int refillIntervalSeconds, Clock clock) {
            this.capacity = capacity;
            this.refillRate = refillRate;
            this.refillIntervalSeconds = refillIntervalSeconds;
            this.tokens = capacity;
            this.lastRefill = clock.instant();
            this.clock = clock;
        }

        /**
         * Attempts to consume tokens.
         *
         * @param count number of tokens to consume
         * @return true if tokens were consumed, false if insufficient
         */
        synchronized boolean tryConsume(long count) {
            refill();
            if (tokens >= count) {
                tokens -= count;
                return true;
            }
            return false;
        }

        /**
         * Refunds tokens (for rejected requests).
         */
        synchronized void refund(long count) {
            tokens = Math.min(capacity, tokens + count);
        }

        /**
         * Gets current token count.
         */
        long getTokens() {
            refill();
            return tokens;
        }

        /**
         * Gets seconds until next refill.
         */
        long getTimeUntilRefill() {
            Instant now = clock.instant();
            long elapsed = now.getEpochSecond() - lastRefill.getEpochSecond();
            return Math.max(0, refillIntervalSeconds - elapsed);
        }

        /**
         * Refills tokens based on elapsed time.
         */
        private void refill() {
            Instant now = clock.instant();
            long elapsed = now.getEpochSecond() - lastRefill.getEpochSecond();

            if (elapsed >= refillIntervalSeconds) {
                long intervals = elapsed / refillIntervalSeconds;
                long tokensToAdd = intervals * refillRate;
                tokens = Math.min(capacity, tokens + tokensToAdd);
                lastRefill = lastRefill.plusSeconds(intervals * refillIntervalSeconds);
            }
        }
    }

    /**
     * Configuration for RateLimiter.
     */
    private static final class Config {
        RateLimitTier defaultTier = RateLimitTier.PRO;
        int refillIntervalSeconds = 1;
        Clock clock = Clock.systemUTC();
    }

    /**
     * Builder for RateLimiter.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        /**
         * Sets the default tier for unregistered tenants.
         */
        public Builder defaultTier(RateLimitTier tier) {
            config.defaultTier = tier;
            return this;
        }

        /**
         * Sets the token refill interval in seconds.
         * <p>
         * Lower values = smoother rate limiting, higher CPU overhead.
         * Recommended: 1 second for most use cases.
         */
        public Builder refillIntervalSeconds(int seconds) {
            config.refillIntervalSeconds = seconds;
            return this;
        }

        /**
         * Sets the clock (for testing).
         */
        public Builder clock(Clock clock) {
            config.clock = clock;
            return this;
        }

        public RateLimiter build() {
            return new RateLimiter(config);
        }
    }
}
