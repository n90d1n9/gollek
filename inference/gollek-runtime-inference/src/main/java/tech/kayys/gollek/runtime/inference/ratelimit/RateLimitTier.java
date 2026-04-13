package tech.kayys.gollek.runtime.inference.ratelimit;

/**
 * Rate limit tier definitions for multi-tenant inference serving.
 * <p>
 * Each tier defines quotas for requests-per-minute, tokens-per-minute,
 * and concurrent requests. Tenants are assigned to tiers based on
 * subscription level or API key configuration.
 *
 * <h2>Tier Comparison</h2>
 * <table>
 *   <tr><th>Tier</th><th>Req/min</th><th>Tokens/min</th><th>Concurrent</th><th>Max Context</th></tr>
 *   <tr><td>FREE</td><td>10</td><td>10K</td><td>1</td><td>2K</td></tr>
 *   <tr><td>BASIC</td><td>60</td><td>100K</td><td>5</td><td>8K</td></tr>
 *   <tr><td>PRO</td><td>300</td><td>1M</td><td>20</td><td>32K</td></tr>
 *   <tr><td>ENTERPRISE</td><td>3000</td><td>10M</td><td>100</td><td>128K</td></tr>
 *   <tr><td>UNLIMITED</td><td>∞</td><td>∞</td><td>∞</td><td>∞</td></tr>
 * </table>
 *
 * @since 0.2.0
 */
public enum RateLimitTier {

    /**
     * Free tier — evaluation and testing only.
     */
    FREE(
        10,         // requests per minute
        10_000,     // tokens per minute
        1,          // max concurrent requests
        2_048       // max context length
    ),

    /**
     * Basic tier — small projects and development.
     */
    BASIC(
        60,
        100_000,
        5,
        8_192
    ),

    /**
     * Professional tier — production applications.
     */
    PRO(
        300,
        1_000_000,
        20,
        32_768
    ),

    /**
     * Enterprise tier — high-volume production.
     */
    ENTERPRISE(
        3_000,
        10_000_000,
        100,
        131_072
    ),

    /**
     * Unlimited — internal services, no rate limiting.
     */
    UNLIMITED(
        Integer.MAX_VALUE,
        Integer.MAX_VALUE,
        Integer.MAX_VALUE,
        Integer.MAX_VALUE
    );

    /** Maximum requests per minute */
    private final int maxRequestsPerMinute;

    /** Maximum tokens per minute (input + output) */
    private final int maxTokensPerMinute;

    /** Maximum concurrent requests */
    private final int maxConcurrentRequests;

    /** Maximum context length (tokens) */
    private final int maxContextLength;

    RateLimitTier(int maxRequestsPerMinute, int maxTokensPerMinute,
                 int maxConcurrentRequests, int maxContextLength) {
        this.maxRequestsPerMinute = maxRequestsPerMinute;
        this.maxTokensPerMinute = maxTokensPerMinute;
        this.maxConcurrentRequests = maxConcurrentRequests;
        this.maxContextLength = maxContextLength;
    }

    public int getMaxRequestsPerMinute() { return maxRequestsPerMinute; }
    public int getMaxTokensPerMinute() { return maxTokensPerMinute; }
    public int getMaxConcurrentRequests() { return maxConcurrentRequests; }
    public int getMaxContextLength() { return maxContextLength; }

    /**
     * Returns whether this tier has unlimited quota.
     */
    public boolean isUnlimited() {
        return this == UNLIMITED;
    }

    /**
     * Gets tier by name (case-insensitive).
     */
    public static RateLimitTier fromName(String name) {
        for (RateLimitTier tier : values()) {
            if (tier.name().equalsIgnoreCase(name)) {
                return tier;
            }
        }
        return FREE;  // Default to free tier
    }
}
