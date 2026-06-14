/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.plugin.quota;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Engine-level quota enforcer that runs at the {@link InferencePhase#AUTHORIZE}
 * phase.  It enforces two independent limits per tenant:
 *
 * <ol>
 *   <li><b>Rate Limit</b> – maximum number of requests per minute (RPM).</li>
 *   <li><b>Bulkhead</b>   – maximum concurrent in-flight requests.</li>
 * </ol>
 *
 * <p>Bulkhead limits are independent of (and complementary to) the per-provider
 * bulkheads in {@link tech.kayys.gollek.plugin.observability.ReliabilityPlugin}.
 * The engine-level bulkhead protects the inference pipeline itself; the provider
 * bulkhead protects individual upstream connections.
 *
 * <h3>Default Limits (override via {@code gollek.quota.*} config)</h3>
 * <ul>
 *   <li>RPM default: 60</li>
 *   <li>Bulkhead concurrency default: 10</li>
 * </ul>
 */
@ApplicationScoped
public class EngineQuotaEnforcer implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(EngineQuotaEnforcer.class.getName());

    private static final long DEFAULT_RPM        = 60;
    private static final int  DEFAULT_CONCURRENT = 10;
    private static final long WINDOW_MS          = 60_000L; // 1-minute sliding window

    /** Per-tenant: epoch-ms of window start and request count in that window. */
    private final Map<String, long[]> rateLimitWindows = new ConcurrentHashMap<>();

    /** Per-tenant concurrency bulkhead. */
    private final Map<String, Semaphore> tenantBulkheads = new ConcurrentHashMap<>();

    /** Per-tenant custom RPM limits (populated via admin API / config). */
    private final Map<String, Long> customRpm = new ConcurrentHashMap<>();

    /** Per-tenant custom concurrency limits. */
    private final Map<String, Integer> customConcurrency = new ConcurrentHashMap<>();

    @Override
    public String pluginId() {
        return "gollek.engine.quota";
    }

    @Override
    public String displayName() {
        return "Engine Quota Enforcer";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.AUTHORIZE;
    }

    @Override
    public int order() {
        return 50; // After authentication, before routing
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        String tenantId = resolveTenantId(context);

        checkRateLimit(tenantId);
        checkBulkhead(tenantId, context);
    }

    // -----------------------------------------------------------------------
    // Rate-limit enforcement (token-bucket per minute)
    // -----------------------------------------------------------------------

    private synchronized void checkRateLimit(String tenantId) throws PluginException {
        long rpmLimit = customRpm.getOrDefault(tenantId, DEFAULT_RPM);
        long[] window = rateLimitWindows.computeIfAbsent(tenantId, k -> new long[]{nowMs(), 0L});

        long elapsed = nowMs() - window[0];
        if (elapsed > WINDOW_MS) {
            // Slide window
            window[0] = nowMs();
            window[1] = 0L;
        }

        if (window[1] >= rpmLimit) {
            long retryAfterSec = (WINDOW_MS - elapsed) / 1000;
            throw new PluginException(
                String.format("Rate limit exceeded for tenant=%s. rpm_limit=%d, retry_after=%ds",
                              tenantId, rpmLimit, retryAfterSec),
                pluginId());
        }
        window[1]++;
    }

    // -----------------------------------------------------------------------
    // Bulkhead enforcement
    // -----------------------------------------------------------------------

    private void checkBulkhead(String tenantId, ExecutionContext context) throws PluginException {
        int concurrency = customConcurrency.getOrDefault(tenantId, DEFAULT_CONCURRENT);
        Semaphore sem   = tenantBulkheads.computeIfAbsent(
            tenantId, k -> new Semaphore(concurrency, true));

        boolean acquired = sem.tryAcquire();
        if (!acquired) {
            throw new PluginException(
                String.format("Bulkhead full for tenant=%s. max_concurrent=%d", tenantId, concurrency),
                pluginId());
        }

        // Store so CLEANUP phase can release the permit
        context.putVariable("_quotaBulkheadSemaphore", sem);
        LOG.fine(() -> String.format(
            "[QuotaEnforcer] Admitted tenant=%s available_permits=%d",
            tenantId, sem.availablePermits()));
    }

    // -----------------------------------------------------------------------
    // Admin API helpers
    // -----------------------------------------------------------------------

    /**
     * Sets a custom RPM limit for a tenant (callable from admin REST API).
     */
    public void setRpmLimit(String tenantId, long rpm) {
        customRpm.put(tenantId, rpm);
        LOG.info("[QuotaEnforcer] RPM limit updated: tenant=" + tenantId + " rpm=" + rpm);
    }

    /**
     * Sets a custom concurrency limit for a tenant.
     * Recreates the semaphore to apply the new limit.
     */
    public void setConcurrencyLimit(String tenantId, int maxConcurrent) {
        customConcurrency.put(tenantId, maxConcurrent);
        tenantBulkheads.put(tenantId, new Semaphore(maxConcurrent, true));
        LOG.info("[QuotaEnforcer] Concurrency limit updated: tenant=" + tenantId +
                 " max_concurrent=" + maxConcurrent);
    }

    /**
     * Returns current rate-limit window stats for a tenant (for admin dashboard).
     */
    public Map<String, Object> getStats(String tenantId) {
        long[] window = rateLimitWindows.getOrDefault(tenantId, new long[]{0L, 0L});
        Semaphore sem = tenantBulkheads.get(tenantId);
        return Map.of(
            "tenantId",          tenantId,
            "windowStartMs",     window[0],
            "requestsInWindow",  window[1],
            "rpmLimit",          customRpm.getOrDefault(tenantId, DEFAULT_RPM),
            "availablePermits",  sem != null ? sem.availablePermits() : DEFAULT_CONCURRENT
        );
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private String resolveTenantId(ExecutionContext context) {
        return context.getVariable("tenantId", String.class)
                      .orElseGet(() -> {
                          try { return context.requestContext().tenantId(); }
                          catch (Exception e) { return "default"; }
                      });
    }

    private long nowMs() {
        return Instant.now().toEpochMilli();
    }
}
