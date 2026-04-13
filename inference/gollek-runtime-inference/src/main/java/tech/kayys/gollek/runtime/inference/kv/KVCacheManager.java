package tech.kayys.gollek.runtime.inference.kv;

import org.jboss.logging.Logger;

import tech.kayys.gollek.runtime.tensor.DType;

import java.lang.foreign.Arena;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Centralized manager for KV cache instances.
 * <p>
 * Provides:
 * <ul>
 *   <li><b>Multi-Model Support:</b> Different models get different KV cache configs</li>
 *   <li><b>OOM Protection:</b> Eviction policies when memory pressure is high</li>
 *   <li><b>Memory Quotas:</b> Per-tenant block limits</li>
 *   <li><b>Defragmentation:</b> Compact blocks when sequences finish</li>
 *   <li><b>Monitoring:</b> Per-model and per-tenant memory tracking</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * KVCacheManager manager = KVCacheManager.builder()
 *     .globalMaxBlocks(4096)
 *     .evictionPolicy(EvictionPolicy.LRU)
 *     .evictionThreshold(0.85)
 *     .build();
 *
 * // Register a model's cache
 * PagedKVCache cache = manager.registerCache("llama-3-70b",
 *     PagedKVCache.builder()
 *         .fromModelSpec(80, 64, 128, 8192, DType.FLOAT16)
 *         .build());
 *
 * // Create sequence cache
 * KVCache seqCache = manager.createSequence("llama-3-70b", "req-123", "tenant-1");
 *
 * // Check memory pressure before scheduling
 * if (manager.isMemoryPressureHigh("llama-3-70b")) {
 *     manager.evict("llama-3-70b");  // Evict oldest sequence
 * }
 * }</pre>
 *
 * @since 0.2.0
 */
public final class KVCacheManager {

    private static final Logger LOG = Logger.getLogger(KVCacheManager.class);

    /** Registered caches per model */
    private final Map<String, PagedKVCache> caches = new ConcurrentHashMap<>();

    /** Per-model arena allocations (for cleanup) */
    private final Map<String, Arena> arenas = new ConcurrentHashMap<>();

    /** Per-tenant block usage: tenantId → blocks used */
    private final Map<String, AtomicLong> tenantBlockUsage = new ConcurrentHashMap<>();

    /** Per-tenant block limits: tenantId → max blocks */
    private final Map<String, Integer> tenantBlockLimits = new ConcurrentHashMap<>();

    /** Global maximum blocks across all models */
    private final int globalMaxBlocks;

    /** Eviction policy */
    private final EvictionPolicy evictionPolicy;

    /** Eviction threshold (0.0 to 1.0) */
    private final double evictionThreshold;

    /** LRU tracking: model → sequenceId → lastAccessTime */
    private final Map<String, Map<String, AtomicLong>> lruTracker = new ConcurrentHashMap<>();

    private volatile boolean closed = false;

    private KVCacheManager(Builder builder) {
        this.globalMaxBlocks = builder.globalMaxBlocks;
        this.evictionPolicy = builder.evictionPolicy;
        this.evictionThreshold = builder.evictionThreshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    // ── Cache Registration ───────────────────────────────────────────────

    /**
     * Registers a KV cache for a model.
     *
     * @param modelId unique model identifier
     * @param cache the paged KV cache
     */
    public void registerCache(String modelId, PagedKVCache cache) {
        ensureOpen();
        caches.put(modelId, cache);
        lruTracker.put(modelId, new ConcurrentHashMap<>());
        LOG.infof("Registered KV cache for model %s: %s", modelId, cache);
    }

    /**
     * Unregisters a model's cache and frees all memory.
     *
     * @param modelId the model to unregister
     */
    public void unregisterCache(String modelId) {
        PagedKVCache cache = caches.remove(modelId);
        if (cache != null) {
            cache.close();
            Arena arena = arenas.remove(modelId);
            if (arena != null) {
                arena.close();
            }
            lruTracker.remove(modelId);
            LOG.infof("Unregistered KV cache for model %s", modelId);
        }
    }

    /**
     * Gets the cache for a model.
     */
    public PagedKVCache getCache(String modelId) {
        return caches.get(modelId);
    }

    // ── Sequence Management ──────────────────────────────────────────────

    /**
     * Creates a sequence cache for a request.
     *
     * @param modelId the model identifier
     * @param sequenceId unique sequence/request ID
     * @param tenantId tenant identifier for quota tracking
     * @return the sequence KV cache
     * @throws IllegalStateException if model not registered
     * @throws IllegalArgumentException if tenant exceeds quota
     */
    public PagedKVCache.SequenceKVCache createSequence(
            String modelId, String sequenceId, String tenantId) {
        ensureOpen();

        PagedKVCache cache = caches.get(modelId);
        if (cache == null) {
            throw new IllegalStateException("Model " + modelId + " not registered");
        }

        // Check tenant quota
        if (!checkTenantQuota(tenantId, cache)) {
            throw new IllegalArgumentException(
                "Tenant " + tenantId + " exceeded KV cache quota");
        }

        PagedKVCache.SequenceKVCache seqCache = cache.createSequenceCache(sequenceId, tenantId);
        trackAccess(modelId, sequenceId);
        return seqCache;
    }

    /**
     * Frees a sequence and updates LRU tracking.
     */
    public void freeSequence(String modelId, String sequenceId) {
        PagedKVCache cache = caches.get(modelId);
        if (cache != null) {
            String tenantId = cache.getTenantId(sequenceId);
            cache.freeSequence(sequenceId);
            Map<String, AtomicLong> modelLru = lruTracker.get(modelId);
            if (modelLru != null) {
                modelLru.remove(sequenceId);
            }
            LOG.debugf("Freed sequence %s for model %s", sequenceId, modelId);
        }
    }

    // ── Memory Management ────────────────────────────────────────────────

    /**
     * Checks if memory pressure is high for a model.
     *
     * @param modelId the model identifier
     * @return true if utilization exceeds eviction threshold
     */
    public boolean isMemoryPressureHigh(String modelId) {
        PagedKVCache cache = caches.get(modelId);
        if (cache == null) return false;
        return cache.memoryUtilization() > evictionThreshold;
    }

    /**
     * Evicts sequences based on the configured eviction policy.
     *
     * @param modelId the model to evict from
     * @return number of sequences evicted
     */
    public int evict(String modelId) {
        PagedKVCache cache = caches.get(modelId);
        if (cache == null) return 0;

        Map<String, AtomicLong> modelLru = lruTracker.get(modelId);
        if (modelLru == null || modelLru.isEmpty()) return 0;

        int evicted = 0;
        switch (evictionPolicy) {
            case LRU -> {
                // Find least recently used sequence
                String oldestSequence = null;
                long oldestTime = Long.MAX_VALUE;
                for (var entry : modelLru.entrySet()) {
                    long time = entry.getValue().get();
                    if (time < oldestTime) {
                        oldestTime = time;
                        oldestSequence = entry.getKey();
                    }
                }
                if (oldestSequence != null) {
                    LOG.warnf("Evicting LRU sequence %s for model %s (utilization=%.1f%%)",
                        oldestSequence, modelId, cache.memoryUtilization() * 100);
                    cache.freeSequence(oldestSequence);
                    modelLru.remove(oldestSequence);
                    evicted = 1;
                }
            }
            case FIFO -> {
                // Evict oldest created sequence (first in map)
                String oldestSequence = modelLru.keySet().stream().findFirst().orElse(null);
                if (oldestSequence != null) {
                    LOG.warnf("Evicting FIFO sequence %s for model %s", oldestSequence, modelId);
                    cache.freeSequence(oldestSequence);
                    modelLru.remove(oldestSequence);
                    evicted = 1;
                }
            }
        }

        return evicted;
    }

    /**
     * Evicts sequences until memory pressure is below threshold.
     *
     * @param modelId the model to relieve pressure from
     * @return total evicted sequences
     */
    public int evictUntilRelieved(String modelId) {
        int totalEvicted = 0;
        while (isMemoryPressureHigh(modelId)) {
            int evicted = evict(modelId);
            if (evicted == 0) break;  // Nothing left to evict
            totalEvicted += evicted;
        }
        return totalEvicted;
    }

    // ── Tenant Quota Management ──────────────────────────────────────────

    /**
     * Sets a block limit for a tenant.
     */
    public void setTenantQuota(String tenantId, int maxBlocks) {
        tenantBlockLimits.put(tenantId, maxBlocks);
    }

    /**
     * Gets a tenant's current block usage.
     */
    public long getTenantUsage(String tenantId) {
        AtomicLong usage = tenantBlockUsage.get(tenantId);
        return usage != null ? usage.get() : 0;
    }

    private boolean checkTenantQuota(String tenantId, PagedKVCache cache) {
        Integer limit = tenantBlockLimits.get(tenantId);
        if (limit == null) return true;  // No quota

        long usage = getTenantUsage(tenantId);
        // Estimate: each sequence uses ~cache.usedBlocks() / cache.sequenceCount blocks
        int sequenceCount = cache.sequenceBlockTables.size();
        if (sequenceCount == 0) return true;

        long avgBlocksPerSequence = cache.usedBlocks() / sequenceCount;
        return usage + avgBlocksPerSequence <= limit;
    }

    // ── Monitoring ───────────────────────────────────────────────────────

    /**
     * Gets cache statistics for a model.
     */
    public CacheStats getStats(String modelId) {
        PagedKVCache cache = caches.get(modelId);
        if (cache == null) {
            return CacheStats.EMPTY;
        }
        return new CacheStats(
            cache.usedBlocks(),
            cache.totalBlocks(),
            cache.memoryUtilization(),
            cache.sequenceBlockTables.size(),
            cache.totalAppends(),
            cache.prefixCacheHits(),
            cache.peakUsedBlocks()
        );
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    /**
     * Closes all caches and frees all memory.
     */
    public void close() {
        if (!closed) {
            closed = true;
            for (var entry : caches.entrySet()) {
                entry.getValue().close();
                Arena arena = arenas.remove(entry.getKey());
                if (arena != null) {
                    arena.close();
                }
            }
            caches.clear();
            arenas.clear();
            tenantBlockUsage.clear();
            tenantBlockLimits.clear();
            lruTracker.clear();
            LOG.info("KVCacheManager closed");
        }
    }

    // ── Internal Methods ─────────────────────────────────────────────────

    private void trackAccess(String modelId, String sequenceId) {
        Map<String, AtomicLong> modelLru = lruTracker.computeIfAbsent(
            modelId, k -> new ConcurrentHashMap<>());
        modelLru.computeIfAbsent(sequenceId, k -> new AtomicLong())
                .set(System.nanoTime());
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("KVCacheManager is closed");
        }
    }

    // ── Nested Types ─────────────────────────────────────────────────────

    /**
     * Eviction policy for KV cache sequences.
     */
    public enum EvictionPolicy {
        /** Least Recently Used — evict sequence with oldest last access */
        LRU,
        /** First In First Out — evict sequence created earliest */
        FIFO
    }

    /**
     * Cache statistics snapshot.
     */
    public record CacheStats(
        int usedBlocks,
        int totalBlocks,
        double utilization,
        int activeSequences,
        long totalAppends,
        long prefixCacheHits,
        int peakUsedBlocks
    ) {
        public static final CacheStats EMPTY = new CacheStats(0, 0, 0.0, 0, 0, 0, 0);
    }

    /**
     * Builder for KVCacheManager.
     */
    public static final class Builder {
        private int globalMaxBlocks = 4096;
        private EvictionPolicy evictionPolicy = EvictionPolicy.LRU;
        private double evictionThreshold = 0.85;

        private Builder() {}

        /**
         * Sets the global maximum blocks across all models.
         */
        public Builder globalMaxBlocks(int globalMaxBlocks) {
            this.globalMaxBlocks = globalMaxBlocks;
            return this;
        }

        /**
         * Sets the eviction policy.
         */
        public Builder evictionPolicy(EvictionPolicy evictionPolicy) {
            this.evictionPolicy = evictionPolicy;
            return this;
        }

        /**
         * Sets the eviction threshold (0.0 to 1.0).
         * <p>
         * When memory utilization exceeds this threshold, eviction is triggered.
         */
        public Builder evictionThreshold(double evictionThreshold) {
            this.evictionThreshold = evictionThreshold;
            return this;
        }

        public KVCacheManager build() {
            return new KVCacheManager(this);
        }
    }
}
