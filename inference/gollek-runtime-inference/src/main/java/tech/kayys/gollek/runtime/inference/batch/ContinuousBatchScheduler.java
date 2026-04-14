package tech.kayys.gollek.runtime.inference.batch;

import tech.kayys.gollek.runtime.inference.kv.PagedKVCache;
import tech.kayys.gollek.runtime.inference.kv.KVCacheManager;
import tech.kayys.gollek.runtime.inference.kv.PagedAttentionKernel;
import tech.kayys.gollek.runtime.inference.kv.TurboQuantKVCacheAdapter;
import tech.kayys.gollek.runtime.inference.kv.KVCacheStorageMode;
import tech.kayys.gollek.runtime.inference.streaming.TokenStreamer;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Production-grade vLLM-style continuous batching scheduler.
 * <p>
 * Unlike static batching (which waits for a full batch before executing),
 * continuous batching dynamically adds and removes requests each iteration:
 * <pre>
 *   t0: [A, B]        — 2 requests active
 *   t1: [A, B, C]     — C joins mid-flight
 *   t2: [B, C]        — A finishes, removed
 *   t3: [B, C, D, E]  — D, E join
 * </pre>
 * <p>
 * This maximizes GPU utilization and minimizes latency.
 * <p>
 * <h2>Integration with PagedKVCache</h2>
 * <p>
 * Each {@link BatchRequest} gets its own {@link PagedKVCache.SequenceKVCache} for
 * storing KV states. The scheduler coordinates block table management with the
 * {@link PagedAttentionKernel} for efficient memory access.
 * <p>
 * <h2>Features</h2>
 * <ul>
 *   <li><b>Dynamic Request Admission:</b> Add requests mid-flight without waiting for batch completion</li>
 *   <li><b>Memory-Aware Scheduling:</b> Reject requests when KV cache is full</li>
 *   <li><b>Request Cancellation:</b> Cancel in-flight requests and reclaim KV cache blocks</li>
 *   <li><b>Per-Request KV Cache:</b> Isolated KV storage per sequence</li>
 *   <li><b>Streaming Support:</b> Token-by-token streaming via {@link TokenStreamer}</li>
 *   <li><b>Multi-Tenant:</b> Tenant-aware scheduling with quota enforcement</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ContinuousBatchScheduler scheduler = ContinuousBatchScheduler.builder()
 *     .maxBatchSize(128)
 *     .kvCacheManager(cacheManager)
 *     .pagedAttentionKernel(pagedAttention)
 *     .modelId("llama-3-70b")
 *     .build();
 *
 * scheduler.start();
 *
 * // Submit requests
 * scheduler.submit(BatchRequest.builder()
 *     .tenantId("tenant-1")
 *     .promptTokens(promptIds)
 *     .maxTokens(256)
 *     .streamer(token -> System.out.print(token))
 *     .build());
 *
 * // Graceful shutdown
 * scheduler.stop();
 * }</pre>
 *
 * @see PagedKVCache
 * @see PagedAttentionKernel
 * @see KVCacheManager
 * @since 0.2.0
 */
public final class ContinuousBatchScheduler {

    private static final Logger LOG = Logger.getLogger(ContinuousBatchScheduler.class);

    // ── Configuration ───────────────────────────────────────────────────

    /** Maximum concurrent active requests */
    private final int maxBatchSize;

    /** Model this scheduler is serving */
    private final String modelId;

    /** KV cache manager for memory tracking */
    private final KVCacheManager kvCacheManager;

    /** Paged attention kernel for GPU execution */
    private final PagedAttentionKernel pagedAttention;

    /** Storage mode (full precision or TurboQuant compressed) */
    private final KVCacheStorageMode storageMode;

    /** TurboQuant adapter (if using compressed storage) */
    private final TurboQuantKVCacheAdapter turboQuantAdapter;

    // ── Request Management ──────────────────────────────────────────────

    /** Incoming request queue */
    private final RequestQueue queue = new RequestQueue();

    /** Currently executing requests */
    private final List<BatchRequest> active = new ArrayList<>();

    /** Per-request KV caches: requestId → SequenceKVCache */
    private final Map<String, PagedKVCache.SequenceKVCache> requestCaches = new ConcurrentHashMap<>();

    /** Per-request TurboQuant caches: requestId → TurboQuantSequenceCache */
    private final Map<String, TurboQuantKVCacheAdapter.TurboQuantSequenceCache> turboQuantRequestCaches = new ConcurrentHashMap<>();

    // ── Scheduler State ─────────────────────────────────────────────────

    /** Whether the scheduler is running */
    private final AtomicBoolean running = new AtomicBoolean(false);

    /** Graceful shutdown flag */
    private final AtomicBoolean shuttingDown = new AtomicBoolean(false);

    // ── Statistics ──────────────────────────────────────────────────────

    /** Total requests processed */
    private final AtomicLong totalBatchedRequests = new AtomicLong(0);

    /** Total tokens generated */
    private final AtomicLong totalTokensGenerated = new AtomicLong(0);

    /** Total scheduling iterations */
    private final AtomicLong totalIterations = new AtomicLong(0);

    /** Requests rejected due to memory pressure */
    private final AtomicLong totalRejected = new AtomicLong(0);

    /** Requests cancelled mid-flight */
    private final AtomicLong totalCancelled = new AtomicLong(0);

    /** Scheduler thread */
    private volatile Thread schedulerThread;

    /**
     * Creates a new continuous batching scheduler with default configuration.
     *
     * @param maxBatchSize maximum concurrent requests
     */
    public ContinuousBatchScheduler(int maxBatchSize) {
        this(maxBatchSize, "default", null, null, KVCacheStorageMode.FULL_PRECISION, null);
    }

    /**
     * Creates a new continuous batching scheduler with full production configuration.
     *
     * @param maxBatchSize maximum concurrent requests
     * @param modelId model identifier for KV cache management
     * @param kvCacheManager cache manager for memory tracking
     * @param pagedAttention attention kernel for execution
     * @param storageMode storage mode (full precision or TurboQuant compressed)
     * @param turboQuantAdapter TurboQuant adapter (if using compressed storage)
     */
    public ContinuousBatchScheduler(
            int maxBatchSize,
            String modelId,
            KVCacheManager kvCacheManager,
            PagedAttentionKernel pagedAttention,
            KVCacheStorageMode storageMode,
            TurboQuantKVCacheAdapter turboQuantAdapter) {
        this.maxBatchSize = maxBatchSize;
        this.modelId = modelId;
        this.kvCacheManager = kvCacheManager;
        this.pagedAttention = pagedAttention;
        this.storageMode = storageMode;
        this.turboQuantAdapter = turboQuantAdapter;
    }

    /**
     * Creates a builder for configuring this scheduler.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Request Submission ──────────────────────────────────────────────

    /**
     * Submits a new inference request for processing.
     * <p>
     * The request is queued and will be picked up by the scheduler loop
     * when a batch slot becomes available.
     *
     * @param request the batch request to submit
     * @throws IllegalStateException if scheduler is shutting down
     * @throws OutOfMemoryError if KV cache is full
     */
    public void submit(BatchRequest request) {
        if (shuttingDown.get()) {
            throw new IllegalStateException("Scheduler is shutting down");
        }

        // Check memory pressure before admitting request
        if (kvCacheManager != null && kvCacheManager.isMemoryPressureHigh(modelId)) {
            totalRejected.incrementAndGet();
            LOG.warnf("Rejecting request: KV cache memory pressure high for model %s", modelId);
            throw new OutOfMemoryError(
                "KV cache full for model " + modelId + 
                ". Try again later or reduce batch size.");
        }

        queue.submit(request);
        LOG.debugf("Submitted request %s to queue (depth: %d)", request.id, queue.pendingCount());
    }

    /**
     * Cancels a running request and reclaims its KV cache.
     *
     * @param requestId the request ID to cancel
     * @return true if request was found and cancelled
     */
    public boolean cancelRequest(String requestId) {
        // Check active requests
        for (BatchRequest req : active) {
            if (req.id.toString().equals(requestId)) {
                req.finished = true;
                totalCancelled.incrementAndGet();
                
                // Free KV cache
                freeRequestCache(requestId);
                
                LOG.infof("Cancelled active request %s", requestId);
                return true;
            }
        }

        // Can't cancel from queue easily (would need to scan PriorityBlockingQueue)
        // In production, this would require a more sophisticated queue
        LOG.warnf("Could not find request %s to cancel", requestId);
        return false;
    }

    // ── Scheduler Loop ──────────────────────────────────────────────────

    /**
     * Starts the scheduler loop in a background thread.
     * <p>
     * The loop runs until {@link #stop()} is called. It will gracefully
     * drain active requests before stopping.
     */
    public void start(BatchExecutor executor) {
        if (!running.compareAndSet(false, true)) {
            LOG.warn("Scheduler already running");
            return;
        }

        shuttingDown.set(false);
        schedulerThread = Thread.ofVirtual()
            .name("gollek-continuous-batch-scheduler-" + modelId)
            .start(() -> schedulerLoop(executor));

        LOG.infof("Continuous batching scheduler started: model=%s, maxBatchSize=%d",
            modelId, maxBatchSize);
    }

    /**
     * Initiates graceful shutdown.
     * <p>
     * The scheduler will finish processing active requests before stopping.
     * New requests will be rejected.
     */
    public void stop() {
        if (!running.get()) {
            return;
        }

        shuttingDown.set(true);
        LOG.infof("Initiating graceful shutdown for model %s (active requests: %d)",
            modelId, active.size());
    }

    /**
     * Force stops the scheduler immediately.
     * <p>
     * Active requests will be cancelled and their KV caches freed.
     */
    public void stopNow() {
        shuttingDown.set(true);
        running.set(false);

        // Cancel all active requests
        for (BatchRequest req : active) {
            req.finished = true;
            totalCancelled.incrementAndGet();
            freeRequestCache(req.id.toString());
        }
        active.clear();

        LOG.warnf("Force stopped scheduler for model %s (cancelled %d requests)",
            modelId, active.size());
    }

    /**
     * Waits for the scheduler to finish (for graceful shutdown).
     *
     * @param timeoutMs maximum time to wait in milliseconds
     * @throws InterruptedException if interrupted while waiting
     */
    public void awaitTermination(long timeoutMs) throws InterruptedException {
        if (schedulerThread != null) {
            schedulerThread.join(timeoutMs);
        }
    }

    // ── Query Methods ───────────────────────────────────────────────────

    /** Whether the scheduler is running. */
    public boolean isRunning() {
        return running.get();
    }

    /** Gets the effective batch size (max concurrent slots). */
    public int getActiveBatchSize() {
        return maxBatchSize;
    }

    /** Whether the scheduler is shutting down. */
    public boolean isShuttingDown() {
        return shuttingDown.get();
    }

    /** Number of currently active requests. */
    public int activeCount() {
        return active.size();
    }

    /** Alias for {@link #activeCount()} used by legacy tests. */
    public int getBatchCount() {
        return activeCount();
    }

    /** Number of pending requests in queue. */
    public int pendingCount() {
        return queue.pendingCount();
    }

    /** Alias for {@link #pendingCount()} used by legacy tests. */
    public int getQueueDepth() {
        return pendingCount();
    }

    /** Total number of requests ever batched by this scheduler. */
    public long getTotalBatchedRequests() {
        return totalBatchedRequests.get();
    }

    /** Total tokens generated across all requests. */
    public long getTotalTokensGenerated() {
        return totalTokensGenerated.get();
    }

    /** Total scheduling iterations. */
    public long getTotalIterations() {
        return totalIterations.get();
    }

    /** Total requests rejected due to memory pressure. */
    public long getTotalRejected() {
        return totalRejected.get();
    }

    /** Total requests cancelled mid-flight. */
    public long getTotalCancelled() {
        return totalCancelled.get();
    }

    /** Gets average tokens generated per request. */
    public double getAvgTokensPerRequest() {
        long completed = totalBatchedRequests.get();
        return completed == 0 ? 0.0 : (double) totalTokensGenerated.get() / completed;
    }

    // ── Internal Scheduler Loop ─────────────────────────────────────────

    private void schedulerLoop(BatchExecutor executor) {
        try {
            while (running.get()) {
                totalIterations.incrementAndGet();

                // 1. Drain new requests into active batch (respecting memory limits)
                int availableSlots = maxBatchSize - active.size();
                if (availableSlots > 0) {
                    List<BatchRequest> drained = queue.drain(availableSlots);
                    for (BatchRequest req : drained) {
                        // Create KV cache for this request
                        if (storageMode.isCompressed() && turboQuantAdapter != null) {
                            // Use TurboQuant compressed cache
                            try {
                                TurboQuantKVCacheAdapter.TurboQuantSequenceCache seqCache = 
                                    turboQuantAdapter.createSequenceCache(req.id.toString(), req.tenantId);
                                turboQuantRequestCaches.put(req.id.toString(), seqCache);
                            } catch (Exception e) {
                                LOG.errorf(e, "Failed to create TurboQuant cache for request %s", req.id);
                                req.finished = true;  // Mark as failed
                                continue;
                            }
                        } else if (kvCacheManager != null) {
                            // Use standard PagedKVCache
                            try {
                                PagedKVCache.SequenceKVCache seqCache = kvCacheManager.createSequence(
                                    modelId, req.id.toString(), req.tenantId);
                                requestCaches.put(req.id.toString(), seqCache);
                            } catch (Exception e) {
                                LOG.errorf(e, "Failed to create KV cache for request %s", req.id);
                                req.finished = true;  // Mark as failed
                                continue;
                            }
                        }
                        active.add(req);
                        totalBatchedRequests.incrementAndGet();
                    }
                }

                // 2. Check if we should stop
                if (shuttingDown.get() && active.isEmpty()) {
                    LOG.infof("Scheduler stopped gracefully: model=%s, processed=%d",
                        modelId, totalBatchedRequests.get());
                    break;
                }

                // 3. Skip if no active requests
                if (active.isEmpty()) {
                    sleep(1);
                    continue;
                }

                // 4. Execute one decode step for all active requests
                try {
                    executor.executeBatchStep(active);
                } catch (Exception e) {
                    LOG.errorf(e, "Error executing batch step for model %s", modelId);
                    // Mark all active as failed
                    for (BatchRequest req : active) {
                        req.finished = true;
                        freeRequestCache(req.id.toString());
                    }
                    active.clear();
                    continue;
                }

                // 5. Remove finished/cancelled requests and reclaim memory
                active.removeIf(req -> {
                    if (req.finished) {
                        freeRequestCache(req.id.toString());
                        return true;
                    }
                    return false;
                });
            }
        } catch (Exception e) {
            LOG.errorf(e, "Fatal error in scheduler loop for model %s", modelId);
        } finally {
            running.set(false);
            // Clean up any remaining requests
            for (BatchRequest req : active) {
                freeRequestCache(req.id.toString());
            }
            active.clear();
        }
    }

    // ── Helper Methods ──────────────────────────────────────────────────

    private void freeRequestCache(String requestId) {
        // Free TurboQuant cache if using compressed mode
        TurboQuantKVCacheAdapter.TurboQuantSequenceCache tqCache = turboQuantRequestCaches.remove(requestId);
        if (tqCache != null && turboQuantAdapter != null) {
            turboQuantAdapter.freeSequence(requestId);
        }

        // Free standard cache if using full precision mode
        PagedKVCache.SequenceKVCache standardCache = requestCaches.remove(requestId);
        if (standardCache != null && kvCacheManager != null) {
            kvCacheManager.freeSequence(modelId, requestId);
        }
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ── Builder ─────────────────────────────────────────────────────────

    /**
     * Builder for ContinuousBatchScheduler.
     */
    public static final class Builder {
        private int maxBatchSize = 128;
        private String modelId = "default";
        private KVCacheManager kvCacheManager;
        private PagedAttentionKernel pagedAttention;
        private KVCacheStorageMode storageMode = KVCacheStorageMode.FULL_PRECISION;
        private TurboQuantKVCacheAdapter turboQuantAdapter;

        private Builder() {}

        /**
         * Sets the maximum concurrent requests.
         */
        public Builder maxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }

        /**
         * Sets the model identifier.
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Sets the KV cache manager for memory tracking.
         */
        public Builder kvCacheManager(KVCacheManager kvCacheManager) {
            this.kvCacheManager = kvCacheManager;
            return this;
        }

        /**
         * Sets the paged attention kernel.
         */
        public Builder pagedAttention(PagedAttentionKernel pagedAttention) {
            this.pagedAttention = pagedAttention;
            return this;
        }

        /**
         * Sets the storage mode.
         * <p>
         * Use {@link KVCacheStorageMode#TURBOQUANT_3BIT} for 6× compression.
         */
        public Builder storageMode(KVCacheStorageMode storageMode) {
            this.storageMode = storageMode;
            return this;
        }

        /**
         * Sets the TurboQuant adapter for compressed storage.
         * <p>
         * Required if storageMode is TURBOQUANT_*.
         */
        public Builder turboQuantAdapter(TurboQuantKVCacheAdapter turboQuantAdapter) {
            this.turboQuantAdapter = turboQuantAdapter;
            return this;
        }

        /**
         * Builds the ContinuousBatchScheduler.
         */
        public ContinuousBatchScheduler build() {
            // Auto-adjust maxBatchSize based on compression
            int effectiveMaxBatchSize = maxBatchSize;
            if (storageMode.isCompressed()) {
                // With 6× compression, can handle 6× more requests
                effectiveMaxBatchSize = (int) (maxBatchSize * storageMode.compressionRatio() / 2.0);
            }

            return new ContinuousBatchScheduler(
                effectiveMaxBatchSize, modelId, kvCacheManager, pagedAttention,
                storageMode, turboQuantAdapter);
        }
    }

    // ── Legacy Interface ────────────────────────────────────────────────

    /**
     * Callback interface for executing a single decode step on a batch.
     * Implementors perform the actual model forward pass and token sampling.
     */
    @FunctionalInterface
    public interface BatchExecutor {
        void executeBatchStep(List<BatchRequest> batch);
    }
}

