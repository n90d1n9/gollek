package tech.kayys.gollek.multimodal;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;

import java.util.*;
import java.util.concurrent.*;

/**
 * Request batcher that coalesces concurrent inference requests into bulk calls.
 *
 * <h3>Problem</h3>
 * <p>Under high concurrency, N identical or batachable requests arrive within
 * a short window and each triggers a separate provider call.  This wastes
 * tokens, increases cost, and creates head-of-line blocking.
 *
 * <h3>Solution — Coalesce window</h3>
 * <p>Requests are held in a pending queue for up to {@link #windowMs} ms.
 * When the window closes (or {@link #maxBatchSize} is reached), all queued
 * requests are dispatched together as a single provider call (where supported)
 * or fan-fanned in parallel with shared connection slots.
 *
 * <h3>Duplicate collapse</h3>
 * <p>If two requests in the same window have the same content hash (from
 * {@link tech.kayys.gollek.cache.SemanticResponseCache}), only one upstream
 * call is made; both waiting callers receive the same response.
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.batcher.enabled=true
 *   gollek.batcher.window-ms=20
 *   gollek.batcher.max-batch-size=32
 *   gollek.batcher.dedup-enabled=true
 * </pre>
 */
@ApplicationScoped
public class RequestBatcher {

    private static final Logger LOG = Logger.getLogger(RequestBatcher.class);

    @ConfigProperty(name = "gollek.batcher.enabled",        defaultValue = "true")
    boolean enabled;

    @ConfigProperty(name = "gollek.batcher.window-ms",      defaultValue = "20")
    long windowMs;

    @ConfigProperty(name = "gollek.batcher.max-batch-size", defaultValue = "32")
    int maxBatchSize;

    @ConfigProperty(name = "gollek.batcher.dedup-enabled",  defaultValue = "true")
    boolean dedupEnabled;

    // providerId → pending batch
    private final ConcurrentHashMap<String, PendingBatch> batches = new ConcurrentHashMap<>();
    private final ScheduledExecutorService flusher =
            Executors.newSingleThreadScheduledExecutor(
                    r -> Thread.ofVirtual().name("batcher-flush").unstarted(r));

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Attempt to batch a request.  Returns a {@link Uni} that will complete
     * when this request's slot in the batch is flushed and a response received.
     *
     * @param request     the incoming request
     * @param contentHash SHA-256 of the request (from SemanticResponseCache)
     * @param dispatcher  the function to call when a batch is flushed
     */
    public Uni<MultimodalResponse> submit(
            MultimodalRequest request,
            String contentHash,
            java.util.function.Function<MultimodalRequest, Uni<MultimodalResponse>> dispatcher) {

        if (!enabled) return dispatcher.apply(request);

        String key = request.getModel();
        PendingBatch batch = batches.computeIfAbsent(key, k -> new PendingBatch(k, windowMs));

        // Duplicate collapse: if same hash already in-flight, wait for its result
        if (dedupEnabled) {
            CompletableFuture<MultimodalResponse> existing = batch.findDuplicate(contentHash);
            if (existing != null) {
                LOG.debugf("[BATCHER-DEDUP] Collapsing duplicate request %s", contentHash);
                return Uni.createFrom().completionStage(existing);
            }
        }

        CompletableFuture<MultimodalResponse> future = new CompletableFuture<>();
        int batchSize = batch.add(contentHash, request, future);

        LOG.debugf("[BATCHER] Queued request %s in batch for %s (size=%d)",
                request.getRequestId(), key, batchSize);

        // Flush immediately if batch is full
        if (batchSize >= maxBatchSize) {
            flushBatch(key, dispatcher);
        } else if (batchSize == 1) {
            // First item: schedule a flush after window
            flusher.schedule(() -> flushBatch(key, dispatcher),
                    windowMs, TimeUnit.MILLISECONDS);
        }

        return Uni.createFrom().completionStage(future);
    }

    // -------------------------------------------------------------------------
    // Flush
    // -------------------------------------------------------------------------

    private void flushBatch(String key,
            java.util.function.Function<MultimodalRequest, Uni<MultimodalResponse>> dispatcher) {
        PendingBatch batch = batches.remove(key);
        if (batch == null || batch.isEmpty()) return;

        List<PendingBatch.Slot> slots = batch.drain();
        LOG.debugf("[BATCHER-FLUSH] Flushing %d requests for %s", slots.size(), key);

        // Dispatch each request concurrently (parallel fan-out within the batch)
        for (PendingBatch.Slot slot : slots) {
            dispatcher.apply(slot.request())
                    .subscribe().with(
                            response -> slot.future().complete(response),
                            error    -> slot.future().completeExceptionally(error));
        }
    }

    /** Returns batcher stats (queue depths, total batches flushed). */
    public Map<String, Object> stats() {
        Map<String, Object> s = new LinkedHashMap<>();
        batches.forEach((k, v) -> s.put(k, Map.of(
                "pendingSlots", v.size(),
                "windowMs", windowMs)));
        return s;
    }

    // =========================================================================
    // Inner: PendingBatch
    // =========================================================================

    static final class PendingBatch {
        private final String key;
        private final long   windowMs;
        private final List<Slot> slots = Collections.synchronizedList(new ArrayList<>());
        private final Map<String, CompletableFuture<MultimodalResponse>> hashIndex =
                new ConcurrentHashMap<>();

        PendingBatch(String key, long windowMs) {
            this.key      = key;
            this.windowMs = windowMs;
        }

        /** Adds a slot; returns the new batch size. */
        int add(String hash, MultimodalRequest req,
                CompletableFuture<MultimodalResponse> future) {
            slots.add(new Slot(hash, req, future));
            hashIndex.put(hash, future);
            return slots.size();
        }

        /** Returns an existing future if the same hash is already in this batch. */
        CompletableFuture<MultimodalResponse> findDuplicate(String hash) {
            return hashIndex.get(hash);
        }

        List<Slot> drain() {
            List<Slot> copy = new ArrayList<>(slots);
            slots.clear();
            hashIndex.clear();
            return copy;
        }

        boolean isEmpty() { return slots.isEmpty(); }
        int     size()    { return slots.size(); }

        record Slot(String hash, MultimodalRequest request,
                    CompletableFuture<MultimodalResponse> future) {}
    }
}
