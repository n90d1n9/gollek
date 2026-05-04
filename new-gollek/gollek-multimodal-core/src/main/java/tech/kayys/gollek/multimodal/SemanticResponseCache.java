package tech.kayys.gollek.multimodal;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.embedding.EmbeddingService;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.model.ModalityType;
import java.util.HexFormat;

import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Two-level response cache for multimodal inference.
 *
 * <h3>Level 1 — Exact hash cache</h3>
 * <p>SHA-256 of the full serialised request (model + all parts).
 * Zero-latency lookup; perfect for repeated identical requests
 * (e.g. batch jobs, A/B test harnesses, API clients that retry).
 *
 * <h3>Level 2 — Semantic similarity cache</h3>
 * <p>Embeds text parts, then does approximate nearest-neighbour
 * search across cached embeddings.  If the cosine similarity of a
 * new request exceeds {@link #semanticThreshold}, the cached response
 * is returned instead of calling the provider.
 *
 * <p>Semantic cache eliminates redundant LLM calls for paraphrased
 * queries ("Summarise X" vs "Give me a summary of X") which are
 * otherwise cache misses at L1.
 *
 * <h3>Performance characteristics</h3>
 * <ul>
 *   <li>L1 hit: ~50 µs (ConcurrentHashMap + SHA-256)</li>
 *   <li>L2 hit: ~5 ms (embedding call + linear scan over cached vectors)</li>
 *   <li>Miss: full provider latency (100 ms – 30 s)</li>
 * </ul>
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.cache.enabled=true
 *   gollek.cache.l1.max-size=10000
 *   gollek.cache.l1.ttl-secs=3600
 *   gollek.cache.l2.enabled=true
 *   gollek.cache.l2.threshold=0.94
 *   gollek.cache.l2.max-size=2000
 *   gollek.cache.l2.embedding-model=openai-embedding
 * </pre>
 */
@ApplicationScoped
public class SemanticResponseCache {

    private static final Logger LOG = Logger.getLogger(SemanticResponseCache.class);

    @ConfigProperty(name = "gollek.cache.enabled",          defaultValue = "true")
    boolean enabled;

    @ConfigProperty(name = "gollek.cache.l1.max-size",      defaultValue = "10000")
    int l1MaxSize;

    @ConfigProperty(name = "gollek.cache.l1.ttl-secs",      defaultValue = "3600")
    long l1TtlSecs;

    @ConfigProperty(name = "gollek.cache.l2.enabled",       defaultValue = "true")
    boolean l2Enabled;

    @ConfigProperty(name = "gollek.cache.l2.threshold",     defaultValue = "0.94")
    double semanticThreshold;

    @ConfigProperty(name = "gollek.cache.l2.max-size",      defaultValue = "2000")
    int l2MaxSize;

    @ConfigProperty(name = "gollek.cache.l2.embedding-model", defaultValue = "openai-embedding")
    String embeddingModel;

    @Inject jakarta.enterprise.inject.Instance<EmbeddingService> embeddingService;

    // L1: SHA-256 → CacheEntry
    private final Map<String, CacheEntry> l1 = new ConcurrentHashMap<>();

    // L2: list of (vector, responseRef) pairs
    private final List<SemanticEntry> l2 = Collections.synchronizedList(new ArrayList<>());

    // Stats
    private final LongAdder l1Hits   = new LongAdder();
    private final LongAdder l2Hits   = new LongAdder();
    private final LongAdder misses   = new LongAdder();
    private final LongAdder stores   = new LongAdder();

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------

    /**
     * Attempts to find a cached response for the given request.
     * Returns empty Optional on cache miss.
     */
    public Uni<Optional<MultimodalResponse>> lookup(MultimodalRequest request) {
        if (!enabled) return Uni.createFrom().item(Optional.empty());

        // L1 exact lookup (synchronous, zero-cost)
        String hash = hash(request);
        CacheEntry l1Entry = l1.get(hash);
        if (l1Entry != null && !l1Entry.isExpired()) {
            l1Hits.increment();
            LOG.debugf("[CACHE-L1-HIT] %s", request.getRequestId());
            return Uni.createFrom().item(Optional.of(l1Entry.response()));
        }
        if (l1Entry != null) l1.remove(hash); // evict expired

        // Only attempt L2 for text-only or text+image requests (need embeddable text)
        if (!l2Enabled || !hasEmbeddableText(request)) {
            misses.increment();
            return Uni.createFrom().item(Optional.empty());
        }

        // L2 semantic lookup (requires embedding call)
        return embedRequest(request)
                .map(queryVec -> {
                    if (queryVec == null) {
                        misses.increment();
                        return Optional.<MultimodalResponse>empty();
                    }
                    Optional<MultimodalResponse> hit = nearestNeighbour(queryVec);
                    if (hit.isPresent()) {
                        l2Hits.increment();
                        LOG.debugf("[CACHE-L2-HIT] similarity>=%.2f for %s",
                                semanticThreshold, request.getRequestId());
                    } else {
                        misses.increment();
                    }
                    return hit;
                });
    }

    // -------------------------------------------------------------------------
    // Store
    // -------------------------------------------------------------------------

    /**
     * Stores a response in both cache levels.
     * Should be called asynchronously (fire-and-forget) after a provider call.
     */
    public Uni<Void> store(MultimodalRequest request, MultimodalResponse response) {
        if (!enabled || response.getStatus() == MultimodalResponse.ResponseStatus.ERROR) {
            return Uni.createFrom().voidItem();
        }

        // L1 store
        String hash = hash(request);
        if (l1.size() >= l1MaxSize) evictOldestL1();
        l1.put(hash, new CacheEntry(response, Instant.now().plusSeconds(l1TtlSecs)));
        stores.increment();

        // L2 store (async, non-blocking)
        if (!l2Enabled || !hasEmbeddableText(request)) {
            return Uni.createFrom().voidItem();
        }

        return embedRequest(request)
                .invoke(vec -> {
                    if (vec != null) {
                        if (l2.size() >= l2MaxSize) l2.remove(0); // FIFO eviction
                        l2.add(new SemanticEntry(vec, response,
                                Instant.now().plusSeconds(l1TtlSecs)));
                    }
                })
                .replaceWithVoid();
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    public CacheStats stats() {
        long total = l1Hits.sum() + l2Hits.sum() + misses.sum();
        double hitRate = total == 0 ? 0 : (l1Hits.sum() + l2Hits.sum()) / (double) total;
        return new CacheStats(l1Hits.sum(), l2Hits.sum(), misses.sum(),
                stores.sum(), l1.size(), l2.size(), hitRate);
    }

    public void invalidate(String requestHash) { l1.remove(requestHash); }
    public void clear() { l1.clear(); l2.clear(); }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private String hash(MultimodalRequest request) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(request.getModel().getBytes());
            for (var part : request.getInputs()) {
                md.update(part.getModality().name().getBytes());
                if (part.getText() != null)       md.update(part.getText().getBytes());
                if (part.getBase64Data() != null) md.update(part.getBase64Data().getBytes());
                if (part.getUri() != null)        md.update(part.getUri().getBytes());
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            return UUID.randomUUID().toString();
        }
    }

    private boolean hasEmbeddableText(MultimodalRequest req) {
        return Arrays.stream(req.getInputs()).anyMatch(p ->
                p.getModality() == ModalityType.TEXT
                && p.getText() != null && !p.getText().isBlank());
    }

    private Uni<float[]> embedRequest(MultimodalRequest request) {
        String combined = Arrays.stream(request.getInputs())
                .filter(p -> p.getModality() == ModalityType.TEXT && p.getText() != null)
                .map(MultimodalContent::getText)
                .reduce("", (a, b) -> a + " " + b)
                .trim();
        if (combined.isBlank()) return Uni.createFrom().nullItem();

        if (embeddingService.isUnsatisfied()) return Uni.createFrom().nullItem();

        return embeddingService.get().embed(MultimodalContent.ofText(combined), embeddingModel)
                .map(result -> result.hasError() ? null : result.vector());
    }

    private Optional<MultimodalResponse> nearestNeighbour(float[] queryVec) {
        double bestSim  = semanticThreshold;
        MultimodalResponse best = null;
        Instant now = Instant.now();

        for (SemanticEntry entry : l2) {
            if (entry.expiresAt().isBefore(now)) continue;
            double sim = cosineSimilarity(queryVec, entry.vector());
            if (sim > bestSim) { bestSim = sim; best = entry.response(); }
        }
        return Optional.ofNullable(best);
    }

    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) return 0.0;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private void evictOldestL1() {
        Instant cutoff = Instant.now();
        l1.entrySet().removeIf(e -> e.getValue().expiresAt().isBefore(cutoff));
        if (l1.size() >= l1MaxSize) {
            // Force-remove oldest 10%
            int toRemove = l1MaxSize / 10;
            l1.entrySet().stream().limit(toRemove)
               .map(Map.Entry::getKey).toList().forEach(l1::remove);
        }
    }

    // -------------------------------------------------------------------------
    // Records
    // -------------------------------------------------------------------------

    private record CacheEntry(MultimodalResponse response, Instant expiresAt) {
        boolean isExpired() { return Instant.now().isAfter(expiresAt); }
    }

    private record SemanticEntry(float[] vector, MultimodalResponse response, Instant expiresAt) {}

    public record CacheStats(long l1Hits, long l2Hits, long misses, long stores,
                              int l1Size, int l2Size, double hitRate) {}
}
