package tech.kayys.gollek.multimodal;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.model.MultimodalRequest;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

/**
 * Adaptive load balancer with real-time provider scoring.
 *
 * <p>
 * Replaces the simple first-match routing in
 * {@link tech.kayys.gollek.core.multimodal.router.MultimodalRouter}
 * with a score-based selection algorithm that continuously learns from
 * observed latency, error rates, and cost.
 *
 * <h3>Scoring model</h3>
 * 
 * <pre>
 *   score(provider) = w_latency × (1 / p50_latency_ms)
 *                   + w_error   × (1 - error_rate)
 *                   + w_cost    × (1 / cost_per_token)
 *                   + w_load    × (1 - current_load)
 * </pre>
 *
 * <p>
 * Weights are configurable. Default: latency=0.5, errors=0.3, cost=0.1,
 * load=0.1.
 *
 * <h3>Strategies</h3>
 * <ul>
 * <li>{@code BEST_SCORE} — always pick the highest-scoring available
 * provider</li>
 * <li>{@code LEAST_LATENCY} — pick the provider with lowest observed P50</li>
 * <li>{@code ROUND_ROBIN} — distribute evenly (ignores latency scores)</li>
 * <li>{@code RANDOM} — weighted random; explores under-sampled providers</li>
 * <li>{@code COST_OPTIMISED}— minimise token cost (good for batch
 * workloads)</li>
 * </ul>
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.lb.strategy=BEST_SCORE
 *   gollek.lb.weight.latency=0.5
 *   gollek.lb.weight.error=0.3
 *   gollek.lb.weight.cost=0.1
 *   gollek.lb.weight.load=0.1
 *   gollek.lb.exploration-rate=0.05   # epsilon-greedy exploration
 * </pre>
 */
@ApplicationScoped
public class AdaptiveLoadBalancer {

    private static final Logger LOG = Logger.getLogger(AdaptiveLoadBalancer.class);

    public enum Strategy {
        BEST_SCORE, LEAST_LATENCY, ROUND_ROBIN, RANDOM, COST_OPTIMISED
    }

    @ConfigProperty(name = "gollek.lb.strategy", defaultValue = "BEST_SCORE")
    String strategyName;

    @ConfigProperty(name = "gollek.lb.weight.latency", defaultValue = "0.5")
    double wLatency;

    @ConfigProperty(name = "gollek.lb.weight.error", defaultValue = "0.3")
    double wError;

    @ConfigProperty(name = "gollek.lb.weight.cost", defaultValue = "0.1")
    double wCost;

    @ConfigProperty(name = "gollek.lb.weight.load", defaultValue = "0.1")
    double wLoad;

    @ConfigProperty(name = "gollek.lb.exploration-rate", defaultValue = "0.05")
    double explorationRate;

    @Inject
    MultimodalCapabilityRegistry registry;

    private final ConcurrentHashMap<String, ProviderStats> stats = new ConcurrentHashMap<>();
    private final AtomicInteger rrCounter = new AtomicInteger(0);
    private final Random rng = new Random();

    // -------------------------------------------------------------------------
    // Provider selection
    // -------------------------------------------------------------------------

    /**
     * Select the best provider for a request given the active strategy.
     *
     * @param request the incoming request
     * @return the selected provider, or empty if none available
     */
    public Optional<MultimodalInferenceProvider> select(MultimodalRequest request) {
        List<MultimodalInferenceProvider> candidates = registry.findCapable(request.inputModalities(),
                request.getOutputConfig() != null ? Arrays.asList(request.getOutputConfig().getOutputModalities())
                        : null);

        if (candidates.isEmpty())
            return Optional.empty();
        if (candidates.size() == 1)
            return Optional.of(candidates.get(0));

        // Epsilon-greedy exploration: occasionally pick randomly to gather data
        if (rng.nextDouble() < explorationRate) {
            MultimodalInferenceProvider chosen = candidates.get(rng.nextInt(candidates.size()));
            LOG.debugf("[LB-EXPLORE] Exploring provider %s", chosen.providerId());
            return Optional.of(chosen);
        }

        Strategy strategy;
        try {
            strategy = Strategy.valueOf(strategyName.toUpperCase());
        } catch (Exception e) {
            strategy = Strategy.BEST_SCORE;
        }

        return switch (strategy) {
            case BEST_SCORE -> selectByScore(candidates);
            case LEAST_LATENCY -> selectByLatency(candidates);
            case ROUND_ROBIN -> selectRoundRobin(candidates);
            case RANDOM -> Optional.of(candidates.get(rng.nextInt(candidates.size())));
            case COST_OPTIMISED -> selectByCost(candidates);
        };
    }

    // -------------------------------------------------------------------------
    // Strategy implementations
    // -------------------------------------------------------------------------

    private Optional<MultimodalInferenceProvider> selectByScore(
            List<MultimodalInferenceProvider> candidates) {
        return candidates.stream()
                .max(Comparator.comparingDouble(p -> score(statsFor(p.providerId()))));
    }

    private Optional<MultimodalInferenceProvider> selectByLatency(
            List<MultimodalInferenceProvider> candidates) {
        return candidates.stream()
                .min(Comparator.comparingDouble(p -> statsFor(p.providerId()).p50Ms()));
    }

    private Optional<MultimodalInferenceProvider> selectRoundRobin(
            List<MultimodalInferenceProvider> candidates) {
        int idx = Math.abs(rrCounter.getAndIncrement()) % candidates.size();
        return Optional.of(candidates.get(idx));
    }

    private Optional<MultimodalInferenceProvider> selectByCost(
            List<MultimodalInferenceProvider> candidates) {
        return candidates.stream()
                .min(Comparator.comparingDouble(p -> statsFor(p.providerId()).costPerToken()));
    }

    // -------------------------------------------------------------------------
    // Feedback loop — called by the router after each completed request
    // -------------------------------------------------------------------------

    /**
     * Record a completed request outcome.
     * This data feeds the scoring model for future selection decisions.
     */
    public void recordOutcome(String providerId, long latencyMs,
            boolean success, long inputTokens, long outputTokens) {
        ProviderStats s = stats.computeIfAbsent(providerId, ProviderStats::new);
        s.record(latencyMs, success, inputTokens + outputTokens);
        LOG.tracef("[LB-FEEDBACK] %s latency=%dms success=%s", providerId, latencyMs, success);
    }

    // -------------------------------------------------------------------------
    // Scoring
    // -------------------------------------------------------------------------

    private double score(ProviderStats s) {
        double latencyScore = s.p50Ms() > 0 ? 1.0 / s.p50Ms() : 1.0;
        double errorScore = 1.0 - s.errorRate();
        double costScore = s.costPerToken() > 0 ? 1.0 / s.costPerToken() : 1.0;
        double loadScore = 1.0 - Math.min(s.currentLoad(), 1.0);

        return wLatency * latencyScore
                + wError * errorScore
                + wCost * costScore
                + wLoad * loadScore;
    }

    private ProviderStats statsFor(String id) {
        return stats.computeIfAbsent(id, ProviderStats::new);
    }

    /** Returns per-provider scores for dashboard display. */
    public Map<String, Object> scoreboard() {
        Map<String, Object> result = new LinkedHashMap<>();
        stats.forEach((id, s) -> result.put(id, Map.of(
                "score", String.format("%.4f", score(s)),
                "p50Ms", (long) s.p50Ms(),
                "p99Ms", (long) s.p99Ms(),
                "errorRate", String.format("%.1f%%", s.errorRate() * 100),
                "totalCalls", s.totalCalls.get(),
                "costPerToken", s.costPerToken())));
        return result;
    }

    // =========================================================================
    // Provider statistics (exponential moving average)
    // =========================================================================

    static final class ProviderStats {
        private final String id;
        // EMA latency (α=0.1 → smooth, reactive)
        private static final double ALPHA = 0.1;
        private volatile double p50Ema = 500.0;
        private volatile double p99Ema = 2000.0;
        private final AtomicLong totalCalls = new AtomicLong();
        private final AtomicLong errorCount = new AtomicLong();
        private final AtomicLong activeReqs = new AtomicLong();
        private volatile double costPerToken = 0.0;

        // Ring buffer for P99 (last 200 samples)
        private static final int RING = 200;
        private final long[] ring = new long[RING];
        private final AtomicInteger ringIdx = new AtomicInteger(0);

        ProviderStats(String id) {
            this.id = id;
        }

        void record(long latencyMs, boolean success, long tokens) {
            // EMA update
            p50Ema = ALPHA * latencyMs + (1 - ALPHA) * p50Ema;
            ring[ringIdx.getAndIncrement() % RING] = latencyMs;

            // P99
            long[] copy = ring.clone();
            Arrays.sort(copy);
            p99Ema = ALPHA * copy[(int) (RING * 0.99)] + (1 - ALPHA) * p99Ema;

            totalCalls.incrementAndGet();
            if (!success)
                errorCount.incrementAndGet();
        }

        double p50Ms() {
            return p50Ema;
        }

        double p99Ms() {
            return p99Ema;
        }

        double errorRate() {
            long total = totalCalls.get();
            return total == 0 ? 0 : errorCount.get() / (double) total;
        }

        double currentLoad() {
            long total = totalCalls.get();
            return total == 0 ? 0 : Math.min(activeReqs.get() / 10.0, 1.0);
        }

        double costPerToken() {
            return costPerToken;
        }

        void setCostPerToken(double cost) {
            this.costPerToken = cost;
        }

        void incActive() {
            activeReqs.incrementAndGet();
        }

        void decActive() {
            activeReqs.decrementAndGet();
        }
    }
}
