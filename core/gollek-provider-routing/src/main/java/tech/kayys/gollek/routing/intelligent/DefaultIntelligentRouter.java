/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.routing.intelligent;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.IntelligentRouter;
import tech.kayys.gollek.spi.routing.RoutingDecision;
import tech.kayys.gollek.spi.routing.RoutingPlan;
import tech.kayys.gollek.spi.routing.SelectionStrategy;
import tech.kayys.gollek.spi.runtime.ExecutionProvider;
import tech.kayys.gollek.spi.runtime.KvCacheLocation;
import tech.kayys.gollek.spi.runtime.KvCacheRegistry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Default implementation of the Intelligent Router Engine.
 *
 * <p>Discovers all available {@link ExecutionProvider} instances,
 * filters out the incapable or unhealthy ones, and scores the remainder
 * to produce the optimal {@link RoutingDecision}.</p>
 */
@ApplicationScoped
public class DefaultIntelligentRouter implements IntelligentRouter {

    private final Instance<ExecutionProvider> providers;
    private final ProviderScorer scorer;
    private final KvCacheRegistry kvRegistry;

    private static final int SPLIT_THRESHOLD_TOKENS = 512;

    @Inject
    public DefaultIntelligentRouter(
            Instance<ExecutionProvider> providers,
            ProviderScorer scorer,
            KvCacheRegistry kvRegistry
    ) {
        this.providers = providers;
        this.scorer = scorer;
        this.kvRegistry = kvRegistry;
    }

    @Override
    public RoutingDecision route(RoutingContext context) {
        // 1. KV-Cache Aware Routing (Shortcut)
        String prefixHash = context.request().prefixHash();
        Optional<KvCacheLocation> cache = kvRegistry.find(prefixHash);
        
        if (cache.isPresent()) {
            String cachedProviderId = cache.get().providerId();
            // Verify cached provider is available for use
            boolean available = providers.stream()
                    .filter(p -> p.id().equals(cachedProviderId))
                    .filter(p -> !context.isExcluded(p.id()))
                    .filter(p -> p.health() != null && p.health().isHealthy())
                    .anyMatch(p -> CapabilityMatcher.supports(p.capabilities(), context.requiredCapabilities()));
            
            if (available) {
                return RoutingDecision.builder()
                        .selectedProviderId(cachedProviderId)
                        .strategyUsed(SelectionStrategy.ROUND_ROBIN) // Mark as cache-hit strategy
                        .score(0) // Best possible score
                        .build();
            }
        }

        // 2. Split Routing for Long Prompts
        if (context.request().getPromptTokenCount() > SPLIT_THRESHOLD_TOKENS) {
            List<ExecutionProvider> candidates = providers.stream()
                    .filter(p -> !context.isExcluded(p.id()))
                    .filter(p -> p.health() != null && p.health().isHealthy())
                    .filter(p -> CapabilityMatcher.supports(p.capabilities(), context.requiredCapabilities()))
                    .toList();

            if (candidates.size() >= 2) {
                // Find fastest for prefill
                ExecutionProvider prefill = candidates.stream()
                        .min(Comparator.comparingDouble(p -> scorer.score(p, context))) // simplified scoring
                        .get();
                
                // Find cheapest for decode
                ExecutionProvider decode = candidates.stream()
                        .filter(p -> !p.id().equals(prefill.id()))
                        .min(Comparator.comparingDouble(p -> p.costProfile().costPer1KTokens()))
                        .get();

                return RoutingDecision.builder()
                        .selectedProviderId(prefill.id()) // Primary is prefill
                        .plan(new RoutingPlan.Split(prefill.id(), decode.id()))
                        .strategyUsed(SelectionStrategy.SCORED)
                        .score(100)
                        .requiresKVMigration(true)
                        .build();
            }
        }

        // 3. Speculative Multi-Provider Decoding
        // Example: If target is a large model, find an injection-compatible draft
        List<ExecutionProvider> candidates = providers.stream()
                .filter(p -> !context.isExcluded(p.id()))
                .filter(p -> p.health() != null && p.health().isHealthy())
                .filter(p -> CapabilityMatcher.supports(p.capabilities(), context.requiredCapabilities()))
                .toList();

        if (candidates.size() >= 2) {
            String targetModel = context.request().getModel();
            // Simple heuristic to identify a "target" model (7B+, 70B, etc.)
            boolean isLargeModel = targetModel.contains("7B") || targetModel.contains("70B") || targetModel.contains("instruct");

            if (isLargeModel) {
                Optional<ExecutionProvider> draft = candidates.stream()
                        .filter(p -> !p.id().equals(targetModel))
                        .filter(p -> p.id().contains("0.5B") || p.id().contains("1.5B") || p.id().contains("tiny"))
                        .findFirst();

                if (draft.isPresent()) {
                    ExecutionProvider target = candidates.stream()
                            .filter(p -> p.id().contains(targetModel) || p.id().equals(targetModel))
                            .findFirst()
                            .orElse(null);

                    if (target != null) {
                        return RoutingDecision.builder()
                                .selectedProviderId(target.id())
                                .plan(new RoutingPlan.Speculative(draft.get().id(), target.id()))
                                .strategyUsed(SelectionStrategy.SCORED)
                                .score(50) // High-priority performance boost
                                .requiresKVMigration(true)
                                .build();
                    }
                }
            }
        }

        // 4. Normal Scored Routing
        return providers.stream()
                // 1. Filter out intentionally excluded providers
                .filter(p -> !context.isExcluded(p.id()))
                // 2. Filter out unhealthy providers
                .filter(p -> p.health() != null && p.health().isHealthy())
                // 3. Filter by required capabilities (Capability Matching)
                .filter(p -> CapabilityMatcher.supports(p.capabilities(), context.requiredCapabilities()))
                // 4. Calculate score for the remainder
                .map(p -> new ScoredProvider(p, scorer.score(p, context)))
                // 5. Sort by best score (lowest is best)
                .min(Comparator.comparingDouble(ScoredProvider::score))
                // 6. Map to decision
                .map(sp -> RoutingDecision.builder()
                        .selectedProviderId(sp.provider.id())
                        .plan(new RoutingPlan.Single(sp.provider.id()))
                        .strategyUsed(context.getEffectiveStrategy(SelectionStrategy.SCORED))
                        .score((int) (sp.score * 100)) // scale for integer representation
                        .build())
                .orElseThrow(() -> new IllegalStateException(
                        "No execution provider found satisfying capabilities and health requirements for request " + context.request().getRequestId()));
    }

    private record ScoredProvider(ExecutionProvider provider, double score) {}
}
