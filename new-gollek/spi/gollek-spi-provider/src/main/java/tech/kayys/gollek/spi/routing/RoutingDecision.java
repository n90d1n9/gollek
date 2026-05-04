package tech.kayys.gollek.spi.routing;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of a routing decision.
 * Contains the selected provider and decision metadata.
 */
public record RoutingDecision(
        String selectedProviderId,
        RoutingPlan plan,
        String poolId,
        SelectionStrategy strategyUsed,
        int score,
        List<String> fallbackProviders,
        Map<String, Object> metadata,
        boolean requiresKVMigration,
        Instant timestamp) {

    public RoutingDecision {
        if (plan == null) {
            plan = new RoutingPlan.Single(selectedProviderId);
        }
        fallbackProviders = fallbackProviders != null 
            ? List.copyOf(fallbackProviders) 
            : Collections.emptyList();
        metadata = metadata != null 
            ? Map.copyOf(metadata) 
            : Collections.emptyMap();
        if (timestamp == null) {
            timestamp = Instant.now();
        }
    }

    /**
     * Create a simple decision without fallbacks
     */
    public static RoutingDecision of(String providerId, SelectionStrategy strategy) {
        return new RoutingDecision(
            providerId,
            new RoutingPlan.Single(providerId),
            null,
            strategy,
            100,
            Collections.emptyList(),
            Collections.emptyMap(),
            false,
            Instant.now()
        );
    }

    /**
     * Create a decision with fallback providers
     */
    public static RoutingDecision withFallbacks(
            String providerId,
            SelectionStrategy strategy,
            List<String> fallbacks) {
        return new RoutingDecision(
            providerId,
            new RoutingPlan.Single(providerId),
            null,
            strategy,
            100,
            fallbacks,
            Collections.emptyMap(),
            false,
            Instant.now()
        );
    }

    /**
     * Check if fallback providers are available
     */
    public boolean hasFallbacks() {
        return !fallbackProviders.isEmpty();
    }

    /**
     * Get next fallback provider
     */
    public String getNextFallback() {
        return fallbackProviders.isEmpty() ? null : fallbackProviders.get(0);
    }

    /**
     * Builder for complex decisions
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String selectedProviderId;
        private RoutingPlan plan;
        private String poolId;
        private SelectionStrategy strategyUsed;
        private int score = 100;
        private List<String> fallbackProviders = Collections.emptyList();
        private Map<String, Object> metadata = Collections.emptyMap();
        private boolean requiresKVMigration = false;
        private Instant timestamp = Instant.now();

        public Builder selectedProviderId(String id) {
            this.selectedProviderId = id;
            return this;
        }

        public Builder plan(RoutingPlan plan) {
            this.plan = plan;
            return this;
        }

        public Builder poolId(String poolId) {
            this.poolId = poolId;
            return this;
        }

        public Builder strategyUsed(SelectionStrategy strategy) {
            this.strategyUsed = strategy;
            return this;
        }

        public Builder score(int score) {
            this.score = score;
            return this;
        }

        public Builder fallbackProviders(List<String> fallbacks) {
            this.fallbackProviders = fallbacks;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder requiresKVMigration(boolean requiresKVMigration) {
            this.requiresKVMigration = requiresKVMigration;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public RoutingDecision build() {
            return new RoutingDecision(
                selectedProviderId, plan, poolId, strategyUsed, score,
                fallbackProviders, metadata, requiresKVMigration, timestamp
            );
        }
    }
}
