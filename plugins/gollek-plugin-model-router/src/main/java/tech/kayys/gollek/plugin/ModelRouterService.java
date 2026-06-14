package tech.kayys.gollek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;

import java.util.List;
import java.util.Map;

/**
 * Service for intelligent model-to-provider routing decisions.
 * Selects the optimal provider for a given model based on various factors.
 */
@Default
@ApplicationScoped
public interface ModelRouterService {

    /**
     * Select the best provider for a given model and tenant
     */
    String selectProvider(String modelId, String requestId);

    /**
     * Select the best provider for a given model, tenant, and request context
     */
    String selectProvider(String modelId, String requestId, Map<String, Object> requestContext);

    /**
     * Get available providers for a model
     */
    List<String> getAvailableProviders(String modelId);

    /**
     * Select a compatible draft model provider for speculative decoding alongside the target model provider.
     * @return Array where index 0 is draft provider ID, index 1 is target provider ID.
     */
    String[] selectDraftAndTargetProviders(String modelId, String requestId);

    /**
     * Get routing score for a provider for a specific model
     */
    double getProviderScore(String modelId, String providerId, String requestId);

    /**
     * Get routing decision details
     */
    RoutingDecision getRoutingDecision(String modelId, String requestId, Map<String, Object> context);

    // -----------------------------------------------------------------------
    // Multi-Cluster Federation (5.2)
    // -----------------------------------------------------------------------

    /**
     * Selects the best provider for a model with region affinity.
     *
     * <p>Prefers providers in {@code preferredRegion} when multiple candidates
     * exist, falling back to the best provider globally if no regional match
     * is available.
     *
     * @param modelId         the model to serve
     * @param requestId       request/tenant identifier
     * @param preferredRegion AWS/GCP/Azure region code (e.g. "us-east-1")
     * @return selected provider ID
     */
    default String selectProviderForRegion(String modelId, String requestId, String preferredRegion) {
        // Default: fall through to basic selection (override for region awareness)
        return selectProvider(modelId, requestId);
    }

    /**
     * Returns the failover provider ID for a given provider, using the
     * {@link tech.kayys.gollek.spi.provider.ProviderHealth#failoverTarget()}
     * field if the primary is unhealthy.
     *
     * @param primaryProviderId  the provider that is currently unhealthy
     * @param modelId            model being routed
     * @return failover provider ID, or null if none is configured
     */
    default String getFailoverProvider(String primaryProviderId, String modelId) {
        return null; // Override when health registry is available
    }

    /**
     * Returns a list of provider IDs in the given cluster.
     * Useful for cluster-local routing in federated deployments.
     *
     * @param clusterId logical cluster identifier (e.g. "k8s-prod-us-east")
     * @return provider IDs belonging to that cluster
     */
    default List<String> getProvidersInCluster(String clusterId) {
        return List.of();
    }
}