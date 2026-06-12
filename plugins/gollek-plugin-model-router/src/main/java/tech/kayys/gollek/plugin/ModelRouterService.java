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
}