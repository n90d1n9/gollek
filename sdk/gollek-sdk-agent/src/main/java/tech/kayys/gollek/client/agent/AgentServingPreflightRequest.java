package tech.kayys.gollek.client.agent;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Configuration for a serving-only agent preflight.
 *
 * <p>The defaults match the full agent-serving preflight: discover MCP tools,
 * validate tool definitions, validate the request, and require all checked
 * surfaces to respect Gollek's serving boundary.
 */
public record AgentServingPreflightRequest(
        String modelId,
        String surface,
        Map<String, Object> request,
        AgentRequestOptions requestOptions,
        boolean discoverMcpTools,
        boolean mcpDiscoveryRequired,
        boolean validateTools,
        boolean toolValidationRequired,
        boolean validateRequest,
        boolean requestValidationRequired,
        boolean openAiToolCompatibility,
        boolean enabledOnly,
        String featureProfile,
        String requiredContractVersion,
        List<String> requiredFeatures,
        List<String> optionalFeatures) {

    public AgentServingPreflightRequest {
        modelId = text(modelId);
        surface = AgentServingRoute.normalizeSurface(surface);
        request = request == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(request));
        requestOptions = requestOptions == null ? AgentRequestOptions.empty() : requestOptions;
        requiredContractVersion = requiredContractVersion == null || requiredContractVersion.isBlank()
                ? AgentServingFeatureCatalog.CONTRACT_VERSION
                : requiredContractVersion.trim();
        featureProfile = AgentServingFeatureProfile.normalizeName(featureProfile);
        Optional<AgentServingFeatureProfile> profile = AgentServingFeatureProfile.find(featureProfile);
        requiredFeatures = requiredFeatures == null || requiredFeatures.isEmpty()
                ? profile.map(AgentServingFeatureProfile::requiredFeatures)
                        .orElse(AgentServingFeatureCatalog.REQUIRED_FEATURES)
                : List.copyOf(requiredFeatures);
        optionalFeatures = optionalFeatures == null || optionalFeatures.isEmpty()
                ? profile.map(AgentServingFeatureProfile::optionalFeatures).orElse(List.of())
                : List.copyOf(optionalFeatures);
    }

    public static AgentServingPreflightRequest of(String modelId, String surface, Map<String, Object> request) {
        return builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .build();
    }

    public AgentServingRoute route() {
        return AgentServingRoute.of(surface, modelId, featureProfile);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String modelId;
        private String surface;
        private Map<String, Object> request = Map.of();
        private AgentRequestOptions requestOptions = AgentRequestOptions.empty();
        private boolean discoverMcpTools = true;
        private boolean mcpDiscoveryRequired = true;
        private boolean validateTools = true;
        private boolean toolValidationRequired = true;
        private boolean validateRequest = true;
        private boolean requestValidationRequired = true;
        private boolean openAiToolCompatibility = true;
        private boolean enabledOnly = true;
        private String featureProfile = AgentServingFeatureProfile.DEFAULT_PROFILE;
        private String requiredContractVersion = AgentServingFeatureCatalog.CONTRACT_VERSION;
        private List<String> requiredFeatures = AgentServingFeatureProfile.defaultProfile().requiredFeatures();
        private List<String> optionalFeatures = AgentServingFeatureProfile.defaultProfile().optionalFeatures();

        private Builder() {
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder surface(String surface) {
            this.surface = AgentServingRoute.normalizeSurface(surface);
            return this;
        }

        public Builder route(AgentServingRoute route) {
            if (route == null) {
                return this;
            }
            modelId(route.model());
            surface(route.surface());
            if (route.featureProfile() != null) {
                featureProfile(route.featureProfile());
            }
            return this;
        }

        public Builder request(Map<String, Object> request) {
            this.request = request == null ? Map.of() : new LinkedHashMap<>(request);
            return this;
        }

        public Builder requestOptions(AgentRequestOptions requestOptions) {
            this.requestOptions = requestOptions == null ? AgentRequestOptions.empty() : requestOptions;
            return this;
        }

        public Builder discoverMcpTools(boolean discoverMcpTools) {
            this.discoverMcpTools = discoverMcpTools;
            return this;
        }

        public Builder mcpDiscoveryRequired(boolean mcpDiscoveryRequired) {
            this.mcpDiscoveryRequired = mcpDiscoveryRequired;
            return this;
        }

        public Builder validateTools(boolean validateTools) {
            this.validateTools = validateTools;
            return this;
        }

        public Builder toolValidationRequired(boolean toolValidationRequired) {
            this.toolValidationRequired = toolValidationRequired;
            return this;
        }

        public Builder validateRequest(boolean validateRequest) {
            this.validateRequest = validateRequest;
            return this;
        }

        public Builder requestValidationRequired(boolean requestValidationRequired) {
            this.requestValidationRequired = requestValidationRequired;
            return this;
        }

        public Builder openAiToolCompatibility(boolean openAiToolCompatibility) {
            this.openAiToolCompatibility = openAiToolCompatibility;
            return this;
        }

        public Builder enabledOnly(boolean enabledOnly) {
            this.enabledOnly = enabledOnly;
            return this;
        }

        public Builder featureProfile(String featureProfile) {
            this.featureProfile = AgentServingFeatureProfile.normalizeName(featureProfile);
            AgentServingFeatureProfile.find(this.featureProfile).ifPresent(profile -> {
                this.requiredFeatures = profile.requiredFeatures();
                this.optionalFeatures = profile.optionalFeatures();
            });
            return this;
        }

        public Builder requiredContractVersion(String requiredContractVersion) {
            this.requiredContractVersion = requiredContractVersion;
            return this;
        }

        public Builder requiredFeatures(List<String> requiredFeatures) {
            this.requiredFeatures = requiredFeatures == null ? List.of() : List.copyOf(requiredFeatures);
            return this;
        }

        public Builder optionalFeatures(List<String> optionalFeatures) {
            this.optionalFeatures = optionalFeatures == null ? List.of() : List.copyOf(optionalFeatures);
            return this;
        }

        public AgentServingPreflightRequest build() {
            return new AgentServingPreflightRequest(
                    modelId,
                    surface,
                    request,
                    requestOptions,
                    discoverMcpTools,
                    mcpDiscoveryRequired,
                    validateTools,
                    toolValidationRequired,
                    validateRequest,
                    requestValidationRequired,
                    openAiToolCompatibility,
                    enabledOnly,
                    featureProfile,
                    requiredContractVersion,
                    requiredFeatures,
                    optionalFeatures);
        }
    }

    private static String text(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }
}
