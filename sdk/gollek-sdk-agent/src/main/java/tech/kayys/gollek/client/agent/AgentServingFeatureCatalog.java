package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stable agent-serving contract metadata shared by Gollek API mappers and SDK
 * clients.
 */
public final class AgentServingFeatureCatalog {
    public static final String CAPABILITIES_OBJECT = "gollek.agent_capabilities";
    public static final String CONTRACT_OBJECT = "gollek.agent_contract";
    public static final String CONTRACT_VERSION = "v1";
    public static final String SERVICE_ROLE_INFERENCE_SERVING_ENGINE = "inference_serving_engine";
    public static final String FEATURE_NAMESPACE = "gollek.agent.compatibility";

    public static final List<String> SUPPORTED_CONTRACT_VERSIONS = List.of(CONTRACT_VERSION);

    public static final List<String> REQUIRED_FEATURES = List.of(
            "openai_chat_completions",
            "openai_chat_streaming",
            "openai_responses",
            "openai_responses_streaming",
            "openai_embeddings",
            "model_capability_matrix",
            "agent_capabilities",
            "agent_contract",
            "agent_feature_negotiation",
            "agent_readiness_issue_catalog",
            "agent_preflight",
            "agent_request_validation",
            "agent_tool_contract_validation",
            "request_trace_context",
            "mcp_tool_discovery",
            "rag_context");

    public static final List<String> OPTIONAL_FEATURES = List.of(
            "openai_models",
            "stream_options",
            "stream_usage_reporting",
            "stream_trace_metadata",
            "native_gollek_completions",
            "native_gollek_streaming",
            "tools",
            "tool_choice",
            "tool_calls",
            "system_prompt",
            "mcp_registry",
            "mcp_tool_definitions",
            "embeddings",
            "sessions",
            "models",
            "providers");

    private AgentServingFeatureCatalog() {
    }

    public static List<String> compatibilityFeatures() {
        return concat(REQUIRED_FEATURES, OPTIONAL_FEATURES);
    }

    public static Map<String, Object> featureNegotiation() {
        Map<String, Object> negotiation = new LinkedHashMap<>();
        negotiation.put("mode", "feature_flags");
        negotiation.put("feature_namespace", FEATURE_NAMESPACE);
        negotiation.put("contract_version", CONTRACT_VERSION);
        negotiation.put("minimum_client_contract_version", CONTRACT_VERSION);
        negotiation.put("supported_contract_versions", SUPPORTED_CONTRACT_VERSIONS);
        negotiation.put("default_feature_profile", AgentServingFeatureProfile.DEFAULT_PROFILE);
        negotiation.put("supported_feature_profiles", AgentServingFeatureProfile.supportedProfileNames());
        negotiation.put("feature_profiles", AgentServingFeatureProfile.catalogMaps());
        negotiation.put("required_features", REQUIRED_FEATURES);
        negotiation.put("optional_features", OPTIONAL_FEATURES);
        negotiation.put("all_features", compatibilityFeatures());
        negotiation.put("unknown_feature_policy", "ignore_for_forward_compatibility");
        negotiation.put("deprecation_policy", "features remain stable for the current contract version");
        return negotiation;
    }

    private static List<String> concat(List<String> first, List<String> second) {
        ArrayList<String> out = new ArrayList<>(first.size() + second.size());
        out.addAll(first);
        out.addAll(second);
        return List.copyOf(out);
    }
}
