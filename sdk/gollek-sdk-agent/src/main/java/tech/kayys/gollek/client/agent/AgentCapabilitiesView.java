package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Read-only view over Gollek's agent-facing capability discovery response.
 *
 * <p>This view is intentionally about serving compatibility and endpoint
 * discovery. It does not model planner state, memory, or tool execution.
 */
public final class AgentCapabilitiesView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();
    private static final List<String> REQUIRED_ENDPOINTS = List.of(
            "openai_chat_completions",
            "openai_responses",
            "openai_embeddings",
            "model_capabilities",
            "agent_contract",
            "agent_readiness_issues",
            "agent_preflight",
            "agent_validation",
            "agent_tool_validation",
            "mcp_tools");

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentCapabilitiesView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentCapabilitiesView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentCapabilitiesView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentCapabilitiesView(mapper, node);
    }

    public static AgentCapabilitiesView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentCapabilitiesView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentCapabilitiesView(objectMapper, response);
    }

    public static AgentCapabilitiesView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentCapabilitiesView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        return new AgentCapabilitiesView(mapper, mapper.readTree(responseJson));
    }

    public String object() {
        return text(raw.path("object"));
    }

    public String version() {
        return text(raw.path("version"));
    }

    public String contractVersion() {
        return featureNegotiation().contractVersion();
    }

    public List<String> supportedContractVersions() {
        return featureNegotiation().supportedContractVersions();
    }

    public boolean supportsContractVersion(String version) {
        return featureNegotiation().supportsContractVersion(version);
    }

    public AgentFeatureNegotiation featureNegotiation() {
        return AgentFeatureNegotiation.from(raw, objectMapper);
    }

    public String serviceRole() {
        return text(raw.path("service_role"));
    }

    public boolean isInferenceServingEngine() {
        return AgentServingFeatureCatalog.SERVICE_ROLE_INFERENCE_SERVING_ENGINE.equals(serviceRole());
    }

    public List<String> compatibility() {
        List<String> out = new ArrayList<>(strings(raw.path("compatibility")));
        for (String feature : featureNegotiation().allFeatures()) {
            if (!out.contains(feature)) {
                out.add(feature);
            }
        }
        return List.copyOf(out);
    }

    public boolean supports(String feature) {
        return feature != null && compatibility().contains(feature);
    }

    public boolean supportsOpenAiChat() {
        return supports("openai_chat_completions") && hasEndpoint("openai_chat_completions");
    }

    public boolean supportsOpenAiResponses() {
        return supports("openai_responses") && hasEndpoint("openai_responses");
    }

    public boolean supportsEmbeddings() {
        return supports("openai_embeddings") && hasEndpoint("openai_embeddings");
    }

    public boolean supportsMcpToolDiscovery() {
        return supports("mcp_tool_discovery") && hasEndpoint("mcp_tools");
    }

    public boolean supportsModelCapabilities() {
        return hasEndpoint("model_capabilities");
    }

    public boolean supportsReadinessIssueCatalog() {
        return supports("agent_readiness_issue_catalog") && hasEndpoint("agent_readiness_issues");
    }

    public boolean supportsRagContext() {
        return supports("rag_context");
    }

    public Map<String, String> endpoints() {
        Map<String, String> out = new LinkedHashMap<>();
        JsonNode endpoints = raw.path("endpoints");
        if (endpoints.isObject()) {
            endpoints.fields().forEachRemaining(entry -> {
                String value = text(entry.getValue());
                if (value != null) {
                    out.put(entry.getKey(), value);
                }
            });
        }
        return Collections.unmodifiableMap(out);
    }

    public boolean hasEndpoint(String name) {
        return endpoint(name) != null;
    }

    public String endpoint(String name) {
        if (name == null || name.isBlank()) {
            return null;
        }
        return text(raw.path("endpoints").path(name));
    }

    public List<String> gollekOwns() {
        return strings(raw.path("agent_boundary").path("gollek_owns"));
    }

    public List<String> orchestratorOwns() {
        return strings(raw.path("agent_boundary").path("agent_orchestrator_owns"));
    }

    public boolean gollekOwns(String responsibility) {
        return containsNormalized(gollekOwns(), responsibility);
    }

    public boolean orchestratorOwns(String responsibility) {
        return containsNormalized(orchestratorOwns(), responsibility);
    }

    public String apiKeyHeader() {
        return text(raw.path("auth").path("x_api_key_header"));
    }

    public String authorizationHeaderScheme() {
        return text(raw.path("auth").path("authorization_header"));
    }

    public String requestIdHeader() {
        return text(raw.path("traceability").path("request_id_header"));
    }

    public String traceIdHeader() {
        return text(raw.path("traceability").path("trace_id_header"));
    }

    public String sessionIdHeader() {
        return text(raw.path("traceability").path("session_id_header"));
    }

    public String userIdHeader() {
        return text(raw.path("traceability").path("user_id_header"));
    }

    public String traceMetadataKey() {
        return text(raw.path("traceability").path("metadata_key"));
    }

    public boolean hasRequiredAgentServingCapabilities() {
        return agentServingCapabilityIssues().isEmpty();
    }

    public List<String> agentServingCapabilityIssues() {
        List<String> issues = new ArrayList<>();
        if (!isInferenceServingEngine()) {
            issues.add("service_role must be inference_serving_engine");
        }
        if (!supportsContractVersion(AgentServingFeatureCatalog.CONTRACT_VERSION)) {
            issues.add("missing supported contract version: " + AgentServingFeatureCatalog.CONTRACT_VERSION);
        }
        for (String feature : AgentServingFeatureCatalog.REQUIRED_FEATURES) {
            if (!supports(feature)) {
                issues.add("missing compatibility: " + feature);
            }
        }
        for (String endpoint : REQUIRED_ENDPOINTS) {
            if (!hasEndpoint(endpoint)) {
                issues.add("missing endpoint: " + endpoint);
            }
        }
        if (!gollekOwns("model serving")) {
            issues.add("agent boundary must keep model serving in Gollek");
        }
        if (!orchestratorOwns("tool execution loops")) {
            issues.add("agent boundary must keep tool execution loops in the orchestrator");
        }
        if (!"X-API-Key".equals(apiKeyHeader())) {
            issues.add("auth.x_api_key_header must be X-API-Key");
        }
        if (!"Bearer token".equals(authorizationHeaderScheme())) {
            issues.add("auth.authorization_header must be Bearer token");
        }
        if (!AgentRequestOptions.REQUEST_ID_HEADER.equals(requestIdHeader())) {
            issues.add("traceability.request_id_header must be " + AgentRequestOptions.REQUEST_ID_HEADER);
        }
        return List.copyOf(issues);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> raw() {
        return objectMapper.convertValue(raw, Map.class);
    }

    public JsonNode rawNode() {
        return raw;
    }

    private static boolean containsNormalized(List<String> values, String value) {
        if (value == null) {
            return false;
        }
        String expected = normalize(value);
        return values.stream().map(AgentCapabilitiesView::normalize).anyMatch(expected::equals);
    }

    private static String normalize(String value) {
        return value == null
                ? ""
                : value.toLowerCase(java.util.Locale.ROOT).replaceAll("[^a-z0-9]+", "_").replaceAll("^_|_$", "");
    }

    private static List<String> strings(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        for (JsonNode item : node) {
            String value = text(item);
            if (value != null) {
                out.add(value);
            }
        }
        return List.copyOf(out);
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
