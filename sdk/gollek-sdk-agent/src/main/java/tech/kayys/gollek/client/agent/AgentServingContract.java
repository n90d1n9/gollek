package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Read-only view over Gollek's agent-facing serving contract.
 *
 * <p>The view is meant for agent runtimes that run on top of Gollek. It helps
 * callers verify that Gollek is acting as an inference and serving boundary
 * while planning, memory policy, tool execution, and workflow state remain
 * outside Gollek.
 */
public final class AgentServingContract {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();
    private static final List<String> REQUIRED_GOLLEK_OWNS = List.of(
            "model_serving",
            "provider_routing",
            "system_prompt_mapping",
            "tool_schema_ingestion",
            "tool_contract_validation",
            "mcp_registry_discovery",
            "embedding_generation",
            "rag_context_injection");
    private static final List<String> REQUIRED_ORCHESTRATOR_OWNS = List.of(
            "planning",
            "memory_policy",
            "retrieval_policy",
            "vector_store_ownership",
            "tool_authorization",
            "tool_execution",
            "tool_result_loop",
            "workflow_state");
    private static final List<String> REQUIRED_ENDPOINTS = List.of(
            "chat_completions",
            "responses",
            "embeddings",
            "model_capabilities",
            "mcp_tools",
            "agent_capabilities",
            "agent_contract",
            "agent_readiness_issues",
            "agent_preflight",
            "agent_validation",
            "agent_tool_validation");

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentServingContract(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentServingContract from(Map<String, Object> contract) {
        return from(contract, null);
    }

    public static AgentServingContract from(Map<String, Object> contract, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = contract == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(contract, JsonNode.class);
        return new AgentServingContract(mapper, node);
    }

    public static AgentServingContract from(JsonNode contract) {
        return from(contract, null);
    }

    public static AgentServingContract from(JsonNode contract, ObjectMapper objectMapper) {
        return new AgentServingContract(objectMapper, contract);
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

    public List<String> compatibility() {
        return featureNegotiation().allFeatures();
    }

    public boolean supportsFeature(String feature) {
        return featureNegotiation().supportsFeature(feature);
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

    public List<String> gollekOwns() {
        return strings(raw.path("boundary").path("gollek_owns"));
    }

    public List<String> orchestratorOwns() {
        return strings(raw.path("boundary").path("agent_orchestrator_owns"));
    }

    public boolean gollekOwns(String responsibility) {
        return contains(gollekOwns(), responsibility);
    }

    public boolean orchestratorOwns(String responsibility) {
        return contains(orchestratorOwns(), responsibility);
    }

    public boolean toolExecutionEnabled() {
        return raw.path("boundary").path("tool_execution").asBoolean(false);
    }

    public boolean retrievalExecutionEnabled() {
        return raw.path("boundary").path("retrieval_execution").asBoolean(false);
    }

    public Map<String, Endpoint> endpoints() {
        Map<String, Endpoint> out = new LinkedHashMap<>();
        JsonNode endpoints = raw.path("endpoints");
        if (endpoints.isObject()) {
            endpoints.fields().forEachRemaining(entry -> out.put(entry.getKey(), Endpoint.from(entry.getValue())));
        }
        return Collections.unmodifiableMap(out);
    }

    public Endpoint endpoint(String name) {
        if (name == null || name.isBlank()) {
            return Endpoint.empty();
        }
        return Endpoint.from(raw.path("endpoints").path(name));
    }

    public boolean hasEndpoint(String name) {
        Endpoint endpoint = endpoint(name);
        return endpoint.method() != null && endpoint.path() != null;
    }

    public String endpointPath(String name) {
        return endpoint(name).path();
    }

    public Map<String, Schema> schemas() {
        Map<String, Schema> out = new LinkedHashMap<>();
        JsonNode schemas = raw.path("schemas");
        if (schemas.isObject()) {
            schemas.fields().forEachRemaining(entry -> out.put(entry.getKey(), new Schema(objectMapper, entry.getValue())));
        }
        return Collections.unmodifiableMap(out);
    }

    public Schema schema(String name) {
        if (name == null || name.isBlank()) {
            return Schema.empty(objectMapper);
        }
        return new Schema(objectMapper, raw.path("schemas").path(name));
    }

    public boolean schemaRequires(String schema, String field) {
        return contains(schema(schema).required(), field);
    }

    public boolean schemaHasProperty(String schema, String property) {
        return schema(schema).hasProperty(property);
    }

    public boolean supportsStreaming() {
        return "[DONE]".equals(text(raw.path("streaming").path("done_sentinel")))
                && !chatStreamingEvents().isEmpty()
                && !responsesStreamingEvents().isEmpty();
    }

    public List<String> chatStreamingEvents() {
        return strings(raw.path("streaming").path("chat_completions_events"));
    }

    public List<String> responsesStreamingEvents() {
        return strings(raw.path("streaming").path("responses_events"));
    }

    public boolean hasRequiredServingBoundary() {
        return servingBoundaryIssues().isEmpty();
    }

    public List<String> servingBoundaryIssues() {
        List<String> issues = new ArrayList<>();
        if (!isInferenceServingEngine()) {
            issues.add("service_role must be inference_serving_engine");
        }
        if (!supportsContractVersion(AgentServingFeatureCatalog.CONTRACT_VERSION)) {
            issues.add("missing supported contract version: " + AgentServingFeatureCatalog.CONTRACT_VERSION);
        }
        for (String required : REQUIRED_GOLLEK_OWNS) {
            if (!gollekOwns(required)) {
                issues.add("missing Gollek-owned responsibility: " + required);
            }
        }
        for (String required : REQUIRED_ORCHESTRATOR_OWNS) {
            if (!orchestratorOwns(required)) {
                issues.add("missing orchestrator-owned responsibility: " + required);
            }
        }
        for (String required : REQUIRED_ENDPOINTS) {
            if (!hasEndpoint(required)) {
                issues.add("missing endpoint: " + required);
            }
        }
        if (toolExecutionEnabled()) {
            issues.add("Gollek contract must not enable tool execution");
        }
        if (retrievalExecutionEnabled()) {
            issues.add("Gollek contract must not enable retrieval execution");
        }
        if (!supportsStreaming()) {
            issues.add("streaming contract must advertise chat and Responses events");
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

    public record Endpoint(String method, String path, String schema, JsonNode raw) {
        static Endpoint from(JsonNode node) {
            if (node == null || !node.isObject()) {
                return empty();
            }
            return new Endpoint(
                    text(node.path("method")),
                    text(node.path("path")),
                    text(node.path("schema")),
                    node);
        }

        static Endpoint empty() {
            return new Endpoint(null, null, null, com.fasterxml.jackson.databind.node.MissingNode.getInstance());
        }
    }

    public static final class Schema {
        private final ObjectMapper objectMapper;
        private final JsonNode raw;

        private Schema(ObjectMapper objectMapper, JsonNode raw) {
            this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
            this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
        }

        static Schema empty(ObjectMapper objectMapper) {
            return new Schema(objectMapper, com.fasterxml.jackson.databind.node.MissingNode.getInstance());
        }

        public String type() {
            return text(raw.path("type"));
        }

        public List<String> required() {
            return strings(raw.path("required"));
        }

        public List<String> properties() {
            JsonNode properties = raw.path("properties");
            if (!properties.isObject()) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            properties.fieldNames().forEachRemaining(out::add);
            return List.copyOf(out);
        }

        public boolean requires(String field) {
            return contains(required(), field);
        }

        public boolean hasProperty(String property) {
            return contains(properties(), property);
        }

        @SuppressWarnings("unchecked")
        public Map<String, Object> raw() {
            return objectMapper.convertValue(raw, Map.class);
        }

        public JsonNode rawNode() {
            return raw;
        }
    }

    private static boolean contains(List<String> values, String value) {
        return value != null && values.contains(value);
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
