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
 * Read-only view over one model's agent-serving capability matrix.
 *
 * <p>This view helps an external agent runtime decide whether a model can be
 * routed through Gollek's serving APIs. It does not choose a model, plan a
 * workflow, execute tools, or own retrieval policy.
 */
public final class AgentModelCapabilitiesView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentModelCapabilitiesView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentModelCapabilitiesView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentModelCapabilitiesView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentModelCapabilitiesView(mapper, node);
    }

    public static AgentModelCapabilitiesView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentModelCapabilitiesView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentModelCapabilitiesView(objectMapper, response);
    }

    public static AgentModelCapabilitiesView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentModelCapabilitiesView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        return new AgentModelCapabilitiesView(mapper, mapper.readTree(responseJson));
    }

    public String modelId() {
        return text(raw.path("model_id"));
    }

    public boolean known() {
        return raw.path("known").asBoolean(false);
    }

    public boolean available() {
        return raw.path("available").asBoolean(false);
    }

    public String preferredProvider() {
        return text(raw.path("preferred_provider"));
    }

    public String format() {
        return text(raw.path("format"));
    }

    public String architecture() {
        return text(raw.path("architecture"));
    }

    public String quantization() {
        return text(raw.path("quantization"));
    }

    public Limits limits() {
        JsonNode limits = raw.path("limits");
        return new Limits(
                nullableLong(limits.path("context_tokens")),
                nullableLong(limits.path("input_tokens")),
                nullableInteger(limits.path("output_tokens")),
                nullableInteger(limits.path("embedding_dimensions")));
    }

    public boolean supportsChatCompletions() {
        return flag("openai_compatibility", "chat_completions") && flag("api_contract", "chat_completions");
    }

    public boolean supportsChatStreaming() {
        return flag("openai_compatibility", "chat_streaming") && flag("api_contract", "chat_streaming");
    }

    public boolean supportsResponses() {
        return flag("openai_compatibility", "responses") && flag("api_contract", "responses");
    }

    public boolean supportsResponsesStreaming() {
        return flag("openai_compatibility", "responses_streaming") && flag("api_contract", "responses_streaming");
    }

    public boolean supportsEmbeddings() {
        return supportsEmbeddingGeneration();
    }

    public boolean supportsEmbeddingEndpoint() {
        return flag("api_contract", "embeddings_endpoint")
                || "/v1/embeddings".equals(embeddings().endpoint());
    }

    public boolean supportsEmbeddingGeneration() {
        return embeddings().generation();
    }

    public boolean embeddingRetrievalPolicyOwnedByGollek() {
        return embeddings().retrievalPolicyOwnedByGollek();
    }

    public boolean embeddingVectorStoreOwnedByGollek() {
        return embeddings().vectorStoreOwnedByGollek();
    }

    public boolean supportsCompletion() {
        return flag("inference", "completion");
    }

    public boolean supportsStreaming() {
        return flag("inference", "streaming");
    }

    public boolean supportsSystemPrompt() {
        return flag("inference", "system_prompt") || flag("api_contract", "system_prompt");
    }

    public boolean supportsJsonMode() {
        return flag("inference", "json_mode");
    }

    public boolean supportsStructuredOutputs() {
        return flag("inference", "structured_outputs");
    }

    public boolean supportsToolDefinitions() {
        return flag("tooling", "tool_definitions") && flag("api_contract", "tools_request_schema");
    }

    public boolean supportsToolChoice() {
        return flag("tooling", "tool_choice");
    }

    public boolean supportsModelToolCalling() {
        return flag("tooling", "model_tool_calling");
    }

    public boolean supportsMcpToolDefinitions() {
        return flag("tooling", "mcp_tool_definitions") && flag("api_contract", "mcp_tool_definitions");
    }

    public boolean toolExecutionEnabled() {
        return flag("tooling", "tool_execution");
    }

    public boolean supportsRagContextInjection() {
        return flag("rag", "context_injection") && flag("api_contract", "rag_context_injection");
    }

    public boolean retrievalPolicyOwnedByGollek() {
        return flag("rag", "retrieval_policy");
    }

    public boolean vectorStoreOwnedByGollek() {
        return flag("rag", "vector_store_ownership");
    }

    public boolean supportsText() {
        return flag("modalities", "text");
    }

    public boolean supportsMultimodal() {
        return flag("modalities", "multimodal");
    }

    public List<ProviderCandidate> providerCandidates() {
        JsonNode candidates = raw.path("provider_candidates");
        if (!candidates.isArray()) {
            return List.of();
        }
        List<ProviderCandidate> out = new ArrayList<>();
        for (JsonNode candidate : candidates) {
            out.add(new ProviderCandidate(
                    text(candidate.path("id")),
                    text(candidate.path("name")),
                    text(candidate.path("health")),
                    candidate.path("supports_model").asBoolean(false),
                    map(candidate.path("capabilities")),
                    candidate));
        }
        return List.copyOf(out);
    }

    public boolean hasProviderCandidates() {
        return !providerCandidates().isEmpty();
    }

    public Map<String, Object> metadata() {
        return map(raw.path("metadata"));
    }

    public Map<String, Object> apiContract() {
        return map(raw.path("api_contract"));
    }

    public Map<String, Object> openAiCompatibility() {
        return map(raw.path("openai_compatibility"));
    }

    public Map<String, Object> inference() {
        return map(raw.path("inference"));
    }

    public Map<String, Object> tooling() {
        return map(raw.path("tooling"));
    }

    public Map<String, Object> rag() {
        return map(raw.path("rag"));
    }

    public EmbeddingCapabilities embeddings() {
        JsonNode embeddings = raw.path("embeddings");
        Integer dimensions = nullableInteger(embeddings.path("dimensions"));
        if (dimensions == null) {
            dimensions = limits().embeddingDimensions();
        }
        boolean generation = fieldBool(embeddings, "generation", rawSupportsEmbeddings());
        return new EmbeddingCapabilities(
                generation,
                textOrDefault(embeddings.path("endpoint"),
                        flag("api_contract", "embeddings_endpoint") ? "/v1/embeddings" : null),
                fieldBool(embeddings, "openai_compatible", flag("openai_compatibility", "embeddings")),
                dimensions,
                stringList(embeddings.path("encoding_formats")),
                stringList(embeddings.path("input_aliases")),
                fieldBool(embeddings, "batch_inputs", false),
                fieldBool(embeddings, "metadata_passthrough", false),
                fieldBool(embeddings, "retrieval_policy", false),
                fieldBool(embeddings, "vector_store_ownership", false),
                map(embeddings));
    }

    public Map<String, Object> embeddingCapabilities() {
        return embeddings().raw();
    }

    public Map<String, Object> modalities() {
        return map(raw.path("modalities"));
    }

    public boolean hasRequiredAgentServingRoute() {
        return agentServingRouteIssues().isEmpty();
    }

    public boolean hasRequiredAgentServingRoute(String surface) {
        return agentServingRouteIssues(surface).isEmpty();
    }

    public boolean hasRequiredEmbeddingRoute() {
        return embeddingRouteIssues().isEmpty();
    }

    public List<String> agentServingRouteIssues() {
        return agentServingRouteIssues(null);
    }

    public List<String> agentServingRouteIssues(String surface) {
        if ("embeddings".equals(normalizeSurface(surface))) {
            return embeddingRouteIssues();
        }
        List<String> issues = new ArrayList<>();
        if (!available()) {
            issues.add("model is not available through Gollek serving");
        }
        if (!supportsCompletion()) {
            issues.add("model does not advertise completion serving");
        }
        if (!supportsChatCompletions()) {
            issues.add("model does not advertise Chat Completions compatibility");
        }
        if (!supportsResponses()) {
            issues.add("model does not advertise Responses compatibility");
        }
        if (!supportsSystemPrompt()) {
            issues.add("model does not advertise system prompt mapping");
        }
        if (!supportsToolDefinitions()) {
            issues.add("model does not advertise tool definition ingestion");
        }
        if (toolExecutionEnabled()) {
            issues.add("Gollek model capability must not enable tool execution");
        }
        if (!supportsRagContextInjection()) {
            issues.add("model does not advertise RAG context injection");
        }
        if (retrievalPolicyOwnedByGollek()) {
            issues.add("retrieval policy must stay with the agent orchestrator");
        }
        if (vectorStoreOwnedByGollek()) {
            issues.add("vector store ownership must stay with the agent orchestrator");
        }
        return List.copyOf(issues);
    }

    public List<String> embeddingRouteIssues() {
        List<String> issues = new ArrayList<>();
        if (!available()) {
            issues.add("model is not available through Gollek serving");
        }
        if (!supportsEmbeddingEndpoint()) {
            issues.add("model route does not advertise the embeddings endpoint");
        }
        if (!supportsEmbeddingGeneration()) {
            issues.add("model route does not support embedding generation for /v1/embeddings");
        }
        if (embeddingRetrievalPolicyOwnedByGollek()) {
            issues.add("embedding route must not own retrieval policy");
        }
        if (embeddingVectorStoreOwnedByGollek()) {
            issues.add("embedding route must not own vector store state");
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

    public record Limits(
            Long contextTokens,
            Long inputTokens,
            Integer outputTokens,
            Integer embeddingDimensions) {
    }

    public record EmbeddingCapabilities(
            boolean generation,
            String endpoint,
            boolean openAiCompatible,
            Integer dimensions,
            List<String> encodingFormats,
            List<String> inputAliases,
            boolean batchInputs,
            boolean metadataPassthrough,
            boolean retrievalPolicyOwnedByGollek,
            boolean vectorStoreOwnedByGollek,
            Map<String, Object> raw) {

        public EmbeddingCapabilities {
            encodingFormats = encodingFormats == null ? List.of() : List.copyOf(encodingFormats);
            inputAliases = inputAliases == null ? List.of() : List.copyOf(inputAliases);
            raw = raw == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(raw));
        }

        public boolean storageOwnedByOrchestrator() {
            return !retrievalPolicyOwnedByGollek && !vectorStoreOwnedByGollek;
        }
    }

    public record ProviderCandidate(
            String id,
            String name,
            String health,
            boolean supportsModel,
            Map<String, Object> capabilities,
            JsonNode raw) {

        public ProviderCandidate {
            capabilities = capabilities == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(capabilities));
        }

        public boolean supportsStreaming() {
            return bool(capabilities.get("streaming"));
        }

        public boolean supportsEmbeddings() {
            return bool(capabilities.get("embeddings"));
        }

        public boolean supportsToolCalling() {
            return bool(capabilities.get("tool_calling")) || bool(capabilities.get("function_calling"));
        }

        public boolean supportsStructuredOutputs() {
            return bool(capabilities.get("structured_outputs"));
        }

        public Integer maxContextTokens() {
            return integer(capabilities.get("max_context_tokens"));
        }

        public Integer maxOutputTokens() {
            return integer(capabilities.get("max_output_tokens"));
        }
    }

    private boolean flag(String section, String field) {
        return raw.path(section).path(field).asBoolean(false);
    }

    private boolean rawSupportsEmbeddings() {
        return flag("embeddings", "generation")
                || flag("openai_compatibility", "embeddings")
                || flag("modalities", "embeddings");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return objectMapper.convertValue(node, Map.class);
    }

    private static Long nullableLong(JsonNode node) {
        return node != null && node.isNumber() ? node.asLong() : null;
    }

    private static Integer nullableInteger(JsonNode node) {
        return node != null && node.isNumber() ? node.asInt() : null;
    }

    private static boolean fieldBool(JsonNode node, String field, boolean fallback) {
        JsonNode value = node == null ? null : node.path(field);
        return value != null && value.isBoolean() ? value.asBoolean() : fallback;
    }

    private static boolean bool(Object value) {
        return value instanceof Boolean bool && bool;
    }

    private static Integer integer(Object value) {
        return value instanceof Number number ? number.intValue() : null;
    }

    private static List<String> stringList(JsonNode node) {
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

    private static String textOrDefault(JsonNode node, String fallback) {
        String value = text(node);
        return value == null ? fallback : value;
    }

    private static String normalizeSurface(String surface) {
        String normalized = surface == null ? "" : surface.trim().toLowerCase(java.util.Locale.ROOT);
        return switch (normalized) {
            case "embedding", "embeddings" -> "embeddings";
            default -> normalized;
        };
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
