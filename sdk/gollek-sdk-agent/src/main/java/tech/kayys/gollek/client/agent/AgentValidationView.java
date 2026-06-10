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
 * Read-only view over Gollek's request validation response.
 *
 * <p>This view exposes the normalized dry-run summary returned by
 * {@code /v1/agent/validate}. It confirms how Gollek would map an agent request
 * into serving inputs without invoking a model, executing tools, running
 * retrieval, or owning workflow state. Embedding validation summaries are
 * exposed as a typed sub-view so agent orchestrators can assert request shape
 * and RAG ownership boundaries without parsing raw maps.
 */
public final class AgentValidationView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();
    private static final String ORCHESTRATOR_OWNER = "agent_orchestrator";

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentValidationView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentValidationView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentValidationView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentValidationView(mapper, node);
    }

    public static AgentValidationView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentValidationView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentValidationView(objectMapper, response);
    }

    public static AgentValidationView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentValidationView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        return new AgentValidationView(mapper, mapper.readTree(responseJson));
    }

    public String object() {
        return text(raw.path("object"));
    }

    public String surface() {
        return text(raw.path("surface"));
    }

    public boolean valid() {
        return raw.path("valid").asBoolean(false);
    }

    public boolean modelInvoked() {
        return raw.path("model_invoked").asBoolean(false);
    }

    public boolean validationOnly() {
        return raw.path("boundary").path("validation_only").asBoolean(false);
    }

    public boolean toolExecutionEnabled() {
        return raw.path("boundary").path("tool_execution").asBoolean(false);
    }

    public boolean retrievalExecutionEnabled() {
        return raw.path("boundary").path("retrieval_execution").asBoolean(false);
    }

    public Map<String, Object> trace() {
        return map(raw.path("trace"));
    }

    public Map<String, Object> normalized() {
        return map(raw.path("normalized"));
    }

    public String requestId() {
        return firstNonBlank(
                text(raw.path("normalized").path("request_id")),
                string(trace().get("request_id")));
    }

    public String traceId() {
        return firstNonBlank(
                text(raw.path("normalized").path("trace_id")),
                string(trace().get("trace_id")));
    }

    public String sessionId() {
        return firstNonBlank(
                text(raw.path("normalized").path("session_id")),
                string(trace().get("session_id")));
    }

    public String userId() {
        return firstNonBlank(
                text(raw.path("normalized").path("user_id")),
                string(trace().get("user_id")));
    }

    public String model() {
        return text(raw.path("normalized").path("model"));
    }

    public boolean streaming() {
        return raw.path("normalized").path("streaming").asBoolean(false);
    }

    public int messageCount() {
        return integer(raw.path("normalized").path("message_count"), 0);
    }

    public int inputCount() {
        return integer(raw.path("normalized").path("input_count"), 0);
    }

    public List<Integer> inputLengths() {
        return integers(raw.path("normalized").path("input_lengths"));
    }

    public Integer requestedDimensions() {
        return nullableInteger(raw.path("normalized").path("requested_dimensions"));
    }

    public String encodingFormat() {
        return text(raw.path("normalized").path("encoding_format"));
    }

    public Map<String, Object> metadata() {
        return map(raw.path("normalized").path("metadata"));
    }

    public int toolCount() {
        return integer(raw.path("normalized").path("tool_count"), tools().size());
    }

    public boolean hasTools() {
        return toolCount() > 0;
    }

    public List<ToolSummary> tools() {
        JsonNode tools = raw.path("normalized").path("tools");
        if (!tools.isArray()) {
            return List.of();
        }
        List<ToolSummary> out = new ArrayList<>();
        for (JsonNode tool : tools) {
            out.add(new ToolSummary(
                    text(tool.path("name")),
                    text(tool.path("type")),
                    tool.path("strict").asBoolean(false),
                    strings(tool.path("parameter_keys")),
                    map(tool.path("metadata")),
                    tool));
        }
        return List.copyOf(out);
    }

    public List<String> parameterKeys() {
        return strings(raw.path("normalized").path("parameter_keys"));
    }

    public boolean ragContextInjected() {
        return raw.path("normalized").path("rag").path("injected").asBoolean(false);
    }

    public int ragContextItems() {
        return integer(raw.path("normalized").path("rag").path("items"), 0);
    }

    public String ragContextAlias() {
        return text(raw.path("normalized").path("rag").path("alias"));
    }

    public boolean includeUsageOnStream() {
        return raw.path("normalized").path("stream_options").path("include_usage").asBoolean(false);
    }

    public Map<String, Object> toolContract() {
        return map(raw.path("normalized").path("tool_contract"));
    }

    public boolean toolContractValid() {
        JsonNode contract = raw.path("normalized").path("tool_contract");
        return !contract.isObject() || contract.path("valid").asBoolean(false);
    }

    public int toolContractWarningCount() {
        return integer(raw.path("normalized").path("tool_contract").path("warning_count"), 0);
    }

    public boolean embeddingSurface() {
        return "embeddings".equals(surface());
    }

    public EmbeddingValidation embeddingValidation() {
        JsonNode normalized = raw.path("normalized");
        JsonNode rag = normalized.path("rag");
        return new EmbeddingValidation(
                model(),
                inputCount(),
                inputLengths(),
                requestedDimensions(),
                encodingFormat(),
                parameterKeys(),
                metadata(),
                new EmbeddingRagBoundary(
                        rag.path("embedding_generation").asBoolean(false),
                        rag.path("retrieval_execution").asBoolean(retrievalExecutionEnabled()),
                        text(rag.path("retrieval_policy_owned_by")),
                        text(rag.path("vector_store_owned_by")),
                        rag),
                normalized);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> raw() {
        return objectMapper.convertValue(raw, Map.class);
    }

    public JsonNode rawNode() {
        return raw;
    }

    public record ToolSummary(
            String name,
            String type,
            boolean strict,
            List<String> parameterKeys,
            Map<String, Object> metadata,
            JsonNode raw) {

        public ToolSummary {
            parameterKeys = parameterKeys == null ? List.of() : List.copyOf(parameterKeys);
            metadata = metadata == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }

    public record EmbeddingValidation(
            String model,
            int inputCount,
            List<Integer> inputLengths,
            Integer requestedDimensions,
            String encodingFormat,
            List<String> parameterKeys,
            Map<String, Object> metadata,
            EmbeddingRagBoundary rag,
            JsonNode raw) {

        public EmbeddingValidation {
            inputLengths = inputLengths == null ? List.of() : List.copyOf(inputLengths);
            parameterKeys = parameterKeys == null ? List.of() : List.copyOf(parameterKeys);
            metadata = metadata == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
            rag = rag == null
                    ? new EmbeddingRagBoundary(false, false, null, null, null)
                    : rag;
            raw = raw == null ? DEFAULT_MAPPER.getNodeFactory().objectNode() : raw;
        }

        public boolean hasRequestedDimensions() {
            return requestedDimensions != null;
        }

        public boolean storageOwnedByOrchestrator() {
            return rag.ownedByOrchestrator();
        }
    }

    public record EmbeddingRagBoundary(
            boolean embeddingGeneration,
            boolean retrievalExecution,
            String retrievalPolicyOwnedBy,
            String vectorStoreOwnedBy,
            JsonNode raw) {

        public EmbeddingRagBoundary {
            raw = raw == null ? DEFAULT_MAPPER.getNodeFactory().objectNode() : raw;
        }

        public boolean retrievalPolicyOwnedByOrchestrator() {
            return ORCHESTRATOR_OWNER.equals(retrievalPolicyOwnedBy);
        }

        public boolean vectorStoreOwnedByOrchestrator() {
            return ORCHESTRATOR_OWNER.equals(vectorStoreOwnedBy);
        }

        public boolean ownedByOrchestrator() {
            return retrievalPolicyOwnedByOrchestrator() && vectorStoreOwnedByOrchestrator();
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return objectMapper.convertValue(node, Map.class);
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

    private static List<Integer> integers(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<Integer> out = new ArrayList<>();
        for (JsonNode item : node) {
            Integer value = nullableInteger(item);
            if (value != null) {
                out.add(value);
            }
        }
        return List.copyOf(out);
    }

    private static int integer(JsonNode node, int fallback) {
        return node != null && node.isNumber() ? node.asInt() : fallback;
    }

    private static Integer nullableInteger(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        if (node.isNumber()) {
            return node.asInt();
        }
        if (node.isTextual()) {
            try {
                return Integer.parseInt(node.asText());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static String string(Object value) {
        return value instanceof String text && !text.isBlank() ? text : null;
    }

    private static String firstNonBlank(String first, String second) {
        return first != null && !first.isBlank() ? first : second;
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
