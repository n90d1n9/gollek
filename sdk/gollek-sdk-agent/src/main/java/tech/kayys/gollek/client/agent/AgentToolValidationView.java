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
 * Read-only view over Gollek's tool contract validation response.
 *
 * <p>This view exposes normalized OpenAI/MCP tool schema summaries and warnings.
 * It deliberately does not authorize, execute, retry, or loop tool calls.
 */
public final class AgentToolValidationView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentToolValidationView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentToolValidationView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentToolValidationView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentToolValidationView(mapper, node);
    }

    public static AgentToolValidationView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentToolValidationView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentToolValidationView(objectMapper, response);
    }

    public static AgentToolValidationView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentToolValidationView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        return new AgentToolValidationView(mapper, mapper.readTree(responseJson));
    }

    public String object() {
        return text(raw.path("object"));
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

    public boolean toolAuthorizationEnabled() {
        return raw.path("boundary").path("tool_authorization").asBoolean(false);
    }

    public Map<String, Object> trace() {
        return map(raw.path("trace"));
    }

    public int toolCount() {
        return integer(raw.path("tool_count"), tools().size());
    }

    public boolean hasWarnings() {
        return !warnings().isEmpty();
    }

    public List<NormalizedTool> tools() {
        JsonNode tools = raw.path("normalized");
        if (!tools.isArray()) {
            return List.of();
        }
        List<NormalizedTool> out = new ArrayList<>();
        for (JsonNode tool : tools) {
            out.add(new NormalizedTool(
                    integer(tool.path("index"), out.size()),
                    text(tool.path("name")),
                    text(tool.path("type")),
                    tool.path("strict").asBoolean(false),
                    tool.path("description_present").asBoolean(false),
                    integer(tool.path("warning_count"), 0),
                    strings(tool.path("parameter_keys")),
                    map(tool.path("parameter_schema")),
                    map(tool.path("metadata")),
                    tool));
        }
        return List.copyOf(out);
    }

    public List<String> toolNames() {
        List<String> out = new ArrayList<>();
        for (NormalizedTool tool : tools()) {
            if (tool.name() != null) {
                out.add(tool.name());
            }
        }
        return List.copyOf(out);
    }

    public boolean hasTool(String name) {
        return name != null && toolNames().contains(name);
    }

    public boolean hasMcpTools() {
        return tools().stream().anyMatch(tool -> "mcp_tool".equals(tool.type()));
    }

    public List<Warning> warnings() {
        JsonNode warnings = raw.path("warnings");
        if (!warnings.isArray()) {
            return List.of();
        }
        List<Warning> out = new ArrayList<>();
        for (JsonNode warning : warnings) {
            out.add(new Warning(
                    integer(warning.path("index"), -1),
                    text(warning.path("path")),
                    text(warning.path("code")),
                    text(warning.path("message")),
                    map(warning),
                    warning));
        }
        return List.copyOf(out);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> raw() {
        return objectMapper.convertValue(raw, Map.class);
    }

    public JsonNode rawNode() {
        return raw;
    }

    public record NormalizedTool(
            int index,
            String name,
            String type,
            boolean strict,
            boolean descriptionPresent,
            int warningCount,
            List<String> parameterKeys,
            Map<String, Object> parameterSchema,
            Map<String, Object> metadata,
            JsonNode raw) {

        public NormalizedTool {
            parameterKeys = parameterKeys == null ? List.of() : List.copyOf(parameterKeys);
            parameterSchema = parameterSchema == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(parameterSchema));
            metadata = metadata == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }

    public record Warning(
            int index,
            String path,
            String code,
            String message,
            Map<String, Object> raw,
            JsonNode rawNode) {

        public Warning {
            raw = raw == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(raw));
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

    private static int integer(JsonNode node, int fallback) {
        return node != null && node.isNumber() ? node.asInt() : fallback;
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        String value = node.asText();
        return value == null || value.isBlank() ? null : value;
    }
}
