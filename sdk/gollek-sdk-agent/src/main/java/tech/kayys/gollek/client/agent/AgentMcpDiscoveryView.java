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
 * Read-only view over Gollek's MCP discovery endpoints.
 *
 * <p>Gollek exposes server registrations and tool schemas for agent runtimes.
 * Tool authorization, execution, retries, and result loops remain owned by the
 * caller.
 */
public final class AgentMcpDiscoveryView {
    private static final ObjectMapper DEFAULT_MAPPER = new ObjectMapper();

    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentMcpDiscoveryView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentMcpDiscoveryView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentMcpDiscoveryView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentMcpDiscoveryView(mapper, node);
    }

    public static AgentMcpDiscoveryView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentMcpDiscoveryView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentMcpDiscoveryView(objectMapper, response);
    }

    public static AgentMcpDiscoveryView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentMcpDiscoveryView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : DEFAULT_MAPPER;
        return new AgentMcpDiscoveryView(mapper, mapper.readTree(responseJson));
    }

    public boolean available() {
        return raw.path("available").asBoolean(false);
    }

    public String registryPath() {
        return text(raw.path("registry_path"));
    }

    public String serverName() {
        return text(raw.path("server"));
    }

    public String compatibility() {
        return text(raw.path("compat"));
    }

    public boolean enabledOnly() {
        return raw.path("enabled_only").asBoolean(false);
    }

    public String message() {
        return text(raw.path("message"));
    }

    public String boundaryRole() {
        return text(raw.path("boundary").path("role"));
    }

    public boolean discoveryOnly() {
        return "discovery_only".equals(boundaryRole()) && !toolExecutionEnabled();
    }

    public boolean toolExecutionEnabled() {
        return raw.path("boundary").path("tool_execution").asBoolean(false);
    }

    public List<String> gollekExposes() {
        return strings(raw.path("boundary").path("gollek_exposes"));
    }

    public List<String> orchestratorOwns() {
        return strings(raw.path("boundary").path("agent_orchestrator_owns"));
    }

    public List<Server> servers() {
        JsonNode servers = raw.path("servers");
        if (servers.isArray()) {
            List<Server> out = new ArrayList<>();
            for (JsonNode server : servers) {
                out.add(server(server));
            }
            return List.copyOf(out);
        }
        JsonNode server = raw.path("server");
        if (server.isObject()) {
            return List.of(server(server));
        }
        return List.of();
    }

    public List<String> serverNames() {
        List<String> out = new ArrayList<>();
        for (Server server : servers()) {
            if (server.name() != null) {
                out.add(server.name());
            }
        }
        return List.copyOf(out);
    }

    public boolean hasServer(String name) {
        return name != null && serverNames().contains(name);
    }

    public List<Tool> tools() {
        JsonNode tools = raw.path("tools");
        if (!tools.isArray()) {
            return List.of();
        }
        List<Tool> out = new ArrayList<>();
        for (JsonNode tool : tools) {
            out.add(tool(tool));
        }
        return List.copyOf(out);
    }

    public List<String> toolNames() {
        List<String> out = new ArrayList<>();
        for (Tool tool : tools()) {
            if (tool.name() != null) {
                out.add(tool.name());
            }
        }
        return List.copyOf(out);
    }

    public boolean hasTool(String name) {
        return name != null && toolNames().contains(name);
    }

    public List<Map<String, Object>> openAiToolDefinitions() {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Tool tool : tools()) {
            if (tool.openAiCompatible()) {
                out.add(tool.raw());
            }
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

    public record Server(
            String name,
            boolean enabled,
            String transport,
            String command,
            int argsCount,
            List<String> envKeys,
            String url,
            boolean rawJsonRedacted,
            JsonNode raw) {

        public Server {
            envKeys = envKeys == null ? List.of() : List.copyOf(envKeys);
        }
    }

    public record Tool(
            String type,
            String server,
            String name,
            String description,
            Map<String, Object> inputSchema,
            boolean executionEnabled,
            String mcpServer,
            String mcpToolName,
            Map<String, Object> raw,
            JsonNode rawNode) {

        public Tool {
            inputSchema = inputSchema == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(inputSchema));
            raw = raw == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(raw));
        }

        public boolean openAiCompatible() {
            return "function".equals(type) && raw.containsKey("function");
        }

        public boolean mcpTool() {
            return "mcp_tool".equals(type) || mcpServer != null || mcpToolName != null;
        }
    }

    private Server server(JsonNode node) {
        return new Server(
                text(node.path("name")),
                node.path("enabled").asBoolean(false),
                text(node.path("transport")),
                text(node.path("command")),
                integer(node.path("args_count"), 0),
                strings(node.path("env_keys")),
                text(node.path("url")),
                node.path("raw_json_redacted").asBoolean(false),
                node);
    }

    private Tool tool(JsonNode node) {
        JsonNode function = node.path("function");
        JsonNode metadata = firstObject(node.path("x_gollek"), node.path("metadata"));
        return new Tool(
                text(node.path("type")),
                firstNonBlank(text(node.path("server")), text(metadata.path("mcp_server"))),
                firstNonBlank(text(function.path("name")), text(node.path("name"))),
                firstNonBlank(text(function.path("description")), text(node.path("description"))),
                firstMap(function.path("parameters"), node.path("input_schema")),
                node.path("execution").asBoolean(metadata.path("tool_execution").asBoolean(false)),
                text(metadata.path("mcp_server")),
                text(metadata.path("mcp_tool_name")),
                map(node),
                node);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> firstMap(JsonNode first, JsonNode second) {
        JsonNode source = first != null && first.isObject() ? first : second;
        if (source == null || !source.isObject()) {
            return Map.of();
        }
        return objectMapper.convertValue(source, Map.class);
    }

    private static JsonNode firstObject(JsonNode first, JsonNode second) {
        if (first != null && first.isObject()) {
            return first;
        }
        if (second != null && second.isObject()) {
            return second;
        }
        return com.fasterxml.jackson.databind.node.MissingNode.getInstance();
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
