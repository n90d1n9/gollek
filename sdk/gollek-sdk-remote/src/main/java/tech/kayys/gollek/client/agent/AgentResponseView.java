package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Read-only view over non-streaming OpenAI-compatible agent responses.
 *
 * <p>This helper extracts response text, usage, metadata, trace, and tool-call
 * declarations. It does not execute tools or own agent workflow state.
 */
public final class AgentResponseView {
    private final ObjectMapper objectMapper;
    private final JsonNode raw;
    private final AgentStreamEvent.Surface surface;

    private AgentResponseView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : new ObjectMapper();
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
        this.surface = detectSurface(this.raw);
    }

    public static AgentResponseView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentResponseView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : new ObjectMapper();
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentResponseView(mapper, node);
    }

    public static AgentResponseView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentResponseView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentResponseView(objectMapper, response);
    }

    public static AgentResponseView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentResponseView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : new ObjectMapper();
        return new AgentResponseView(mapper, mapper.readTree(responseJson));
    }

    public AgentStreamEvent.Surface surface() {
        return surface;
    }

    public String id() {
        return text(raw.path("id"));
    }

    public String model() {
        return text(raw.path("model"));
    }

    public String outputText() {
        if (surface == AgentStreamEvent.Surface.CHAT_COMPLETIONS) {
            return firstNonBlank(
                    text(first(raw.path("choices")).path("message").path("content")),
                    "");
        }
        String topLevel = text(raw.path("output_text"));
        if (topLevel != null) {
            return topLevel;
        }
        List<String> parts = new ArrayList<>();
        JsonNode output = raw.path("output");
        if (output.isArray()) {
            for (JsonNode item : output) {
                appendContentText(parts, item.path("content"));
            }
        }
        return String.join("", parts);
    }

    public String finishReason() {
        if (surface == AgentStreamEvent.Surface.CHAT_COMPLETIONS) {
            return text(first(raw.path("choices")).path("finish_reason"));
        }
        return text(raw.path("metadata").path("gollek_stream").path("finish_reason"));
    }

    public AgentStreamEvent.Usage usage() {
        JsonNode node = raw.path("usage");
        if (!node.isObject()) {
            return null;
        }
        long promptTokens = longValue(node.path("prompt_tokens"), longValue(node.path("input_tokens"), 0));
        long completionTokens = longValue(
                node.path("completion_tokens"),
                longValue(node.path("output_tokens"), 0));
        long totalTokens = longValue(node.path("total_tokens"), promptTokens + completionTokens);
        return new AgentStreamEvent.Usage(promptTokens, completionTokens, totalTokens);
    }

    public Map<String, Object> metadata() {
        return map(raw.path("metadata"));
    }

    public Map<String, Object> trace() {
        Map<String, Object> trace = map(raw.path("trace"));
        if (!trace.isEmpty()) {
            return trace;
        }
        Object metadataTrace = metadata().get("gollek_trace");
        if (metadataTrace instanceof Map<?, ?> source) {
            LinkedHashMap<String, Object> out = new LinkedHashMap<>();
            source.forEach((key, value) -> {
                if (key instanceof String name) {
                    out.put(name, value);
                }
            });
            return out;
        }
        return Map.of();
    }

    public boolean hasToolCalls() {
        return !toolCalls().isEmpty();
    }

    public List<AgentStreamEvent.ToolCall> toolCalls() {
        if (surface == AgentStreamEvent.Surface.CHAT_COMPLETIONS) {
            return chatToolCalls();
        }
        if (surface == AgentStreamEvent.Surface.RESPONSES) {
            return responseToolCalls();
        }
        return List.of();
    }

    public JsonNode raw() {
        return raw;
    }

    private List<AgentStreamEvent.ToolCall> chatToolCalls() {
        JsonNode calls = first(raw.path("choices")).path("message").path("tool_calls");
        if (!calls.isArray()) {
            return List.of();
        }
        List<AgentStreamEvent.ToolCall> out = new ArrayList<>();
        for (JsonNode call : calls) {
            JsonNode function = call.path("function");
            out.add(new AgentStreamEvent.ToolCall(
                    text(call.path("id")),
                    integer(call.path("index"), out.size()),
                    firstNonBlank(text(call.path("type")), "function"),
                    text(function.path("name")),
                    text(function.path("arguments")),
                    text(call.path("call_id")),
                    text(call.path("status")),
                    call));
        }
        return List.copyOf(out);
    }

    private List<AgentStreamEvent.ToolCall> responseToolCalls() {
        JsonNode output = raw.path("output");
        if (!output.isArray()) {
            return List.of();
        }
        List<AgentStreamEvent.ToolCall> out = new ArrayList<>();
        for (JsonNode item : output) {
            String type = text(item.path("type"));
            if (!"function_call".equals(type) && !"custom_tool_call".equals(type)) {
                continue;
            }
            out.add(new AgentStreamEvent.ToolCall(
                    text(item.path("id")),
                    integer(item.path("output_index"), out.size()),
                    type,
                    text(item.path("name")),
                    text(item.path("arguments")),
                    firstNonBlank(text(item.path("call_id")), text(item.path("tool_call_id"))),
                    text(item.path("status")),
                    item));
        }
        return List.copyOf(out);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return objectMapper.convertValue(node, Map.class);
    }

    private static AgentStreamEvent.Surface detectSurface(JsonNode node) {
        String object = text(node.path("object"));
        if ("chat.completion".equals(object) || "chat.completion.chunk".equals(object) || node.has("choices")) {
            return AgentStreamEvent.Surface.CHAT_COMPLETIONS;
        }
        if ("response".equals(object) || node.has("output") || node.has("output_text")) {
            return AgentStreamEvent.Surface.RESPONSES;
        }
        return AgentStreamEvent.Surface.UNKNOWN;
    }

    private static void appendContentText(List<String> out, JsonNode content) {
        if (!content.isArray()) {
            return;
        }
        for (JsonNode item : content) {
            String text = firstNonBlank(text(item.path("text")), text(item.path("output_text")));
            if (text != null) {
                out.add(text);
            }
        }
    }

    private static JsonNode first(JsonNode node) {
        if (node != null && node.isArray() && node.size() > 0) {
            return node.get(0);
        }
        return com.fasterxml.jackson.databind.node.MissingNode.getInstance();
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

    private static int integer(JsonNode node, int fallback) {
        return node != null && node.isNumber() ? node.asInt() : fallback;
    }

    private static long longValue(JsonNode node, long fallback) {
        return node != null && node.isNumber() ? node.asLong() : fallback;
    }
}
