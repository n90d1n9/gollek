package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed event emitted by OpenAI-compatible Gollek agent streams.
 */
public record AgentStreamEvent(
        Surface surface,
        String type,
        int sequenceNumber,
        String id,
        String responseId,
        String itemId,
        String role,
        String delta,
        String text,
        String outputText,
        String finishReason,
        Usage usage,
        Map<String, Object> metadata,
        Map<String, Object> trace,
        String errorMessage,
        JsonNode raw) {

    public enum Surface {
        CHAT_COMPLETIONS,
        RESPONSES,
        UNKNOWN
    }

    public record Usage(long promptTokens, long completionTokens, long totalTokens) {
    }

    public record ToolCall(
            String id,
            int index,
            String type,
            String name,
            String arguments,
            String callId,
            String status,
            JsonNode raw) {

        public boolean hasArguments() {
            return arguments != null && !arguments.isBlank();
        }

        /**
         * Parses the tool-call arguments as JSON using a default mapper.
         */
        public JsonNode argumentsJson() throws JsonProcessingException {
            return argumentsJson(null);
        }

        /**
         * Parses the tool-call arguments as JSON.
         */
        public JsonNode argumentsJson(ObjectMapper objectMapper) throws JsonProcessingException {
            ObjectMapper mapper = mapper(objectMapper);
            if (!hasArguments()) {
                return mapper.getNodeFactory().objectNode();
            }
            return mapper.readTree(arguments);
        }

        /**
         * Parses object-shaped tool-call arguments into a map using a default mapper.
         */
        public Map<String, Object> argumentsMap() throws JsonProcessingException {
            return argumentsMap(null);
        }

        /**
         * Parses object-shaped tool-call arguments into a map.
         */
        public Map<String, Object> argumentsMap(ObjectMapper objectMapper) throws JsonProcessingException {
            ObjectMapper mapper = mapper(objectMapper);
            JsonNode node = argumentsJson(mapper);
            if (!node.isObject()) {
                throw new IllegalStateException("Tool-call arguments must be a JSON object");
            }
            return mapper.convertValue(node, MAP_TYPE);
        }
    }

    private static final TypeReference<LinkedHashMap<String, Object>> MAP_TYPE = new TypeReference<>() {
    };

    public boolean hasDelta() {
        return delta != null && !delta.isBlank();
    }

    public boolean hasToolCalls() {
        return !toolCalls().isEmpty();
    }

    public List<ToolCall> toolCalls() {
        if (raw == null || raw.isMissingNode() || raw.isNull()) {
            return List.of();
        }
        if (surface == Surface.CHAT_COMPLETIONS) {
            return chatToolCalls(raw);
        }
        if (surface == Surface.RESPONSES) {
            return responseToolCalls(raw);
        }
        return List.of();
    }

    public boolean isOutputDone() {
        return "response.output_text.done".equals(type);
    }

    public boolean isCompleted() {
        return "response.completed".equals(type) || finishReason != null;
    }

    public boolean isError() {
        return errorMessage != null && !errorMessage.isBlank();
    }

    private static List<ToolCall> chatToolCalls(JsonNode root) {
        JsonNode choice = first(root.path("choices"));
        JsonNode calls = choice.path("delta").path("tool_calls");
        if (!calls.isArray()) {
            calls = choice.path("message").path("tool_calls");
        }
        return toolCallArray(calls);
    }

    private static List<ToolCall> responseToolCalls(JsonNode root) {
        List<ToolCall> calls = new ArrayList<>();
        addResponseToolCall(calls, root);

        JsonNode item = root.path("item");
        if (item.isObject()) {
            addResponseToolCall(calls, item);
        }

        JsonNode responseOutput = root.path("response").path("output");
        if (responseOutput.isArray()) {
            for (JsonNode outputItem : responseOutput) {
                addResponseToolCall(calls, outputItem);
            }
        }

        JsonNode directCalls = root.path("tool_calls");
        if (directCalls.isArray()) {
            calls.addAll(toolCallArray(directCalls));
        }
        return List.copyOf(calls);
    }

    private static List<ToolCall> toolCallArray(JsonNode calls) {
        if (!calls.isArray()) {
            return List.of();
        }
        List<ToolCall> out = new ArrayList<>();
        for (JsonNode call : calls) {
            JsonNode function = call.path("function");
            out.add(new ToolCall(
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

    private static void addResponseToolCall(List<ToolCall> out, JsonNode item) {
        String type = text(item.path("type"));
        if (!isResponseToolCall(type)) {
            return;
        }
        if (("response.output_item.added".equals(type) || "response.output_item.done".equals(type))
                && item.path("item").isObject()) {
            return;
        }
        out.add(new ToolCall(
                firstNonBlank(text(item.path("id")), text(item.path("item_id"))),
                integer(item.path("output_index"), out.size()),
                type,
                text(item.path("name")),
                firstNonBlank(text(item.path("arguments")), text(item.path("delta"))),
                firstNonBlank(text(item.path("call_id")), text(item.path("tool_call_id"))),
                text(item.path("status")),
                item));
    }

    private static boolean isResponseToolCall(String type) {
        return "function_call".equals(type)
                || "custom_tool_call".equals(type)
                || "response.output_item.added".equals(type)
                || "response.output_item.done".equals(type)
                || "response.function_call_arguments.delta".equals(type)
                || "response.function_call_arguments.done".equals(type);
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

    private static ObjectMapper mapper(ObjectMapper objectMapper) {
        return objectMapper != null ? objectMapper : new ObjectMapper();
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
}
