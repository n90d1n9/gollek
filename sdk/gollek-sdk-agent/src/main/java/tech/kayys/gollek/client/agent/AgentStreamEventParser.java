package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Parses OpenAI-compatible Gollek SSE data payloads into stable client events.
 */
public final class AgentStreamEventParser {
    public static final String DONE = "[DONE]";

    private final ObjectMapper objectMapper;

    public AgentStreamEventParser(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper != null ? objectMapper : new ObjectMapper();
    }

    public AgentStreamEvent parse(String data) throws JsonProcessingException {
        if (data == null || data.isBlank()) {
            throw new JsonProcessingException("SSE data payload is blank") {
            };
        }
        if (DONE.equals(data.trim())) {
            return new AgentStreamEvent(
                    AgentStreamEvent.Surface.UNKNOWN,
                    DONE,
                    -1,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    Map.of(),
                    Map.of(),
                    null,
                    objectMapper.getNodeFactory().nullNode());
        }

        JsonNode root = objectMapper.readTree(data);
        if ("chat.completion.chunk".equals(text(root.path("object")))) {
            return chatEvent(root);
        }
        return responsesEvent(root);
    }

    private AgentStreamEvent chatEvent(JsonNode root) {
        JsonNode choice = first(root.path("choices"));
        JsonNode delta = choice.path("delta");
        Map<String, Object> metadata = map(root.path("metadata"));
        String finishReason = text(choice.path("finish_reason"));
        String errorMessage = text(root.path("error").path("message"));

        return new AgentStreamEvent(
                AgentStreamEvent.Surface.CHAT_COMPLETIONS,
                "chat.completion.chunk",
                streamSequenceNumber(root, metadata),
                text(root.path("id")),
                null,
                null,
                text(delta.path("role")),
                text(delta.path("content")),
                null,
                null,
                finishReason,
                usage(root.path("usage")),
                metadata,
                trace(root, metadata),
                errorMessage,
                root);
    }

    private AgentStreamEvent responsesEvent(JsonNode root) {
        String type = text(root.path("type"));
        JsonNode response = root.path("response");
        Map<String, Object> metadata = map(root.path("metadata"));
        if (metadata.isEmpty() && response.isObject()) {
            metadata = map(response.path("metadata"));
        }
        String finishReason = text(root.path("metadata").path("gollek_stream").path("finish_reason"));
        if (finishReason == null && response.isObject()) {
            finishReason = text(response.path("metadata").path("gollek_stream").path("finish_reason"));
        }

        return new AgentStreamEvent(
                AgentStreamEvent.Surface.RESPONSES,
                type != null ? type : "response.event",
                integer(root.path("sequence_number"), streamSequenceNumber(root, metadata)),
                text(response.path("id")),
                firstNonBlank(text(root.path("response_id")), text(response.path("id"))),
                text(root.path("item_id")),
                null,
                text(root.path("delta")),
                text(root.path("text")),
                text(response.path("output_text")),
                finishReason,
                usage(response.path("usage")),
                metadata,
                trace(root, metadata),
                text(root.path("error").path("message")),
                root);
    }

    private int streamSequenceNumber(JsonNode root, Map<String, Object> metadata) {
        int fallback = integer(root.path("sequence_number"), -1);
        Object stream = metadata.get("gollek_stream");
        if (stream instanceof Map<?, ?> streamMap) {
            Object value = streamMap.get("sequence_number");
            if (value instanceof Number number) {
                return number.intValue();
            }
        }
        return fallback;
    }

    private Map<String, Object> trace(JsonNode root, Map<String, Object> metadata) {
        Map<String, Object> trace = map(root.path("trace"));
        if (!trace.isEmpty()) {
            return trace;
        }
        Object metadataTrace = metadata.get("gollek_trace");
        if (metadataTrace instanceof Map<?, ?> traceMap) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : traceMap.entrySet()) {
                if (entry.getKey() instanceof String key) {
                    out.put(key, entry.getValue());
                }
            }
            return out;
        }
        return Map.of();
    }

    private AgentStreamEvent.Usage usage(JsonNode node) {
        if (node == null || !node.isObject()) {
            return null;
        }
        long promptTokens = longValue(node.path("prompt_tokens"), longValue(node.path("input_tokens"), 0));
        long completionTokens = longValue(
                node.path("completion_tokens"),
                longValue(node.path("output_tokens"), 0));
        long totalTokens = longValue(node.path("total_tokens"), promptTokens + completionTokens);
        return new AgentStreamEvent.Usage(promptTokens, completionTokens, totalTokens);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> map(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return objectMapper.convertValue(node, Map.class);
    }

    private static JsonNode first(JsonNode node) {
        if (node != null && node.isArray() && node.size() > 0) {
            return node.get(0);
        }
        return MissingNodeHolder.MISSING;
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

    private static final class MissingNodeHolder {
        private static final JsonNode MISSING = com.fasterxml.jackson.databind.node.MissingNode.getInstance();

        private MissingNodeHolder() {
        }
    }
}
