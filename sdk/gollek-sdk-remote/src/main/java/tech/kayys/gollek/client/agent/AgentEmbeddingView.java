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
 * Read-only view over OpenAI-compatible and native Gollek embedding responses.
 *
 * <p>This helper extracts vectors, dimensions, usage, metadata, and trace
 * details for caller-owned RAG pipelines. It does not execute retrieval or own
 * vector-store state.
 */
public final class AgentEmbeddingView {
    private final ObjectMapper objectMapper;
    private final JsonNode raw;

    private AgentEmbeddingView(ObjectMapper objectMapper, JsonNode raw) {
        this.objectMapper = objectMapper != null ? objectMapper : new ObjectMapper();
        this.raw = raw == null ? this.objectMapper.getNodeFactory().objectNode() : raw;
    }

    public static AgentEmbeddingView from(Map<String, Object> response) {
        return from(response, null);
    }

    public static AgentEmbeddingView from(Map<String, Object> response, ObjectMapper objectMapper) {
        ObjectMapper mapper = objectMapper != null ? objectMapper : new ObjectMapper();
        JsonNode node = response == null
                ? mapper.getNodeFactory().objectNode()
                : mapper.convertValue(response, JsonNode.class);
        return new AgentEmbeddingView(mapper, node);
    }

    public static AgentEmbeddingView from(JsonNode response) {
        return from(response, null);
    }

    public static AgentEmbeddingView from(JsonNode response, ObjectMapper objectMapper) {
        return new AgentEmbeddingView(objectMapper, response);
    }

    public static AgentEmbeddingView fromJson(String responseJson) throws JsonProcessingException {
        return fromJson(responseJson, null);
    }

    public static AgentEmbeddingView fromJson(String responseJson, ObjectMapper objectMapper)
            throws JsonProcessingException {
        ObjectMapper mapper = objectMapper != null ? objectMapper : new ObjectMapper();
        return new AgentEmbeddingView(mapper, mapper.readTree(responseJson));
    }

    public String object() {
        return text(raw.path("object"));
    }

    public String requestId() {
        return firstNonBlank(text(raw.path("request_id")), text(raw.path("requestId")));
    }

    public String model() {
        return text(raw.path("model"));
    }

    public boolean hasEmbeddings() {
        return !embeddings().isEmpty();
    }

    public int count() {
        return embeddings().size();
    }

    public int dimensions() {
        int reported = integer(raw.path("dimension"), 0);
        if (reported > 0) {
            return reported;
        }
        List<EmbeddingItem> items = embeddings();
        return items.isEmpty() ? 0 : items.get(0).dimensions();
    }

    public List<EmbeddingItem> embeddings() {
        List<EmbeddingItem> out = new ArrayList<>();
        addEmbeddingArray(out, raw.path("data"));
        if (!out.isEmpty()) {
            return List.copyOf(out);
        }

        addEmbeddingArray(out, raw.path("embeddings"));
        addEmbedding(out, raw.path("embedding"), 0);
        addEmbedding(out, raw.path("vector"), 0);
        return List.copyOf(out);
    }

    public List<List<Double>> vectors() {
        List<List<Double>> out = new ArrayList<>();
        for (EmbeddingItem item : embeddings()) {
            out.add(item.embedding());
        }
        return List.copyOf(out);
    }

    public List<Double> firstVector() {
        List<EmbeddingItem> items = embeddings();
        return items.isEmpty() ? List.of() : items.get(0).embedding();
    }

    public AgentStreamEvent.Usage usage() {
        JsonNode node = raw.path("usage");
        if (!node.isObject()) {
            return null;
        }
        long promptTokens = longValue(node.path("prompt_tokens"), longValue(node.path("input_tokens"), 0));
        long totalTokens = longValue(node.path("total_tokens"), promptTokens);
        return new AgentStreamEvent.Usage(promptTokens, 0, totalTokens);
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

    public JsonNode raw() {
        return raw;
    }

    public record EmbeddingItem(
            int index,
            String object,
            List<Double> embedding,
            Map<String, Object> metadata,
            JsonNode raw) {

        public EmbeddingItem {
            embedding = embedding == null ? List.of() : List.copyOf(embedding);
            metadata = metadata == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }

        public List<Double> vector() {
            return embedding;
        }

        public int dimensions() {
            return embedding.size();
        }

        public boolean isEmpty() {
            return embedding.isEmpty();
        }
    }

    private void addEmbeddingArray(List<EmbeddingItem> out, JsonNode node) {
        if (!node.isArray()) {
            return;
        }
        for (int i = 0; i < node.size(); i++) {
            addEmbedding(out, node.get(i), i);
        }
    }

    private void addEmbedding(List<EmbeddingItem> out, JsonNode node, int fallbackIndex) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return;
        }
        JsonNode vector = node;
        String object = "embedding";
        Map<String, Object> metadata = Map.of();
        int index = fallbackIndex;
        if (node.isObject()) {
            vector = firstPresent(node, "embedding", "vector");
            object = firstNonBlank(text(node.path("object")), object);
            index = integer(node.path("index"), fallbackIndex);
            metadata = map(node.path("metadata"));
        }
        if (!vector.isArray()) {
            return;
        }
        out.add(new EmbeddingItem(index, object, numbers(vector), metadata, node));
    }

    private static JsonNode firstPresent(JsonNode node, String... fields) {
        for (String field : fields) {
            JsonNode value = node.path(field);
            if (!value.isMissingNode() && !value.isNull()) {
                return value;
            }
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

    private static List<Double> numbers(JsonNode node) {
        List<Double> out = new ArrayList<>();
        for (JsonNode item : node) {
            if (item.isNumber()) {
                out.add(item.asDouble());
            }
        }
        return List.copyOf(out);
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
