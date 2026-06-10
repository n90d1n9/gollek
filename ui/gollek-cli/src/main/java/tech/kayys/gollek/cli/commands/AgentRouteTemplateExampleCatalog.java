package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Describes local-only starter route templates for agent-serving builders and documentation.
 */
final class AgentRouteTemplateExampleCatalog {
    private static final String OBJECT = "gollek.agent_route_template_examples";
    private static final String VERSION = "v1";
    private static final String TEMPLATE_CHECK_COMMAND =
            "gollek agent template-check --request-file <file> --json";
    private static final List<Entry> ENTRIES = List.of(
            new Entry(
                    "chat",
                    "Chat Completions route with system prompt, tool-schema validation, and caller-owned RAG context.",
                    "chat",
                    AgentServingFeatureProfile.CHAT_AGENT,
                    "gollek agent template-example --id chat --json"),
            new Entry(
                    "responses",
                    "Responses route for agent platforms that prefer a single input field with tool schemas.",
                    "responses",
                    AgentServingFeatureProfile.RESPONSES_AGENT,
                    "gollek agent template-example --id responses --json"),
            new Entry(
                    "embedding_rag",
                    "Embeddings route for caller-owned vector storage and retrieval outside Gollek.",
                    "embeddings",
                    AgentServingFeatureProfile.EMBEDDING_RAG,
                    "gollek agent template-example --id embedding_rag --json"),
            new Entry(
                    "mcp_tools",
                    "MCP-aware route that discovers and validates tool definitions without executing tools.",
                    "chat",
                    AgentServingFeatureProfile.MCP_TOOLS,
                    "gollek agent template-example --id mcp_tools --json"));

    private AgentRouteTemplateExampleCatalog() {
    }

    static List<Entry> entries() {
        return ENTRIES;
    }

    static Optional<Entry> find(String id) {
        String normalizedId = normalizeId(id);
        if (normalizedId == null) {
            return Optional.empty();
        }
        return ENTRIES.stream()
                .filter(entry -> entry.id().equals(normalizedId))
                .findFirst();
    }

    static List<String> supportedIds() {
        return ENTRIES.stream()
                .map(Entry::id)
                .toList();
    }

    static Map<String, Object> toMetadata() {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("object", OBJECT);
        catalog.put("version", VERSION);
        catalog.put("examples", ENTRIES.stream()
                .map(Entry::toMetadata)
                .toList());
        catalog.put("boundary", boundary());
        return catalog;
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("local_only", true);
        boundary.put("http_request", false);
        boundary.put("model_invoked", false);
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        boundary.put("workflow_state", false);
        return boundary;
    }

    private static String normalizeId(String id) {
        if (id == null || id.isBlank()) {
            return null;
        }
        return id.trim()
                .toLowerCase(Locale.ROOT)
                .replace('-', '_');
    }

    /**
     * One generated route-template example exposed by the CLI catalog.
     */
    record Entry(String id, String description, String surface, String featureProfile, String command) {
        Map<String, Object> toMetadata() {
            Map<String, Object> example = new LinkedHashMap<>();
            example.put("id", id);
            example.put("description", description);
            example.put("surface", surface);
            example.put("feature_profile", featureProfile);
            example.put("command", command);
            example.put("template_check_command", TEMPLATE_CHECK_COMMAND);
            return example;
        }
    }
}
