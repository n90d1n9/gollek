package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Named serving-feature bundles for callers that do not want to hardcode every
 * feature flag.
 *
 * <p>Profiles describe Gollek serving surfaces only. They do not select an
 * agent plan, retrieval policy, tool authorization policy, or workflow state.
 */
public final class AgentServingFeatureProfile {
    public static final String AGENT_SERVING = "agent_serving";
    public static final String CHAT_AGENT = "chat_agent";
    public static final String RESPONSES_AGENT = "responses_agent";
    public static final String EMBEDDING_RAG = "embedding_rag";
    public static final String MCP_TOOLS = "mcp_tools";
    public static final String DEFAULT_PROFILE = AGENT_SERVING;

    private final String name;
    private final String summary;
    private final List<String> surfaces;
    private final List<String> requiredFeatures;
    private final List<String> optionalFeatures;

    private AgentServingFeatureProfile(
            String name,
            String summary,
            List<String> surfaces,
            List<String> requiredFeatures,
            List<String> optionalFeatures) {
        this.name = normalizeName(name);
        this.summary = summary == null || summary.isBlank() ? this.name : summary;
        this.surfaces = surfaces == null ? List.of() : List.copyOf(surfaces);
        this.requiredFeatures = distinct(requiredFeatures);
        this.optionalFeatures = distinct(optionalFeatures);
    }

    public static AgentServingFeatureProfile defaultProfile() {
        return catalog().get(0);
    }

    public static List<AgentServingFeatureProfile> catalog() {
        return List.of(
                agentServing(),
                chatAgent(),
                responsesAgent(),
                embeddingRag(),
                mcpTools());
    }

    public static Optional<AgentServingFeatureProfile> find(String name) {
        String normalized = normalizeName(name);
        if (normalized == null) {
            normalized = DEFAULT_PROFILE;
        }
        for (AgentServingFeatureProfile profile : catalog()) {
            if (profile.name().equals(normalized)) {
                return Optional.of(profile);
            }
        }
        return Optional.empty();
    }

    public static List<String> supportedProfileNames() {
        return catalog().stream()
                .map(AgentServingFeatureProfile::name)
                .toList();
    }

    public static List<Map<String, Object>> catalogMaps() {
        return catalog().stream()
                .map(AgentServingFeatureProfile::toMap)
                .toList();
    }

    public static String normalizeName(String value) {
        if (value == null || value.isBlank()) {
            return DEFAULT_PROFILE;
        }
        String normalized = value.trim()
                .replaceAll("([a-z0-9])([A-Z])", "$1_$2")
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "_")
                .replaceAll("^_|_$", "");
        return normalized.isBlank() ? DEFAULT_PROFILE : normalized;
    }

    public String name() {
        return name;
    }

    public String summary() {
        return summary;
    }

    public List<String> surfaces() {
        return surfaces;
    }

    public List<String> requiredFeatures() {
        return requiredFeatures;
    }

    public List<String> optionalFeatures() {
        return optionalFeatures;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("name", name);
        out.put("summary", summary);
        out.put("surfaces", surfaces);
        out.put("required_features", requiredFeatures);
        out.put("optional_features", optionalFeatures);
        out.put("boundary", Map.of(
                "serving_profile_only", true,
                "tool_execution", false,
                "retrieval_execution", false,
                "workflow_state", false));
        return Collections.unmodifiableMap(out);
    }

    private static AgentServingFeatureProfile agentServing() {
        return new AgentServingFeatureProfile(
                AGENT_SERVING,
                "Full Gollek agent-serving contract with chat, Responses, embeddings, MCP discovery, validation, and RAG context injection.",
                List.of("chat", "responses", "embeddings"),
                AgentServingFeatureCatalog.REQUIRED_FEATURES,
                List.of());
    }

    private static AgentServingFeatureProfile chatAgent() {
        return new AgentServingFeatureProfile(
                CHAT_AGENT,
                "Chat Completions serving profile for orchestrators that own planning, tools, memory, and retrieval policy.",
                List.of("chat"),
                concat(
                        servingCore(),
                        List.of(
                                "openai_chat_completions",
                                "openai_chat_streaming",
                                "agent_tool_contract_validation",
                                "mcp_tool_discovery",
                                "rag_context")),
                List.of(
                        "tools",
                        "tool_choice",
                        "tool_calls",
                        "system_prompt",
                        "mcp_registry",
                        "mcp_tool_definitions",
                        "stream_usage_reporting",
                        "stream_trace_metadata"));
    }

    private static AgentServingFeatureProfile responsesAgent() {
        return new AgentServingFeatureProfile(
                RESPONSES_AGENT,
                "Responses serving profile for orchestrators that use OpenAI-compatible Responses requests and streams.",
                List.of("responses"),
                concat(
                        servingCore(),
                        List.of(
                                "openai_responses",
                                "openai_responses_streaming",
                                "agent_tool_contract_validation",
                                "mcp_tool_discovery",
                                "rag_context")),
                List.of(
                        "tools",
                        "tool_choice",
                        "tool_calls",
                        "system_prompt",
                        "mcp_registry",
                        "mcp_tool_definitions",
                        "stream_usage_reporting",
                        "stream_trace_metadata"));
    }

    private static AgentServingFeatureProfile embeddingRag() {
        return new AgentServingFeatureProfile(
                EMBEDDING_RAG,
                "Embedding generation profile for caller-owned RAG indexing and retrieval.",
                List.of("embeddings"),
                concat(
                        servingCore(),
                        List.of(
                                "openai_embeddings",
                                "rag_context")),
                List.of(
                        "embeddings",
                        "models",
                        "providers"));
    }

    private static AgentServingFeatureProfile mcpTools() {
        return new AgentServingFeatureProfile(
                MCP_TOOLS,
                "MCP discovery and tool-schema validation profile without tool execution.",
                List.of("chat", "responses"),
                concat(
                        servingCore(),
                        List.of(
                                "agent_tool_contract_validation",
                                "mcp_tool_discovery")),
                List.of(
                        "tools",
                        "mcp_registry",
                        "mcp_tool_definitions"));
    }

    private static List<String> servingCore() {
        return List.of(
                "model_capability_matrix",
                "agent_capabilities",
                "agent_contract",
                "agent_feature_negotiation",
                "agent_readiness_issue_catalog",
                "agent_preflight",
                "agent_request_validation",
                "request_trace_context");
    }

    private static List<String> concat(List<String> first, List<String> second) {
        ArrayList<String> out = new ArrayList<>();
        if (first != null) {
            out.addAll(first);
        }
        if (second != null) {
            out.addAll(second);
        }
        return distinct(out);
    }

    private static List<String> distinct(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<String> out = new LinkedHashSet<>();
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                out.add(value.trim());
            }
        }
        return List.copyOf(out);
    }
}
