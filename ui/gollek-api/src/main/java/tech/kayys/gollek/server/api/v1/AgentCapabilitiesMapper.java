package tech.kayys.gollek.server.api.v1;

import tech.kayys.gollek.client.agent.AgentServingFeatureCatalog;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class AgentCapabilitiesMapper {

    private AgentCapabilitiesMapper() {
    }

    static Map<String, Object> capabilities() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", AgentServingFeatureCatalog.CAPABILITIES_OBJECT);
        payload.put("version", AgentServingFeatureCatalog.CONTRACT_VERSION);
        payload.put("contract_version", AgentServingFeatureCatalog.CONTRACT_VERSION);
        payload.put("supported_contract_versions", AgentServingFeatureCatalog.SUPPORTED_CONTRACT_VERSIONS);
        payload.put("service_role", AgentServingFeatureCatalog.SERVICE_ROLE_INFERENCE_SERVING_ENGINE);
        payload.put("feature_negotiation", AgentServingFeatureCatalog.featureNegotiation());
        payload.put("agent_boundary", Map.of(
                "gollek_owns", List.of(
                        "model serving",
                        "provider routing",
                        "system prompts",
                        "tool-call request and response schema",
                        "embedding generation",
                        "RAG context injection",
                        "MCP server registry and tool definitions"),
                "agent_orchestrator_owns", List.of(
                        "planning",
                        "memory policy",
                        "tool authorization",
                        "tool execution loops",
                        "workflow state")));
        payload.put("compatibility", AgentServingFeatureCatalog.compatibilityFeatures());
        payload.put("endpoints", Map.ofEntries(
                Map.entry("openai_chat_completions", "/v1/chat/completions"),
                Map.entry("openai_chat_streaming", "/v1/chat/completions"),
                Map.entry("openai_responses", "/v1/responses"),
                Map.entry("openai_responses_streaming", "/v1/responses"),
                Map.entry("openai_embeddings", "/v1/embeddings"),
                Map.entry("openai_models", "/v1/models?compat=openai"),
                Map.entry("model_capabilities", "/v1/models/{id}/capabilities"),
                Map.entry("agent_contract", "/v1/agent/contract"),
                Map.entry("agent_readiness_issues", "/v1/agent/readiness/issues"),
                Map.entry("agent_preflight", "/v1/agent/preflight"),
                Map.entry("agent_validation", "/v1/agent/validate"),
                Map.entry("agent_tool_validation", "/v1/agent/tools/validate"),
                Map.entry("mcp_servers", "/v1/mcp/servers"),
                Map.entry("mcp_tools", "/v1/mcp/tools"),
                Map.entry("mcp_server_tools", "/v1/mcp/servers/{name}/tools"),
                Map.entry("native_completions", "/v1/completions"),
                Map.entry("native_streaming", "/v1/completions/stream"),
                Map.entry("models", "/v1/models"),
                Map.entry("providers", "/v1/providers"),
                Map.entry("health", "/health")));
        payload.put("auth", Map.of(
                "x_api_key_header", "X-API-Key",
                "authorization_header", "Bearer token"));
        payload.put("traceability", Map.of(
                "request_id_header", AgentTraceContext.REQUEST_ID_HEADER,
                "trace_id_header", AgentTraceContext.TRACE_ID_HEADER,
                "session_id_header", AgentTraceContext.SESSION_ID_HEADER,
                "user_id_header", AgentTraceContext.USER_ID_HEADER,
                "metadata_key", "gollek_trace"));
        return payload;
    }
}
