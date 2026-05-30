package tech.kayys.gollek.server.api.v1;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.List;
import java.util.Map;

@Path("/v1/agent")
@Tag(name = "Agentic AI", description = "OpenAI-compatible serving endpoints for agent integrations")
public class AgentCapabilitiesResource {

    @GET
    @Path("/capabilities")
    @Produces(MediaType.APPLICATION_JSON)
    @Operation(
            operationId = "getAgentCapabilities",
            summary = "Get agent-facing service capabilities",
            description = "Summarizes Gollek's serving-engine role, supported compatibility surfaces, "
                    + "and endpoint discovery links.")
    @APIResponse(responseCode = "200", description = "Agent-facing capability discovery")
    public Response capabilities() {
        return Response.ok(Map.of(
                "service_role", "inference_serving_engine",
                "agent_boundary", Map.of(
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
                                "workflow state")),
                "compatibility", List.of(
                        "openai_chat_completions",
                        "openai_chat_streaming",
                        "openai_responses",
                        "openai_responses_streaming",
                        "openai_embeddings",
                        "openai_models",
                        "model_capability_matrix",
                        "agent_contract",
                        "agent_request_validation",
                        "agent_tool_contract_validation",
                        "request_trace_context",
                        "stream_options",
                        "stream_usage_reporting",
                        "stream_trace_metadata",
                        "native_gollek_completions",
                        "native_gollek_streaming",
                        "tools",
                        "tool_choice",
                        "tool_calls",
                        "system_prompt",
                        "mcp_registry",
                        "mcp_tool_discovery",
                        "mcp_tool_definitions",
                        "embeddings",
                        "rag_context",
                        "sessions",
                        "models",
                        "providers"),
                "endpoints", Map.ofEntries(
                        Map.entry("openai_chat_completions", "/v1/chat/completions"),
                        Map.entry("openai_chat_streaming", "/v1/chat/completions"),
                        Map.entry("openai_responses", "/v1/responses"),
                        Map.entry("openai_responses_streaming", "/v1/responses"),
                        Map.entry("openai_embeddings", "/v1/embeddings"),
                        Map.entry("openai_models", "/v1/models?compat=openai"),
                        Map.entry("model_capabilities", "/v1/models/{id}/capabilities"),
                        Map.entry("agent_contract", "/v1/agent/contract"),
                        Map.entry("agent_validation", "/v1/agent/validate"),
                        Map.entry("agent_tool_validation", "/v1/agent/tools/validate"),
                        Map.entry("mcp_servers", "/v1/mcp/servers"),
                        Map.entry("mcp_tools", "/v1/mcp/tools"),
                        Map.entry("mcp_server_tools", "/v1/mcp/servers/{name}/tools"),
                        Map.entry("native_completions", "/v1/completions"),
                        Map.entry("native_streaming", "/v1/completions/stream"),
                        Map.entry("models", "/v1/models"),
                        Map.entry("providers", "/v1/providers"),
                        Map.entry("health", "/health")),
                "auth", Map.of(
                        "x_api_key_header", "X-API-Key",
                        "authorization_header", "Bearer token"),
                "traceability", Map.of(
                        "request_id_header", AgentTraceContext.REQUEST_ID_HEADER,
                        "trace_id_header", AgentTraceContext.TRACE_ID_HEADER,
                        "session_id_header", AgentTraceContext.SESSION_ID_HEADER,
                        "user_id_header", AgentTraceContext.USER_ID_HEADER,
                        "metadata_key", "gollek_trace"))).build();
    }
}
