package tech.kayys.gollek.server.api.v1;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class AgentContractMapper {

    private AgentContractMapper() {
    }

    static Map<String, Object> contract() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("object", "gollek.agent_contract");
        payload.put("version", "v1");
        payload.put("service_role", "inference_serving_engine");
        payload.put("boundary", boundary());
        payload.put("auth", auth());
        payload.put("traceability", traceability());
        payload.put("streaming", streaming());
        payload.put("endpoints", endpoints());
        payload.put("schemas", schemas());
        return payload;
    }

    private static Map<String, Object> boundary() {
        Map<String, Object> boundary = new LinkedHashMap<>();
        boundary.put("gollek_owns", List.of(
                "model_serving",
                "provider_routing",
                "system_prompt_mapping",
                "tool_schema_ingestion",
                "tool_contract_validation",
                "mcp_registry_discovery",
                "embedding_generation",
                "rag_context_injection"));
        boundary.put("agent_orchestrator_owns", List.of(
                "planning",
                "memory_policy",
                "retrieval_policy",
                "vector_store_ownership",
                "tool_authorization",
                "tool_execution",
                "tool_result_loop",
                "workflow_state"));
        boundary.put("tool_execution", false);
        boundary.put("retrieval_execution", false);
        return boundary;
    }

    private static Map<String, Object> auth() {
        Map<String, Object> auth = new LinkedHashMap<>();
        auth.put("accepted", List.of("Authorization: Bearer <token>", "X-API-Key: <token>"));
        auth.put("scope", "serving_api");
        return auth;
    }

    private static Map<String, Object> traceability() {
        Map<String, Object> traceability = new LinkedHashMap<>();
        traceability.put("accepted_headers", Map.of(
                "request_id", List.of(AgentTraceContext.REQUEST_ID_HEADER, "X-Request-Id", "X-Correlation-Id"),
                "trace_id", List.of(AgentTraceContext.TRACE_ID_HEADER, "X-Trace-Id", "traceparent"),
                "session_id", List.of(AgentTraceContext.SESSION_ID_HEADER, "X-Session-Id", "X-Conversation-Id"),
                "user_id", List.of(AgentTraceContext.USER_ID_HEADER, "X-User-Id")));
        traceability.put("accepted_body_fields", Map.of(
                "request_id", List.of("request_id"),
                "trace_id", List.of("trace_id", "trace"),
                "session_id", List.of("session_id", "conversation", "previous_response_id"),
                "user_id", List.of("user", "user_id")));
        traceability.put("response_headers", List.of(
                AgentTraceContext.REQUEST_ID_HEADER,
                AgentTraceContext.TRACE_ID_HEADER,
                AgentTraceContext.SESSION_ID_HEADER,
                AgentTraceContext.USER_ID_HEADER));
        traceability.put("response_metadata_key", "gollek_trace");
        traceability.put("error_fields", List.of("request_id", "trace_id", "session_id", "user_id"));
        traceability.put("propagates_to_inference_request", true);
        return traceability;
    }

    private static Map<String, Object> streaming() {
        Map<String, Object> streaming = new LinkedHashMap<>();
        streaming.put("done_sentinel", "[DONE]");
        streaming.put("stream_options", Map.of(
                "include_usage", "Emit usage on final events when provider usage is available.",
                "include_trace", "Include trace metadata on stream events. Defaults to true.",
                "include_stream_metadata", "Include gollek_stream metadata on stream events. Defaults to true."));
        streaming.put("chat_completions_events", List.of(
                "chat.completion.chunk",
                "[DONE]"));
        streaming.put("responses_events", List.of(
                "response.created",
                "response.output_text.delta",
                "response.output_text.done",
                "response.completed",
                "[DONE]"));
        streaming.put("error_events", Map.of(
                "chat_completions", "OpenAI-style error chunk followed by [DONE].",
                "responses", "Responses error event followed by [DONE]."));
        return streaming;
    }

    private static Map<String, Object> endpoints() {
        Map<String, Object> endpoints = new LinkedHashMap<>();
        endpoints.put("chat_completions", endpoint("POST", "/v1/chat/completions", "chat_completions_request"));
        endpoints.put("responses", endpoint("POST", "/v1/responses", "responses_request"));
        endpoints.put("embeddings", endpoint("POST", "/v1/embeddings", "embeddings_request"));
        endpoints.put("models", endpoint("GET", "/v1/models", null));
        endpoints.put("openai_models", endpoint("GET", "/v1/models?compat=openai", null));
        endpoints.put("model_capabilities", endpoint("GET", "/v1/models/{id}/capabilities", null));
        endpoints.put("mcp_servers", endpoint("GET", "/v1/mcp/servers", null));
        endpoints.put("mcp_tools", endpoint("GET", "/v1/mcp/tools?compat=openai", "openai_tool_definition"));
        endpoints.put("agent_capabilities", endpoint("GET", "/v1/agent/capabilities", null));
        endpoints.put("agent_contract", endpoint("GET", "/v1/agent/contract", null));
        endpoints.put("agent_validation", endpoint("POST", "/v1/agent/validate?surface={surface}", "agent_validation_request"));
        endpoints.put("agent_tool_validation", endpoint("POST", "/v1/agent/tools/validate", "tool_contract_validation_request"));
        return endpoints;
    }

    private static Map<String, Object> endpoint(String method, String path, String requestSchema) {
        Map<String, Object> endpoint = new LinkedHashMap<>();
        endpoint.put("method", method);
        endpoint.put("path", path);
        if (requestSchema != null) {
            endpoint.put("schema", requestSchema);
        }
        return endpoint;
    }

    private static Map<String, Object> schemas() {
        Map<String, Object> schemas = new LinkedHashMap<>();
        schemas.put("chat_completions_request", chatCompletionsRequest());
        schemas.put("responses_request", responsesRequest());
        schemas.put("embeddings_request", embeddingsRequest());
        schemas.put("message", message());
        schemas.put("stream_options", streamOptions());
        schemas.put("openai_tool_definition", openAiToolDefinition());
        schemas.put("rag_context", ragContext());
        schemas.put("rag_context_item", ragContextItem());
        schemas.put("error_response", errorResponse());
        schemas.put("agent_validation_request", agentValidationRequest());
        schemas.put("agent_validation_response", agentValidationResponse());
        schemas.put("tool_contract_validation_request", toolContractValidationRequest());
        schemas.put("tool_contract_validation_response", toolContractValidationResponse());
        return schemas;
    }

    private static Map<String, Object> chatCompletionsRequest() {
        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("model", "messages"));
        schema.put("properties", Map.ofEntries(
                Map.entry("model", string("Model id or alias served by Gollek.")),
                Map.entry("request_id", string("Caller-supplied request correlation id.")),
                Map.entry("trace_id", string("Distributed trace id propagated to request metadata.")),
                Map.entry("session_id", string("Conversation/session continuity id.")),
                Map.entry("user", string("End-user id supplied by the agent orchestrator.")),
                Map.entry("messages", arrayOf(ref("message"))),
                Map.entry("stream", booleanSchema("Return server-sent event chunks when true.")),
                Map.entry("stream_options", ref("stream_options")),
                Map.entry("tools", arrayOf(ref("openai_tool_definition"))),
                Map.entry("tool_choice", anyOf(List.of(string(null), objectSchema()))),
                Map.entry("rag_context", ref("rag_context")),
                Map.entry("retrieval_context", ref("rag_context")),
                Map.entry("context_documents", ref("rag_context")),
                Map.entry("embedding_model", string("Optional embedding model hint for upstream RAG.")),
                Map.entry("metadata", objectSchema())));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> responsesRequest() {
        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("model", "input"));
        schema.put("properties", Map.ofEntries(
                Map.entry("model", string("Model id or alias served by Gollek.")),
                Map.entry("request_id", string("Caller-supplied request correlation id.")),
                Map.entry("trace_id", string("Distributed trace id propagated to request metadata.")),
                Map.entry("user", string("End-user id supplied by the agent orchestrator.")),
                Map.entry("instructions", string("System/developer instruction text.")),
                Map.entry("input", anyOf(List.of(string(null), arrayOf(objectSchema()), objectSchema()))),
                Map.entry("stream", booleanSchema("Return OpenAI Responses-compatible SSE events when true.")),
                Map.entry("stream_options", ref("stream_options")),
                Map.entry("tools", arrayOf(ref("openai_tool_definition"))),
                Map.entry("tool_choice", anyOf(List.of(string(null), objectSchema()))),
                Map.entry("previous_response_id", string("Conversation/session continuity id.")),
                Map.entry("conversation", string("Conversation/session id.")),
                Map.entry("rag_context", ref("rag_context")),
                Map.entry("retrieval_context", ref("rag_context")),
                Map.entry("context_documents", ref("rag_context")),
                Map.entry("metadata", objectSchema())));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> embeddingsRequest() {
        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("model"));
        schema.put("properties", Map.ofEntries(
                Map.entry("model", string("Embedding model id or alias served by Gollek.")),
                Map.entry("request_id", string("Caller-supplied request correlation id.")),
                Map.entry("trace_id", string("Distributed trace id propagated to request metadata.")),
                Map.entry("user", string("End-user id supplied by the agent orchestrator.")),
                Map.entry("input", anyOf(List.of(string(null), arrayOf(string(null))))),
                Map.entry("inputs", anyOf(List.of(string(null), arrayOf(string(null))))),
                Map.entry("dimensions", integer("Optional output dimension request.")),
                Map.entry("encoding_format", string("float or base64.")),
                Map.entry("metadata", objectSchema())));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> message() {
        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("role", "content"));
        schema.put("properties", Map.ofEntries(
                Map.entry("role", enumSchema(List.of("system", "developer", "user", "assistant", "tool", "function"))),
                Map.entry("content", anyOf(List.of(string(null), arrayOf(objectSchema()), objectSchema()))),
                Map.entry("name", string("Optional participant name.")),
                Map.entry("tool_call_id", string("Tool call id for tool result messages.")),
                Map.entry("tool_calls", arrayOf(objectSchema()))));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> streamOptions() {
        Map<String, Object> schema = objectSchema();
        schema.put("properties", Map.ofEntries(
                Map.entry("include_usage", booleanSchema("Include usage on final stream events when provider usage is available.")),
                Map.entry("include_trace", booleanSchema("Include request trace metadata on stream events.")),
                Map.entry("include_stream_metadata", booleanSchema("Include gollek_stream sequence metadata on stream events."))));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> openAiToolDefinition() {
        Map<String, Object> function = objectSchema();
        function.put("required", List.of("name"));
        function.put("properties", Map.ofEntries(
                Map.entry("name", string("Stable function/tool name.")),
                Map.entry("description", string("Human-readable tool purpose.")),
                Map.entry("parameters", objectSchema()),
                Map.entry("strict", booleanSchema("Whether arguments must strictly match parameters."))));

        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("type", "function"));
        schema.put("properties", Map.ofEntries(
                Map.entry("type", enumSchema(List.of("function", "mcp_tool", "code_interpreter", "file_search"))),
                Map.entry("function", function),
                Map.entry("x_gollek", objectSchema()),
                Map.entry("metadata", objectSchema())));
        schema.put("execution", false);
        schema.put("validation_endpoint", "/v1/agent/tools/validate");
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> ragContext() {
        Map<String, Object> schema = anyOf(List.of(
                string("Plain retrieved context."),
                arrayOf(ref("rag_context_item")),
                objectSchema()));
        schema.put("accepted_aliases", List.of(
                "rag_context",
                "retrieval_context",
                "retrieved_context",
                "context_documents",
                "retrieved_documents"));
        schema.put("retrieval_policy_owned_by", "agent_orchestrator");
        schema.put("vector_store_owned_by", "agent_orchestrator");
        return schema;
    }

    private static Map<String, Object> ragContextItem() {
        Map<String, Object> schema = objectSchema();
        schema.put("properties", Map.ofEntries(
                Map.entry("text", string("Retrieved text chunk.")),
                Map.entry("content", string("Alias for retrieved text chunk.")),
                Map.entry("page_content", string("LangChain-style retrieved text chunk.")),
                Map.entry("id", string("Stable source/chunk id.")),
                Map.entry("chunk_id", string("Stable source/chunk id alias.")),
                Map.entry("source", string("URI, path, or source label.")),
                Map.entry("title", string("Human-readable source title.")),
                Map.entry("score", number("Retriever relevance score.")),
                Map.entry("metadata", objectSchema())));
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> errorResponse() {
        Map<String, Object> error = objectSchema();
        error.put("required", List.of("message", "type"));
        error.put("properties", Map.ofEntries(
                Map.entry("message", string("Human-readable error detail.")),
                Map.entry("type", enumSchema(List.of(
                        AgentErrorMapper.INVALID_REQUEST,
                        AgentErrorMapper.UNSUPPORTED_STREAMING_ACCEPT,
                        AgentErrorMapper.SERVER_ERROR))),
                Map.entry("request_id", string("Request correlation id.")),
                Map.entry("trace_id", string("Distributed trace id.")),
                Map.entry("session_id", string("Session id when supplied.")),
                Map.entry("user_id", string("User id when supplied."))));

        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("error"));
        schema.put("properties", Map.of("error", error));
        return schema;
    }

    private static Map<String, Object> agentValidationRequest() {
        Map<String, Object> schema = objectSchema();
        schema.put("description", "Pass a chat, Responses, or embeddings request body and select surface via query.");
        schema.put("surfaces", List.of("chat", "responses", "embeddings"));
        schema.put("model_invoked", false);
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> agentValidationResponse() {
        Map<String, Object> normalized = objectSchema();
        normalized.put("additionalProperties", true);

        Map<String, Object> boundary = objectSchema();
        boundary.put("properties", Map.ofEntries(
                Map.entry("validation_only", booleanSchema("Always true for this endpoint.")),
                Map.entry("tool_execution", booleanSchema("Always false for this endpoint.")),
                Map.entry("retrieval_execution", booleanSchema("Always false for this endpoint."))));

        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of("object", "surface", "valid", "model_invoked", "trace", "normalized", "boundary"));
        schema.put("properties", Map.ofEntries(
                Map.entry("object", string("gollek.agent_validation")),
                Map.entry("surface", enumSchema(List.of("chat", "responses", "embeddings"))),
                Map.entry("valid", booleanSchema("Whether the payload maps successfully.")),
                Map.entry("model_invoked", booleanSchema("Always false for validation.")),
                Map.entry("trace", objectSchema()),
                Map.entry("normalized", normalized),
                Map.entry("boundary", boundary)));
        return schema;
    }

    private static Map<String, Object> toolContractValidationRequest() {
        Map<String, Object> schema = objectSchema();
        schema.put("description", "Pass either an array of OpenAI-compatible tool definitions or an object with tools.");
        schema.put("properties", Map.of(
                "tools", arrayOf(ref("openai_tool_definition"))));
        schema.put("tool_execution", false);
        schema.put("additionalProperties", true);
        return schema;
    }

    private static Map<String, Object> toolContractValidationResponse() {
        Map<String, Object> boundary = objectSchema();
        boundary.put("properties", Map.ofEntries(
                Map.entry("validation_only", booleanSchema("Always true for this endpoint.")),
                Map.entry("tool_execution", booleanSchema("Always false for this endpoint.")),
                Map.entry("tool_authorization", booleanSchema("Always false for this endpoint."))));

        Map<String, Object> schema = objectSchema();
        schema.put("required", List.of(
                "object",
                "valid",
                "model_invoked",
                "tool_count",
                "normalized",
                "warnings",
                "boundary"));
        schema.put("properties", Map.ofEntries(
                Map.entry("object", string("gollek.tool_contract_validation")),
                Map.entry("valid", booleanSchema("Whether all tool definitions map successfully.")),
                Map.entry("model_invoked", booleanSchema("Always false for validation.")),
                Map.entry("tool_count", integer("Number of normalized tools.")),
                Map.entry("normalized", arrayOf(objectSchema())),
                Map.entry("warnings", arrayOf(objectSchema())),
                Map.entry("boundary", boundary)));
        return schema;
    }

    private static Map<String, Object> objectSchema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        return schema;
    }

    private static Map<String, Object> string(String description) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "string");
        if (description != null) {
            schema.put("description", description);
        }
        return schema;
    }

    private static Map<String, Object> number(String description) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "number");
        schema.put("description", description);
        return schema;
    }

    private static Map<String, Object> integer(String description) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "integer");
        schema.put("description", description);
        return schema;
    }

    private static Map<String, Object> booleanSchema(String description) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "boolean");
        schema.put("description", description);
        return schema;
    }

    private static Map<String, Object> arrayOf(Map<String, Object> items) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "array");
        schema.put("items", items);
        return schema;
    }

    private static Map<String, Object> enumSchema(List<String> values) {
        Map<String, Object> schema = string(null);
        schema.put("enum", values);
        return schema;
    }

    private static Map<String, Object> ref(String name) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("$ref", "#/schemas/" + name);
        return schema;
    }

    private static Map<String, Object> anyOf(List<Map<String, Object>> options) {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("anyOf", options);
        return schema;
    }
}
