package tech.kayys.gollek.server.api.v1;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentContractMapperTest {

    @Test
    @SuppressWarnings("unchecked")
    void publishesStableAgentIntegrationContract() {
        Map<String, Object> contract = AgentContractMapper.contract();

        assertEquals("gollek.agent_contract", contract.get("object"));
        assertEquals("v1", contract.get("version"));
        assertEquals("inference_serving_engine", contract.get("service_role"));

        Map<String, Object> boundary = (Map<String, Object>) contract.get("boundary");
        assertEquals(false, boundary.get("tool_execution"));
        assertEquals(false, boundary.get("retrieval_execution"));
        assertTrue(((List<String>) boundary.get("gollek_owns")).contains("rag_context_injection"));
        assertTrue(((List<String>) boundary.get("agent_orchestrator_owns")).contains("retrieval_policy"));
        assertTrue(((List<String>) boundary.get("agent_orchestrator_owns")).contains("tool_result_loop"));

        Map<String, Object> traceability = (Map<String, Object>) contract.get("traceability");
        assertEquals(true, traceability.get("propagates_to_inference_request"));
        assertEquals("gollek_trace", traceability.get("response_metadata_key"));
        assertTrue(((List<String>) traceability.get("response_headers")).contains("X-Gollek-Request-Id"));

        Map<String, Object> streaming = (Map<String, Object>) contract.get("streaming");
        assertEquals("[DONE]", streaming.get("done_sentinel"));
        assertTrue(((List<String>) streaming.get("chat_completions_events")).contains("chat.completion.chunk"));
        assertTrue(((List<String>) streaming.get("responses_events")).contains("response.completed"));
        assertTrue(((Map<String, Object>) streaming.get("stream_options")).containsKey("include_usage"));

        Map<String, Object> endpoints = (Map<String, Object>) contract.get("endpoints");
        assertEquals("/v1/chat/completions", ((Map<String, Object>) endpoints.get("chat_completions")).get("path"));
        assertEquals("/v1/agent/contract", ((Map<String, Object>) endpoints.get("agent_contract")).get("path"));
        assertEquals("/v1/agent/validate?surface={surface}",
                ((Map<String, Object>) endpoints.get("agent_validation")).get("path"));
        assertEquals("/v1/agent/tools/validate",
                ((Map<String, Object>) endpoints.get("agent_tool_validation")).get("path"));

        Map<String, Object> schemas = (Map<String, Object>) contract.get("schemas");
        Map<String, Object> streamOptions = (Map<String, Object>) schemas.get("stream_options");
        assertTrue(((Map<String, Object>) streamOptions.get("properties")).containsKey("include_stream_metadata"));

        Map<String, Object> chatRequest = (Map<String, Object>) schemas.get("chat_completions_request");
        Map<String, Object> chatProperties = (Map<String, Object>) chatRequest.get("properties");
        assertEquals("#/schemas/stream_options",
                ((Map<String, Object>) chatProperties.get("stream_options")).get("$ref"));

        Map<String, Object> responsesRequest = (Map<String, Object>) schemas.get("responses_request");
        Map<String, Object> responsesProperties = (Map<String, Object>) responsesRequest.get("properties");
        assertEquals("#/schemas/stream_options",
                ((Map<String, Object>) responsesProperties.get("stream_options")).get("$ref"));

        Map<String, Object> ragContext = (Map<String, Object>) schemas.get("rag_context");
        assertTrue(((List<String>) ragContext.get("accepted_aliases")).contains("context_documents"));
        assertEquals("agent_orchestrator", ragContext.get("vector_store_owned_by"));

        Map<String, Object> tool = (Map<String, Object>) schemas.get("openai_tool_definition");
        assertEquals(false, tool.get("execution"));
        assertEquals("/v1/agent/tools/validate", tool.get("validation_endpoint"));

        Map<String, Object> error = (Map<String, Object>) schemas.get("error_response");
        assertTrue(((List<String>) error.get("required")).contains("error"));

        Map<String, Object> validation = (Map<String, Object>) schemas.get("agent_validation_response");
        assertTrue(((List<String>) validation.get("required")).contains("normalized"));
        assertTrue(((List<String>) validation.get("required")).contains("trace"));

        Map<String, Object> toolValidation = (Map<String, Object>) schemas.get("tool_contract_validation_response");
        assertTrue(((List<String>) toolValidation.get("required")).contains("warnings"));
    }
}
