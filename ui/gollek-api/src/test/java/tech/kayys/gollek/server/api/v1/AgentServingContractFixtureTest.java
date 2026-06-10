package tech.kayys.gollek.server.api.v1;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.ws.rs.core.Cookie;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.MultivaluedHashMap;
import jakarta.ws.rs.core.MultivaluedMap;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.client.agent.AgentServingFeatureProfile;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingContractFixtureTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final JsonNode FIXTURE = loadFixture();
    private static final HttpHeaders HEADERS = new FixtureHeaders(Map.of("X-API-Key", "fixture-api-key"));

    @Test
    void agentContractSatisfiesRequiredFixture() {
        Map<String, Object> contract = AgentContractMapper.contract();
        JsonNode required = FIXTURE.path("required");

        assertEquals(required.path("version").asText(), contract.get("version"));
        assertEquals(required.path("version").asText(), contract.get("contract_version"));
        assertEquals(List.of(required.path("version").asText()), contract.get("supported_contract_versions"));
        assertEquals(required.path("service_role").asText(), contract.get("service_role"));
        assertContainsAll(list(contract.get("compatibility")), strings(required.path("compatibility")));
        Map<String, Object> featureNegotiation = map(contract.get("feature_negotiation"));
        assertEquals(required.path("version").asText(), featureNegotiation.get("contract_version"));
        assertContainsAll(list(featureNegotiation.get("required_features")), strings(required.path("compatibility")));

        Map<String, Object> boundary = map(contract.get("boundary"));
        assertContainsAll(list(boundary.get("gollek_owns")), strings(required.path("gollek_owns")));
        assertContainsAll(list(boundary.get("agent_orchestrator_owns")), strings(required.path("orchestrator_owns")));
        assertEquals(false, boundary.get("tool_execution"));
        assertEquals(false, boundary.get("retrieval_execution"));

        JsonNode expectedEndpoints = required.path("endpoints");
        Map<String, Object> endpoints = map(contract.get("endpoints"));
        expectedEndpoints.fields().forEachRemaining(entry -> {
            Map<String, Object> actual = map(endpoints.get(entry.getKey()));
            assertEquals(entry.getValue().path("method").asText(), actual.get("method"), entry.getKey());
            assertEquals(entry.getValue().path("path").asText(), actual.get("path"), entry.getKey());
        });

        JsonNode expectedSchemas = required.path("schemas");
        Map<String, Object> schemas = map(contract.get("schemas"));
        expectedSchemas.fields().forEachRemaining(entry -> {
            Map<String, Object> actual = map(schemas.get(entry.getKey()));
            List<String> expectedRequired = strings(entry.getValue().path("required"));
            if (!expectedRequired.isEmpty()) {
                assertContainsAll(list(actual.get("required")), expectedRequired);
            }
            Map<String, Object> properties = map(actual.get("properties"));
            assertContainsAll(new ArrayList<>(properties.keySet()), strings(entry.getValue().path("properties")));
        });

        Map<String, Object> streaming = map(contract.get("streaming"));
        assertEquals("[DONE]", streaming.get("done_sentinel"));
        assertContainsAll(list(streaming.get("chat_completions_events")), List.of("chat.completion.chunk", "[DONE]"));
        assertContainsAll(list(streaming.get("responses_events")), List.of("response.output_text.delta", "response.completed"));
    }

    @Test
    void capabilitiesAlignWithContractAndFixture() {
        Response response = new AgentCapabilitiesResource().capabilities();
        Map<String, Object> capabilities = map(response.getEntity());
        JsonNode required = FIXTURE.path("required");
        Map<String, Object> contractEndpoints = map(AgentContractMapper.contract().get("endpoints"));
        Map<String, Object> capabilityEndpoints = map(capabilities.get("endpoints"));

        assertEquals(200, response.getStatus());
        assertEquals("gollek.agent_capabilities", capabilities.get("object"));
        assertEquals(required.path("version").asText(), capabilities.get("version"));
        assertEquals(required.path("version").asText(), capabilities.get("contract_version"));
        assertEquals(List.of(required.path("version").asText()), capabilities.get("supported_contract_versions"));
        assertEquals(required.path("service_role").asText(), capabilities.get("service_role"));
        assertContainsAll(list(capabilities.get("compatibility")), strings(required.path("compatibility")));
        Map<String, Object> featureNegotiation = map(capabilities.get("feature_negotiation"));
        assertEquals("feature_flags", featureNegotiation.get("mode"));
        assertContainsAll(list(featureNegotiation.get("required_features")), strings(required.path("compatibility")));
        assertEquals(path(contractEndpoints, "chat_completions"), capabilityEndpoints.get("openai_chat_completions"));
        assertEquals(path(contractEndpoints, "responses"), capabilityEndpoints.get("openai_responses"));
        assertEquals(path(contractEndpoints, "embeddings"), capabilityEndpoints.get("openai_embeddings"));
        assertEquals(path(contractEndpoints, "model_capabilities"), capabilityEndpoints.get("model_capabilities"));
        assertEquals(path(contractEndpoints, "agent_contract"), capabilityEndpoints.get("agent_contract"));
        assertEquals(path(contractEndpoints, "agent_readiness_issues"),
                capabilityEndpoints.get("agent_readiness_issues"));
        assertEquals(path(contractEndpoints, "agent_preflight"), capabilityEndpoints.get("agent_preflight"));
        assertEquals("/v1/agent/validate", capabilityEndpoints.get("agent_validation"));
        assertEquals(path(contractEndpoints, "agent_tool_validation"), capabilityEndpoints.get("agent_tool_validation"));
        assertEquals("/v1/mcp/tools", capabilityEndpoints.get("mcp_tools"));

        Map<String, Object> boundary = map(capabilities.get("agent_boundary"));
        assertContainsAll(list(boundary.get("agent_orchestrator_owns")), List.of(
                "planning",
                "tool execution loops",
                "workflow state"));
    }

    @Test
    void readinessIssueCatalogResourcePublishesStableCodes() {
        Response response = new AgentReadinessIssueCatalogResource().catalog();
        Map<String, Object> catalog = map(response.getEntity());

        assertEquals(200, response.getStatus());
        assertEquals("gollek.agent_readiness_issue_catalog", catalog.get("object"));
        assertEquals("v1", catalog.get("version"));
        assertEquals("inference_serving_engine", catalog.get("service_role"));
        assertEquals(false, map(catalog.get("boundary")).get("model_invoked"));
        assertTrue((Integer) catalog.get("count") > 0);

        Map<String, Object> byCode = map(catalog.get("by_code"));
        Map<String, Object> toolInvalid = map(byCode.get("TOOL_DEFINITIONS_INVALID"));
        assertEquals("tool_validation", toolInvalid.get("area"));
        assertEquals("blocking", toolInvalid.get("default_severity"));
        assertTrue(Objects.toString(toolInvalid.get("remediation")).contains("tool schema"));

        Map<String, Object> byArea = map(catalog.get("by_area"));
        assertTrue(list(byArea.get("request_validation")).stream()
                .map(AgentServingContractFixtureTest::map)
                .anyMatch(entry -> "AGENT_REQUEST_INVALID".equals(entry.get("code"))));
    }

    @Test
    void validationDryRunsAgentSurfacesFromFixture() {
        JsonNode requests = FIXTURE.path("requests");

        Map<String, Object> chat = AgentValidationMapper.validate(
                "chat",
                HEADERS,
                requests.path("chat"),
                AgentTraceContext.fromPayload(requests.path("chat")));
        assertEquals("chat", chat.get("surface"));
        assertValidationOnly(chat);
        Map<String, Object> chatNormalized = map(chat.get("normalized"));
        assertEquals("demo-chat", chatNormalized.get("model"));
        assertEquals(true, chatNormalized.get("streaming"));
        assertEquals(1, chatNormalized.get("tool_count"));
        Map<String, Object> firstTool = map(list(chatNormalized.get("tools")).get(0));
        assertEquals("knowledge", map(firstTool.get("metadata")).get("mcp_server"));
        assertEquals(true, map(chatNormalized.get("rag")).get("injected"));
        assertEquals(1, map(chatNormalized.get("rag")).get("items"));
        assertEquals(true, map(chatNormalized.get("stream_options")).get("include_usage"));

        Map<String, Object> responses = AgentValidationMapper.validate(
                "responses",
                HEADERS,
                requests.path("responses"),
                AgentTraceContext.fromPayload(requests.path("responses")));
        assertEquals("responses", responses.get("surface"));
        assertValidationOnly(responses);
        Map<String, Object> responsesNormalized = map(responses.get("normalized"));
        assertEquals("demo-response", responsesNormalized.get("model"));
        assertEquals(true, map(responsesNormalized.get("rag")).get("injected"));
        assertEquals("context_documents", map(responsesNormalized.get("rag")).get("alias"));
        assertEquals("session-response-fixture", responsesNormalized.get("session_id"));

        Map<String, Object> embeddings = AgentValidationMapper.validate(
                "embeddings",
                HEADERS,
                requests.path("embeddings"),
                AgentTraceContext.fromPayload(requests.path("embeddings")));
        assertEquals("embeddings", embeddings.get("surface"));
        assertValidationOnly(embeddings);
        Map<String, Object> embeddingNormalized = map(embeddings.get("normalized"));
        assertEquals("demo-embed", embeddingNormalized.get("model"));
        assertEquals(2, embeddingNormalized.get("input_count"));
        assertEquals(384, embeddingNormalized.get("requested_dimensions"));
        assertEquals("float", embeddingNormalized.get("encoding_format"));
        assertContainsAll(list(embeddingNormalized.get("parameter_keys")),
                List.of("dimensions", "encoding_format", "metadata"));
        Map<String, Object> embeddingRag = map(embeddingNormalized.get("rag"));
        assertEquals(true, embeddingRag.get("embedding_generation"));
        assertEquals(false, embeddingRag.get("retrieval_execution"));
        assertEquals("agent_orchestrator", embeddingRag.get("retrieval_policy_owned_by"));
        assertEquals("agent_orchestrator", embeddingRag.get("vector_store_owned_by"));
    }

    @Test
    void toolContractFixtureIsValidatedWithoutExecution() {
        JsonNode payload = FIXTURE.path("requests").path("tools");
        AgentTraceContext trace = AgentTraceContext.fromPayload(payload);

        Map<String, Object> validation = AgentToolContractMapper.validatePayload(payload, trace);

        assertEquals("gollek.tool_contract_validation", validation.get("object"));
        assertEquals(true, validation.get("valid"));
        assertEquals(false, validation.get("model_invoked"));
        assertEquals(2, validation.get("tool_count"));
        assertFalse(list(validation.get("warnings")).isEmpty());
        assertEquals(false, map(validation.get("boundary")).get("tool_execution"));
        assertEquals("lookup_context", map(list(validation.get("normalized")).get(0)).get("name"));
        assertEquals("mcp_knowledge_search", map(list(validation.get("normalized")).get(1)).get("name"));
    }

    @Test
    void preflightFixturePublishesStableReadinessContract() {
        JsonNode request = FIXTURE.path("requests").path("chat");
        ObjectNode payload = MAPPER.createObjectNode();
        payload.put("model", request.path("model").asText());
        payload.put("surface", "chat");
        payload.put("feature_profile", AgentServingFeatureProfile.CHAT_AGENT);
        payload.put("mcp_discovery_required", true);
        payload.set("request", request.deepCopy());

        Map<String, Object> report = AgentPreflightMapper.preflight(
                HEADERS,
                payload,
                AgentTraceContext.fromPayload(request),
                AgentFixtureSdk.serving());

        assertEquals("gollek.agent_preflight", report.get("object"));
        assertEquals("ready", report.get("status"));
        assertEquals(true, report.get("ready"));
        assertEquals("chat", report.get("surface"));
        assertEquals("demo-chat", report.get("model"));
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, report.get("feature_profile"));
        assertEquals(0, report.get("blocking_issue_count"));
        Map<String, Object> boundary = map(report.get("boundary"));
        assertEquals(true, boundary.get("validation_only"));
        assertEquals(false, boundary.get("model_invoked"));
        assertEquals(false, boundary.get("tool_execution"));
        assertEquals(false, boundary.get("retrieval_execution"));

        Map<String, Object> checkResults = map(report.get("check_results"));
        assertContainsAll(new ArrayList<>(checkResults.keySet()), List.of(
                "capabilities",
                "contract",
                "feature_negotiation",
                "model_route",
                "mcp_discovery",
                "tool_validation",
                "request_validation"));
        Map<String, Object> featureNegotiation = map(checkResults.get("feature_negotiation"));
        assertEquals("ready", featureNegotiation.get("status"));
        Map<String, Object> featureDetails = map(featureNegotiation.get("details"));
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, featureDetails.get("feature_profile"));
        assertEquals(true, featureDetails.get("feature_profile_supported"));

        Map<String, Object> mcpDiscovery = map(report.get("mcp_discovery"));
        assertEquals(true, mcpDiscovery.get("available"));
        assertEquals("openai", mcpDiscovery.get("compat"));
        assertEquals(false, map(mcpDiscovery.get("boundary")).get("tool_execution"));

        Map<String, Object> toolValidation = map(report.get("tool_validation"));
        assertEquals(true, toolValidation.get("valid"));
        assertEquals(false, toolValidation.get("model_invoked"));
        assertEquals(false, map(toolValidation.get("boundary")).get("tool_execution"));

        Map<String, Object> requestValidation = map(report.get("request_validation"));
        assertValidationOnly(requestValidation);
        Map<String, Object> requestCheck = map(checkResults.get("request_validation"));
        Map<String, Object> requestDetails = map(map(requestCheck.get("details")).get("request"));
        assertEquals(true, requestDetails.get("streaming"));
        assertEquals(3, requestDetails.get("message_count"));
        assertEquals(1, requestDetails.get("tool_count"));
        assertEquals(true, map(requestDetails.get("rag")).get("injected"));

        Map<String, Object> readiness = map(report.get("readiness_report"));
        assertEquals("gollek.agent_readiness_report", readiness.get("object"));
        assertEquals(report.get("status"), readiness.get("status"));
        assertEquals(report.get("feature_profile"), readiness.get("feature_profile"));
        assertEquals(checkResults, readiness.get("checks"));
        assertEquals(report.get("checked_areas"), readiness.get("checked_areas"));
    }

    private static void assertValidationOnly(Map<String, Object> payload) {
        assertEquals(true, payload.get("valid"));
        assertEquals(false, payload.get("model_invoked"));
        Map<String, Object> boundary = map(payload.get("boundary"));
        assertEquals(true, boundary.get("validation_only"));
        assertEquals(false, boundary.get("tool_execution"));
        assertEquals(false, boundary.get("retrieval_execution"));
    }

    private static String path(Map<String, Object> endpoints, String name) {
        return Objects.toString(map(endpoints.get(name)).get("path"));
    }

    private static void assertContainsAll(List<?> actual, List<String> expected) {
        for (String value : expected) {
            assertTrue(actual.contains(value), "missing " + value + " from " + actual);
        }
    }

    private static List<String> strings(JsonNode node) {
        List<String> out = new ArrayList<>();
        if (node != null && node.isArray()) {
            node.forEach(value -> out.add(value.asText()));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        assertTrue(value instanceof Map<?, ?>, "expected map but got " + value);
        return (Map<String, Object>) value;
    }

    private static List<?> list(Object value) {
        assertTrue(value instanceof List<?>, "expected list but got " + value);
        return (List<?>) value;
    }

    private static JsonNode loadFixture() {
        try (InputStream stream = AgentServingContractFixtureTest.class.getResourceAsStream(
                "/agent-serving-contract/required-contract.json")) {
            return MAPPER.readTree(Objects.requireNonNull(stream, "missing agent serving contract fixture"));
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private record FixtureHeaders(Map<String, String> values) implements HttpHeaders {
        @Override
        public List<String> getRequestHeader(String name) {
            String value = getHeaderString(name);
            return value == null ? List.of() : List.of(value);
        }

        @Override
        public String getHeaderString(String name) {
            if (name == null) {
                return null;
            }
            for (Map.Entry<String, String> entry : values.entrySet()) {
                if (entry.getKey().equalsIgnoreCase(name)) {
                    return entry.getValue();
                }
            }
            return null;
        }

        @Override
        public MultivaluedMap<String, String> getRequestHeaders() {
            MultivaluedMap<String, String> headers = new MultivaluedHashMap<>();
            values.forEach(headers::add);
            return headers;
        }

        @Override
        public List<MediaType> getAcceptableMediaTypes() {
            return List.of();
        }

        @Override
        public List<Locale> getAcceptableLanguages() {
            return List.of();
        }

        @Override
        public MediaType getMediaType() {
            return null;
        }

        @Override
        public Locale getLanguage() {
            return null;
        }

        @Override
        public Map<String, Cookie> getCookies() {
            return Map.of();
        }

        @Override
        public Date getDate() {
            return null;
        }

        @Override
        public int getLength() {
            return -1;
        }
    }
}
