package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSession;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.Authenticator;
import java.net.CookieHandler;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentIntegrationClientTest {

    @Test
    void readsAgentCapabilitiesWithAuthHeaders() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"service_role":"inference_serving_engine","endpoints":{"agent_contract":"/v1/agent/contract"}}
                """);
        AgentIntegrationClient client = client(httpClient);

        Map<String, Object> response = client.capabilities();

        assertEquals("inference_serving_engine", response.get("service_role"));
        assertEquals("GET", httpClient.lastRequest().method());
        assertEquals(URI.create("http://localhost:8080/v1/agent/capabilities"), httpClient.lastRequest().uri());
        assertEquals("test-key", httpClient.lastRequest().headers()
                .firstValue("X-API-Key").orElseThrow());
        assertEquals("Bearer test-key", httpClient.lastRequest().headers()
                .firstValue("Authorization").orElseThrow());
    }

    @Test
    void readsAgentCapabilitiesView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_capabilities",
                  "version":"v1",
                  "contract_version":"v1",
                  "supported_contract_versions":["v1"],
                  "service_role":"inference_serving_engine",
                  "feature_negotiation":{
                    "mode":"feature_flags",
                    "feature_namespace":"gollek.agent.compatibility",
                    "contract_version":"v1",
                    "supported_contract_versions":["v1"],
                    "required_features":[
                      "openai_chat_completions",
                      "openai_chat_streaming",
                      "openai_responses",
                      "openai_responses_streaming",
                      "openai_embeddings",
                      "model_capability_matrix",
                      "agent_capabilities",
                      "agent_contract",
                      "agent_feature_negotiation",
                      "agent_readiness_issue_catalog",
                      "agent_preflight",
                      "agent_request_validation",
                      "agent_tool_contract_validation",
                      "request_trace_context",
                      "mcp_tool_discovery",
                      "rag_context"
                    ],
                    "optional_features":["stream_usage_reporting"]
                  },
                  "agent_boundary":{
                    "gollek_owns":[
                      "model serving",
                      "provider routing",
                      "system prompts",
                      "tool-call request and response schema",
                      "embedding generation",
                      "RAG context injection",
                      "MCP server registry and tool definitions"
                    ],
                    "agent_orchestrator_owns":[
                      "planning",
                      "memory policy",
                      "tool authorization",
                      "tool execution loops",
                      "workflow state"
                    ]
                  },
                  "compatibility":[
                    "openai_chat_completions",
                    "openai_chat_streaming",
                    "openai_responses",
                    "openai_responses_streaming",
                    "openai_embeddings",
                    "model_capability_matrix",
                    "agent_capabilities",
                    "agent_contract",
                    "agent_feature_negotiation",
                    "agent_readiness_issue_catalog",
                    "agent_preflight",
                    "agent_request_validation",
                    "agent_tool_contract_validation",
                    "request_trace_context",
                    "mcp_tool_discovery",
                    "rag_context"
                  ],
                  "endpoints":{
                    "openai_chat_completions":"/v1/chat/completions",
                    "openai_responses":"/v1/responses",
                    "openai_embeddings":"/v1/embeddings",
                    "model_capabilities":"/v1/models/{id}/capabilities",
                    "agent_contract":"/v1/agent/contract",
                    "agent_readiness_issues":"/v1/agent/readiness/issues",
                    "agent_preflight":"/v1/agent/preflight",
                    "agent_validation":"/v1/agent/validate",
                    "agent_tool_validation":"/v1/agent/tools/validate",
                    "mcp_tools":"/v1/mcp/tools"
                  },
                  "auth":{
                    "x_api_key_header":"X-API-Key",
                    "authorization_header":"Bearer token"
                  },
                  "traceability":{
                    "request_id_header":"X-Gollek-Request-Id",
                    "trace_id_header":"X-Gollek-Trace-Id",
                    "session_id_header":"X-Gollek-Session-Id",
                    "user_id_header":"X-Gollek-User-Id",
                    "metadata_key":"gollek_trace"
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentCapabilitiesView capabilities = client.capabilitiesView();

        assertEquals("gollek.agent_capabilities", capabilities.object());
        assertEquals("v1", capabilities.version());
        assertEquals("v1", capabilities.contractVersion());
        assertEquals(List.of("v1"), capabilities.supportedContractVersions());
        assertTrue(capabilities.supportsContractVersion("v1"));
        assertEquals("feature_flags", capabilities.featureNegotiation().mode());
        assertEquals("gollek.agent.compatibility", capabilities.featureNegotiation().featureNamespace());
        assertTrue(capabilities.featureNegotiation().supportsFeature("agent_feature_negotiation"));
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE,
                capabilities.featureNegotiation().defaultFeatureProfile());
        assertTrue(capabilities.featureNegotiation()
                .supportedFeatureProfiles()
                .contains(AgentServingFeatureProfile.EMBEDDING_RAG));
        assertTrue(capabilities.featureNegotiation()
                .supportsFeatureProfile(AgentServingFeatureProfile.CHAT_AGENT));
        assertEquals(List.of("wayang_planner"),
                capabilities.featureNegotiation().unsupportedFeatures(List.of("rag_context", "wayang_planner")));
        assertEquals("inference_serving_engine", capabilities.serviceRole());
        assertTrue(capabilities.hasRequiredAgentServingCapabilities(), capabilities.agentServingCapabilityIssues().toString());
        assertTrue(capabilities.supportsOpenAiChat());
        assertTrue(capabilities.supportsOpenAiResponses());
        assertTrue(capabilities.supportsEmbeddings());
        assertTrue(capabilities.supportsModelCapabilities());
        assertTrue(capabilities.supportsReadinessIssueCatalog());
        assertTrue(capabilities.supportsMcpToolDiscovery());
        assertTrue(capabilities.supportsRagContext());
        assertEquals("/v1/responses", capabilities.endpoint("openai_responses"));
        assertEquals("X-API-Key", capabilities.apiKeyHeader());
        assertEquals("Bearer token", capabilities.authorizationHeaderScheme());
        assertEquals("X-Gollek-Trace-Id", capabilities.traceIdHeader());
        assertEquals("gollek_trace", capabilities.traceMetadataKey());
        assertTrue(capabilities.gollekOwns("rag context injection"));
        assertTrue(capabilities.orchestratorOwns("tool_execution_loops"));
        assertEquals(URI.create("http://localhost:8080/v1/agent/capabilities"), httpClient.lastRequest().uri());
    }

    @Test
    void capabilitiesViewReportsMissingServingFeatures() {
        AgentCapabilitiesView capabilities = AgentCapabilitiesView.from(Map.of(
                "service_role", "agent_runtime",
                "compatibility", List.of("openai_chat_completions"),
                "endpoints", Map.of("openai_chat_completions", "/v1/chat/completions"),
                "agent_boundary", Map.of(
                        "gollek_owns", List.of("provider routing"),
                        "agent_orchestrator_owns", List.of("planning")),
                "auth", Map.of(
                        "x_api_key_header", "X-API-Key",
                        "authorization_header", "ApiKey token"),
                "traceability", Map.of(
                        "request_id_header", "X-Request-Id")));

        List<String> issues = capabilities.agentServingCapabilityIssues();

        assertFalse(capabilities.hasRequiredAgentServingCapabilities());
        assertTrue(issues.contains("service_role must be inference_serving_engine"));
        assertTrue(issues.contains("missing supported contract version: v1"));
        assertTrue(issues.contains("missing compatibility: openai_responses"));
        assertTrue(issues.contains("missing endpoint: agent_contract"));
        assertTrue(issues.contains("missing endpoint: model_capabilities"));
        assertTrue(issues.contains("agent boundary must keep model serving in Gollek"));
        assertTrue(issues.contains("agent boundary must keep tool execution loops in the orchestrator"));
        assertTrue(issues.contains("auth.authorization_header must be Bearer token"));
        assertTrue(issues.contains("traceability.request_id_header must be X-Gollek-Request-Id"));
    }

    @Test
    void readsAgentServingContractView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, servingContractJson());
        AgentIntegrationClient client = client(httpClient);

        AgentServingContract contract = client.contractView();

        assertEquals("v1", contract.version());
        assertEquals("v1", contract.contractVersion());
        assertEquals(List.of("v1"), contract.supportedContractVersions());
        assertTrue(contract.supportsContractVersion("v1"));
        assertTrue(contract.supportsFeature("agent_feature_negotiation"));
        assertEquals(List.of("wayang_planner"),
                contract.featureNegotiation().unsupportedFeatures(List.of("rag_context", "wayang_planner")));
        assertEquals("inference_serving_engine", contract.serviceRole());
        assertTrue(contract.hasRequiredServingBoundary(), contract.servingBoundaryIssues().toString());
        assertEquals("/v1/responses", contract.endpointPath("responses"));
        assertEquals("/v1/models/{id}/capabilities", contract.endpointPath("model_capabilities"));
        assertEquals("POST", contract.endpoint("agent_validation").method());
        assertTrue(contract.schemaRequires("chat_completions_request", "messages"));
        assertTrue(contract.schemaHasProperty("responses_request", "rag_context"));
        assertTrue(contract.gollekOwns("mcp_registry_discovery"));
        assertTrue(contract.orchestratorOwns("tool_execution"));
        assertFalse(contract.toolExecutionEnabled());
        assertFalse(contract.retrievalExecutionEnabled());
        assertEquals(URI.create("http://localhost:8080/v1/agent/contract"), httpClient.lastRequest().uri());
    }

    @Test
    void servingContractReportsBoundaryIssues() {
        AgentServingContract contract = AgentServingContract.from(Map.of(
                "service_role", "agent_runtime",
                "boundary", Map.of(
                        "gollek_owns", List.of("model_serving"),
                        "agent_orchestrator_owns", List.of("planning"),
                        "tool_execution", true,
                        "retrieval_execution", true)));

        List<String> issues = contract.servingBoundaryIssues();

        assertFalse(contract.hasRequiredServingBoundary());
        assertTrue(issues.contains("service_role must be inference_serving_engine"));
        assertTrue(issues.contains("missing supported contract version: v1"));
        assertTrue(issues.contains("Gollek contract must not enable tool execution"));
        assertTrue(issues.contains("Gollek contract must not enable retrieval execution"));
        assertTrue(issues.contains("missing endpoint: chat_completions"));
        assertTrue(issues.contains("missing Gollek-owned responsibility: mcp_registry_discovery"));
        assertTrue(issues.contains("missing orchestrator-owned responsibility: tool_execution"));
    }

    @Test
    void readsReadinessIssueCatalogView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, readinessIssueCatalogJson());
        AgentIntegrationClient client = client(httpClient);

        AgentReadinessIssueCatalogView catalog = client.readinessIssueCatalogView();

        assertEquals("gollek.agent_readiness_issue_catalog", catalog.object());
        assertEquals("v1", catalog.version());
        assertEquals("inference_serving_engine", catalog.serviceRole());
        assertTrue(catalog.servingBoundaryValid(), catalog.servingBoundaryIssues().toString());
        assertEquals(2, catalog.count());
        assertEquals(2, catalog.entries().size());
        assertTrue(catalog.hasCode("tool-definitions-invalid"));
        assertEquals(
                "Fix the OpenAI-compatible tool schema before serving the route.",
                catalog.find("TOOL_DEFINITIONS_INVALID").orElseThrow().remediation());
        assertEquals(1, catalog.entriesForArea("request-validation").size());
        assertEquals("AGENT_REQUEST_INVALID", catalog.byArea().get("request_validation").get(0).code());
        assertEquals(URI.create("http://localhost:8080/v1/agent/readiness/issues"),
                httpClient.lastRequest().uri());
    }

    @Test
    void readsModelCapabilitiesViewForAgentRoute() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "model_id":"org/demo-model",
                  "known":true,
                  "available":true,
                  "preferred_provider":"local",
                  "format":"GGUF",
                  "architecture":"llama",
                  "quantization":"q4_k_m",
                  "limits":{
                    "context_tokens":4096,
                    "input_tokens":2048,
                    "output_tokens":512,
                    "embedding_dimensions":768
                  },
                  "api_contract":{
                    "chat_completions":true,
                    "chat_streaming":true,
                    "responses":true,
                    "responses_streaming":true,
                    "embeddings_endpoint":true,
                    "system_prompt":true,
                    "tools_request_schema":true,
                    "mcp_tool_definitions":true,
                    "rag_context_injection":true
                  },
                  "openai_compatibility":{
                    "chat_completions":true,
                    "chat_streaming":true,
                    "responses":true,
                    "responses_streaming":true,
                    "embeddings":true
                  },
                  "inference":{
                    "completion":true,
                    "streaming":true,
                    "system_prompt":true,
                    "json_mode":true,
                    "structured_outputs":true
                  },
                  "tooling":{
                    "tool_definitions":true,
                    "tool_choice":true,
                    "model_tool_calling":true,
                    "mcp_tool_definitions":true,
                    "tool_execution":false
                  },
                  "rag":{
                    "context_injection":true,
                    "sources_metadata":true,
                    "embedding_model_hint":true,
                    "retrieval_policy":false,
                    "vector_store_ownership":false
                  },
                  "embeddings":{
                    "generation":true,
                    "endpoint":"/v1/embeddings",
                    "openai_compatible":true,
                    "dimensions":768,
                    "encoding_formats":["float","base64"],
                    "input_aliases":["input","inputs"],
                    "batch_inputs":true,
                    "metadata_passthrough":true,
                    "retrieval_policy":false,
                    "vector_store_ownership":false
                  },
                  "modalities":{
                    "text":true,
                    "embeddings":true,
                    "multimodal":false
                  },
                  "provider_candidates":[{
                    "id":"local",
                    "name":"Local",
                    "health":"healthy",
                    "supports_model":true,
                    "capabilities":{
                      "streaming":true,
                      "function_calling":true,
                      "tool_calling":true,
                      "structured_outputs":true,
                      "embeddings":true,
                      "max_context_tokens":8192,
                      "max_output_tokens":1024
                    }
                  }],
                  "metadata":{"family":"demo"}
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentModelCapabilitiesView view = client.modelCapabilitiesView("org/demo-model");

        assertEquals("org/demo-model", view.modelId());
        assertTrue(view.known());
        assertTrue(view.available());
        assertEquals("local", view.preferredProvider());
        assertEquals("GGUF", view.format());
        assertEquals("llama", view.architecture());
        assertEquals("q4_k_m", view.quantization());
        assertEquals(4096L, view.limits().contextTokens());
        assertEquals(512, view.limits().outputTokens());
        assertEquals(768, view.limits().embeddingDimensions());
        assertTrue(view.supportsChatCompletions());
        assertTrue(view.supportsChatStreaming());
        assertTrue(view.supportsResponses());
        assertTrue(view.supportsResponsesStreaming());
        assertTrue(view.supportsEmbeddings());
        assertTrue(view.supportsEmbeddingEndpoint());
        assertTrue(view.supportsEmbeddingGeneration());
        assertTrue(view.hasRequiredEmbeddingRoute(), view.embeddingRouteIssues().toString());
        AgentModelCapabilitiesView.EmbeddingCapabilities embeddings = view.embeddings();
        assertEquals("/v1/embeddings", embeddings.endpoint());
        assertEquals(768, embeddings.dimensions());
        assertEquals(List.of("float", "base64"), embeddings.encodingFormats());
        assertEquals(List.of("input", "inputs"), embeddings.inputAliases());
        assertTrue(embeddings.batchInputs());
        assertTrue(embeddings.metadataPassthrough());
        assertTrue(embeddings.storageOwnedByOrchestrator());
        assertTrue(view.supportsStreaming());
        assertTrue(view.supportsJsonMode());
        assertTrue(view.supportsStructuredOutputs());
        assertTrue(view.supportsToolDefinitions());
        assertTrue(view.supportsToolChoice());
        assertTrue(view.supportsModelToolCalling());
        assertTrue(view.supportsMcpToolDefinitions());
        assertFalse(view.toolExecutionEnabled());
        assertTrue(view.supportsRagContextInjection());
        assertFalse(view.retrievalPolicyOwnedByGollek());
        assertFalse(view.vectorStoreOwnedByGollek());
        assertTrue(view.hasRequiredAgentServingRoute(), view.agentServingRouteIssues().toString());
        assertEquals(1, view.providerCandidates().size());
        assertTrue(view.providerCandidates().get(0).supportsToolCalling());
        assertEquals(8192, view.providerCandidates().get(0).maxContextTokens());
        assertEquals("demo", view.metadata().get("family"));
        assertEquals(URI.create("http://localhost:8080/v1/models/org%2Fdemo-model/capabilities"),
                httpClient.lastRequest().uri());
    }

    @Test
    void modelCapabilitiesViewReportsServingRouteIssues() {
        AgentModelCapabilitiesView view = AgentModelCapabilitiesView.from(Map.of(
                "model_id", "bad-model",
                "available", false,
                "api_contract", Map.of(
                        "chat_completions", false,
                        "responses", false,
                        "system_prompt", false,
                        "tools_request_schema", false,
                        "rag_context_injection", false),
                "openai_compatibility", Map.of(
                        "chat_completions", false,
                        "responses", false,
                        "embeddings", false),
                "inference", Map.of("completion", false),
                "tooling", Map.of("tool_execution", true),
                "rag", Map.of(
                        "context_injection", false,
                        "retrieval_policy", true,
                        "vector_store_ownership", true),
                "embeddings", Map.of(
                        "generation", false,
                        "retrieval_policy", true,
                        "vector_store_ownership", true)));

        List<String> issues = view.agentServingRouteIssues();
        List<String> embeddingIssues = view.embeddingRouteIssues();

        assertFalse(view.hasRequiredAgentServingRoute());
        assertFalse(view.hasRequiredEmbeddingRoute());
        assertTrue(issues.contains("model is not available through Gollek serving"));
        assertTrue(issues.contains("model does not advertise completion serving"));
        assertTrue(issues.contains("model does not advertise Chat Completions compatibility"));
        assertTrue(issues.contains("model does not advertise Responses compatibility"));
        assertTrue(issues.contains("model does not advertise system prompt mapping"));
        assertTrue(issues.contains("model does not advertise tool definition ingestion"));
        assertTrue(issues.contains("Gollek model capability must not enable tool execution"));
        assertTrue(issues.contains("model does not advertise RAG context injection"));
        assertTrue(issues.contains("retrieval policy must stay with the agent orchestrator"));
        assertTrue(issues.contains("vector store ownership must stay with the agent orchestrator"));
        assertTrue(embeddingIssues.contains("model route does not advertise the embeddings endpoint"));
        assertTrue(embeddingIssues.contains("model route does not support embedding generation for /v1/embeddings"));
        assertTrue(embeddingIssues.contains("embedding route must not own retrieval policy"));
        assertTrue(embeddingIssues.contains("embedding route must not own vector store state"));
    }

    @Test
    void servingReadinessRunsDiscoveryAndValidationOnly() throws Exception {
        FakeHttpClient httpClient = FakeHttpClient.sequence(
                FakeHttpClient.json(200, capabilitiesJson()),
                FakeHttpClient.json(200, servingContractJson()),
                FakeHttpClient.json(200, modelCapabilitiesJson()),
                FakeHttpClient.json(200, mcpToolsJson()),
                FakeHttpClient.json(200, toolValidationJson()),
                FakeHttpClient.json(200, requestValidationJson()));
        AgentIntegrationClient client = client(httpClient);
        AgentRequestOptions options = AgentRequestOptions.builder()
                .requestId("req-preflight")
                .traceId("trace-preflight")
                .build();
        Map<String, Object> callerTool = Map.of(
                "type", "function",
                "function", Map.of(
                        "name", "caller_tool",
                        "parameters", Map.of("type", "object")));
        Map<String, Object> request = Map.of(
                "model", "demo-model",
                "messages", List.of(Map.of("role", "user", "content", "Check readiness")),
                "tools", List.of(callerTool));

        AgentServingReadinessReport readiness = client.servingReadiness(
                "demo-model",
                "chat",
                request,
                options);

        assertTrue(readiness.ready(), readiness.issues().toString());
        assertTrue(readiness.featureNegotiationCheck().ready());
        assertEquals(AgentServingFeatureProfile.DEFAULT_PROFILE,
                readiness.featureNegotiationCheck().details().get("feature_profile"));
        assertEquals("v1", readiness.featureNegotiationCheck().details().get("required_contract_version"));
        assertEquals(6, httpClient.requests().size());
        assertEquals(URI.create("http://localhost:8080/v1/agent/capabilities"),
                httpClient.requests().get(0).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/contract"),
                httpClient.requests().get(1).uri());
        assertEquals(URI.create("http://localhost:8080/v1/models/demo-model/capabilities"),
                httpClient.requests().get(2).uri());
        assertEquals(URI.create("http://localhost:8080/v1/mcp/tools?compat=openai&enabledOnly=true"),
                httpClient.requests().get(3).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/tools/validate"),
                httpClient.requests().get(4).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=chat"),
                httpClient.requests().get(5).uri());
        assertTrue(httpClient.requests().stream()
                .noneMatch(requestItem -> requestItem.uri().getPath().contains("/v1/chat/completions")));
        for (HttpRequest sent : httpClient.requests()) {
            assertEquals("req-preflight", sent.headers()
                    .firstValue(AgentRequestOptions.REQUEST_ID_HEADER).orElseThrow());
        }

        Map<String, Object> toolValidationBody = new ObjectMapper().readValue(
                requestBody(httpClient.requests().get(4)),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
                });
        List<?> tools = (List<?>) toolValidationBody.get("tools");
        Map<?, ?> tool = (Map<?, ?>) tools.get(0);
        Map<?, ?> function = (Map<?, ?>) tool.get("function");
        assertEquals("caller_tool", function.get("name"));

        Map<String, Object> requestValidationBody = new ObjectMapper().readValue(
                requestBody(httpClient.requests().get(5)),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
        });
        assertEquals("demo-model", requestValidationBody.get("model"));
    }

    @Test
    void servingReadinessRequestCanSkipMcpAndToolValidation() throws Exception {
        FakeHttpClient httpClient = FakeHttpClient.sequence(
                FakeHttpClient.json(200, capabilitiesJson()),
                FakeHttpClient.json(200, servingContractJson()),
                FakeHttpClient.json(200, modelCapabilitiesJson()),
                FakeHttpClient.json(200, requestValidationJson()));
        AgentIntegrationClient client = client(httpClient);
        AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                .modelId("demo-model")
                .surface("chat")
                .request(Map.of(
                        "model", "demo-model",
                        "messages", List.of(Map.of("role", "user", "content", "Check readiness"))))
                .discoverMcpTools(false)
                .mcpDiscoveryRequired(false)
                .validateTools(false)
                .build();

        AgentServingReadinessReport readiness = client.servingReadiness(preflight);

        assertTrue(readiness.ready(), readiness.issues().toString());
        assertEquals(4, httpClient.requests().size());
        assertEquals(URI.create("http://localhost:8080/v1/agent/capabilities"),
                httpClient.requests().get(0).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/contract"),
                httpClient.requests().get(1).uri());
        assertEquals(URI.create("http://localhost:8080/v1/models/demo-model/capabilities"),
                httpClient.requests().get(2).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=chat"),
                httpClient.requests().get(3).uri());
        assertTrue(httpClient.requests().stream()
                .noneMatch(request -> request.uri().getPath().contains("/v1/mcp/tools")));
        assertTrue(httpClient.requests().stream()
                .noneMatch(request -> request.uri().getPath().contains("/v1/agent/tools/validate")));
    }

    @Test
    void servingReadinessUsesEmbeddingSurfaceModelRoute() throws Exception {
        FakeHttpClient httpClient = FakeHttpClient.sequence(
                FakeHttpClient.json(200, capabilitiesJson()),
                FakeHttpClient.json(200, servingContractJson()),
                FakeHttpClient.json(200, embeddingModelCapabilitiesJson()),
                FakeHttpClient.json(200, embeddingRequestValidationJson()));
        AgentIntegrationClient client = client(httpClient);
        AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                .modelId("demo-embed")
                .surface("embeddings")
                .featureProfile(AgentServingFeatureProfile.EMBEDDING_RAG)
                .request(Map.of(
                        "model", "demo-embed",
                        "input", List.of("Index this document"),
                        "dimensions", 768,
                        "encoding_format", "float"))
                .discoverMcpTools(false)
                .mcpDiscoveryRequired(false)
                .validateTools(false)
                .toolValidationRequired(false)
                .build();

        AgentServingReadinessReport readiness = client.servingReadiness(preflight);

        assertTrue(readiness.ready(), readiness.issues().toString());
        assertTrue(readiness.check(AgentServingReadinessReport.AREA_MODEL_ROUTE).ready());
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG,
                readiness.featureNegotiationCheck().details().get("feature_profile"));
        assertTrue(((List<?>) readiness.featureNegotiationCheck().details().get("required_features"))
                .contains("openai_embeddings"));
        assertEquals(4, httpClient.requests().size());
        assertEquals(URI.create("http://localhost:8080/v1/models/demo-embed/capabilities"),
                httpClient.requests().get(2).uri());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=embeddings"),
                httpClient.requests().get(3).uri());
    }

    @Test
    void servingPreflightCallsServerEndpointOnce() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_preflight",
                  "status":"ready",
                  "ready":true,
                  "surface":"chat",
                  "model":"demo-model",
                  "blocking_issue_count":0,
                  "warning_count":0,
                  "issues":[],
                  "issues_by_area":{},
                  "blocking_messages":[],
                  "warning_messages":[],
                  "boundary":{
                    "validation_only":true,
                    "model_invoked":false,
                    "tool_execution":false,
                    "retrieval_execution":false,
                    "tool_authorization":false
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);
        AgentRequestOptions options = AgentRequestOptions.builder()
                .requestId("req-server-preflight")
                .traceId("trace-server-preflight")
                .build();

        Map<String, Object> response = client.servingPreflight(
                AgentServingPreflightRequest.builder()
                        .modelId("demo-model")
                        .surface("chat")
                        .request(Map.of(
                                "model", "demo-model",
                                "messages", List.of(Map.of("role", "user", "content", "Check server preflight"))))
                        .requestOptions(options)
                        .featureProfile(AgentServingFeatureProfile.CHAT_AGENT)
                        .discoverMcpTools(false)
                        .mcpDiscoveryRequired(false)
                        .validateTools(false)
                        .requiredFeatures(List.of("rag_context", "mcp_tool_discovery"))
                        .optionalFeatures(List.of("stream_usage_reporting"))
                        .build());

        assertEquals("gollek.agent_preflight", response.get("object"));
        assertEquals("ready", response.get("status"));
        assertEquals(1, httpClient.requests().size());
        assertEquals(URI.create("http://localhost:8080/v1/agent/preflight"),
                httpClient.lastRequest().uri());
        assertEquals("req-server-preflight", httpClient.lastRequest().headers()
                .firstValue(AgentRequestOptions.REQUEST_ID_HEADER).orElseThrow());

        Map<String, Object> body = new ObjectMapper().readValue(
                requestBody(httpClient.lastRequest()),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
                });
        assertEquals("demo-model", body.get("model"));
        assertEquals("chat", body.get("surface"));
        assertEquals(false, body.get("discover_mcp_tools"));
        assertEquals(false, body.get("mcp_discovery_required"));
        assertEquals(false, body.get("validate_tools"));
        assertEquals(true, body.get("validate_request"));
        assertEquals(AgentServingFeatureProfile.CHAT_AGENT, body.get("feature_profile"));
        assertEquals("v1", body.get("required_contract_version"));
        assertEquals(List.of("rag_context", "mcp_tool_discovery"), body.get("required_features"));
        assertEquals(List.of("stream_usage_reporting"), body.get("optional_features"));
        assertTrue(body.get("request") instanceof Map<?, ?>);
    }

    @Test
    void servingPreflightProfileDerivesEmbeddingRagFeatureRequirements() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_preflight",
                  "status":"ready",
                  "ready":true,
                  "issues":[]
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        client.servingPreflight(AgentServingPreflightRequest.builder()
                .modelId("demo-embed")
                .surface("embeddings")
                .featureProfile(AgentServingFeatureProfile.EMBEDDING_RAG)
                .request(Map.of(
                        "model", "demo-embed",
                        "input", List.of("Index this document")))
                .discoverMcpTools(false)
                .mcpDiscoveryRequired(false)
                .validateTools(false)
                .toolValidationRequired(false)
                .build());

        Map<String, Object> body = new ObjectMapper().readValue(
                requestBody(httpClient.lastRequest()),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
                });
        List<?> required = (List<?>) body.get("required_features");
        assertEquals(AgentServingFeatureProfile.EMBEDDING_RAG, body.get("feature_profile"));
        assertTrue(required.contains("openai_embeddings"));
        assertTrue(required.contains("rag_context"));
        assertFalse(required.contains("mcp_tool_discovery"));
        assertEquals(List.of("embeddings", "models", "providers"), body.get("optional_features"));
    }

    @Test
    void servingPreflightReadinessMapsServerIssues() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_preflight",
                  "status":"blocked",
                  "ready":false,
                  "issues":[{
                    "area":"model_route",
                    "severity":"blocking",
                    "message":"model is not available through Gollek serving"
                  },{
                    "area":"tool_validation",
                    "severity":"warning",
                    "message":"tool schema warning (schema_feature_may_be_ignored)"
                  }]
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentServingReadinessReport readiness = client.servingPreflightReadiness(
                "demo-model",
                "chat",
                Map.of("model", "demo-model"));

        assertFalse(readiness.ready());
        assertTrue(readiness.hasWarnings());
        assertTrue(readiness.hasIssue(
                AgentServingReadinessReport.AREA_MODEL_ROUTE,
                "model is not available through Gollek serving"));
        assertEquals(List.of("tool schema warning (schema_feature_may_be_ignored)"),
                readiness.warningMessages());
        assertEquals(URI.create("http://localhost:8080/v1/agent/preflight"),
                httpClient.lastRequest().uri());
    }

    @Test
    void servingPreflightResultKeepsRawPayloadAndRouteComparison() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_preflight",
                  "status":"ready",
                  "ready":true,
                  "surface":"chat",
                  "model":"server-selected-model",
                  "feature_profile":"chat_agent",
                  "issues":[]
                }
                """);
        AgentIntegrationClient client = client(httpClient);
        AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                .modelId("demo-model")
                .surface("chat")
                .featureProfile(AgentServingFeatureProfile.CHAT_AGENT)
                .request(Map.of("model", "demo-model"))
                .build();

        AgentServingPreflightResult result = client.servingPreflightResult(preflight);

        assertTrue(result.ready());
        assertEquals(preflight, result.request());
        assertEquals("gollek.agent_preflight", result.response().get("object"));
        assertEquals("server-selected-model", result.readiness().model().orElseThrow());
        assertEquals("demo-model", result.requestedRoute().model());
        assertEquals("server-selected-model", result.selectedRoute().model());
        assertFalse(result.routeMatches());
        assertEquals(List.of("model"), result.routeMismatchFields());
        assertEquals(List.of("model"), result.routeComparison().mismatchFields());
        assertEquals(false, map(result.toMetadata().get("route_comparison")).get("matches"));
        assertEquals(1, httpClient.requests().size());
        assertEquals(URI.create("http://localhost:8080/v1/agent/preflight"),
                httpClient.lastRequest().uri());
    }

    @Test
    void servingPreflightGateEvaluatesCallerPolicy() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_preflight",
                  "status":"ready",
                  "ready":true,
                  "surface":"chat",
                  "model":"server-selected-model",
                  "feature_profile":"chat_agent",
                  "warning_messages":["mcp discovery unavailable"],
                  "warning_codes":["MCP_DISCOVERY_UNAVAILABLE"]
                }
                """);
        AgentIntegrationClient client = client(httpClient);
        AgentServingPreflightRequest preflight = AgentServingPreflightRequest.builder()
                .modelId("demo-model")
                .surface("chat")
                .featureProfile(AgentServingFeatureProfile.CHAT_AGENT)
                .request(Map.of("model", "demo-model"))
                .build();

        AgentServingPreflightGate gate = client.servingPreflightGate(
                preflight,
                AgentServingPreflightPolicy.clean());

        assertFalse(gate.ready());
        assertEquals(AgentServingPreflightPolicy.PROFILE_CLEAN, gate.policy().profile());
        assertTrue(gate.blocksOnRouteMismatch());
        assertTrue(gate.blocksOnWarnings());
        assertEquals(List.of(
                        AgentServingPreflightGate.REASON_ROUTE_MISMATCH,
                        AgentServingPreflightGate.REASON_WARNINGS_PRESENT),
                gate.blockingReasons());
        assertEquals(URI.create("http://localhost:8080/v1/agent/preflight"),
                httpClient.lastRequest().uri());
    }

    @Test
    void validatesRequestForSurface() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"valid":true,"surface":"responses","normalized":{"model":"demo"}}
                """);
        AgentIntegrationClient client = client(httpClient);

        Map<String, Object> response = client.validateRequest("responses", Map.of("model", "demo"));

        assertEquals(true, response.get("valid"));
        assertEquals("POST", httpClient.lastRequest().method());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=responses"),
                httpClient.lastRequest().uri());
        assertEquals("application/json", httpClient.lastRequest().headers()
                .firstValue("Content-Type").orElseThrow());
    }

    @Test
    void validatesRequestViewForServingDryRun() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.agent_validation",
                  "surface":"chat",
                  "valid":true,
                  "model_invoked":false,
                  "trace":{
                    "request_id":"req-chat",
                    "trace_id":"trace-chat",
                    "session_id":"session-chat",
                    "user_id":"user-chat"
                  },
                  "boundary":{
                    "validation_only":true,
                    "tool_execution":false,
                    "retrieval_execution":false
                  },
                  "normalized":{
                    "request_id":"req-chat",
                    "trace_id":"trace-chat",
                    "model":"demo-chat",
                    "streaming":true,
                    "message_count":2,
                    "tool_count":1,
                    "tools":[{
                      "name":"lookup_context",
                      "type":"function",
                      "strict":true,
                      "parameter_keys":["query"],
                      "metadata":{"mcp_server":"knowledge"}
                    }],
                    "parameter_keys":["temperature"],
                    "rag":{"injected":true,"items":2,"alias":"context_documents"},
                    "stream_options":{"include_usage":true},
                    "tool_contract":{"valid":true,"tool_count":1,"warning_count":0}
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentValidationView view = client.validateRequestView("chat", Map.of("model", "demo-chat"));

        assertEquals("gollek.agent_validation", view.object());
        assertEquals("chat", view.surface());
        assertTrue(view.valid());
        assertFalse(view.modelInvoked());
        assertTrue(view.validationOnly());
        assertFalse(view.toolExecutionEnabled());
        assertFalse(view.retrievalExecutionEnabled());
        assertEquals("req-chat", view.requestId());
        assertEquals("trace-chat", view.traceId());
        assertEquals("session-chat", view.sessionId());
        assertEquals("user-chat", view.userId());
        assertEquals("demo-chat", view.model());
        assertTrue(view.streaming());
        assertEquals(2, view.messageCount());
        assertEquals(1, view.toolCount());
        assertTrue(view.hasTools());
        assertEquals("lookup_context", view.tools().get(0).name());
        assertEquals("knowledge", view.tools().get(0).metadata().get("mcp_server"));
        assertEquals(List.of("temperature"), view.parameterKeys());
        assertTrue(view.ragContextInjected());
        assertEquals(2, view.ragContextItems());
        assertEquals("context_documents", view.ragContextAlias());
        assertTrue(view.includeUsageOnStream());
        assertTrue(view.toolContractValid());
        assertEquals(0, view.toolContractWarningCount());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=chat"),
                httpClient.lastRequest().uri());
    }

    @Test
    void validatesRequestViewForEmbeddingDryRun() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, embeddingRequestValidationJson());
        AgentIntegrationClient client = client(httpClient);

        AgentValidationView view = client.validateRequestView("embeddings", Map.of("model", "demo-embed"));

        assertEquals("embeddings", view.surface());
        assertTrue(view.embeddingSurface());
        assertEquals("demo-embed", view.model());
        assertEquals(1, view.inputCount());
        assertEquals(List.of(19), view.inputLengths());
        assertEquals(768, view.requestedDimensions());
        assertEquals("float", view.encodingFormat());
        assertEquals("agent-project", view.metadata().get("tenant"));
        AgentValidationView.EmbeddingValidation embedding = view.embeddingValidation();
        assertEquals("demo-embed", embedding.model());
        assertEquals(1, embedding.inputCount());
        assertEquals(List.of(19), embedding.inputLengths());
        assertTrue(embedding.hasRequestedDimensions());
        assertEquals(768, embedding.requestedDimensions());
        assertEquals("float", embedding.encodingFormat());
        assertEquals(List.of("dimensions", "encoding_format", "metadata"), embedding.parameterKeys());
        assertEquals("agent-project", embedding.metadata().get("tenant"));
        assertTrue(embedding.rag().embeddingGeneration());
        assertFalse(embedding.rag().retrievalExecution());
        assertEquals("agent_orchestrator", embedding.rag().retrievalPolicyOwnedBy());
        assertEquals("agent_orchestrator", embedding.rag().vectorStoreOwnedBy());
        assertTrue(embedding.rag().retrievalPolicyOwnedByOrchestrator());
        assertTrue(embedding.rag().vectorStoreOwnedByOrchestrator());
        assertTrue(embedding.storageOwnedByOrchestrator());
        assertEquals(URI.create("http://localhost:8080/v1/agent/validate?surface=embeddings"),
                httpClient.lastRequest().uri());
    }

    @Test
    void validatesToolContractViewWithoutExecution() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"gollek.tool_contract_validation",
                  "valid":true,
                  "model_invoked":false,
                  "trace":{"request_id":"req-tools","trace_id":"trace-tools"},
                  "tool_count":2,
                  "normalized":[{
                    "index":0,
                    "name":"lookup_context",
                    "type":"function",
                    "strict":true,
                    "description_present":true,
                    "parameter_keys":["properties","required","type"],
                    "parameter_schema":{
                      "type":"object",
                      "property_count":1,
                      "required":["query"],
                      "unsupported_keywords":["anyOf"]
                    },
                    "metadata":{"mcp_server":"knowledge"},
                    "warning_count":1
                  },{
                    "index":1,
                    "name":"mcp_knowledge_search",
                    "type":"mcp_tool",
                    "strict":false,
                    "description_present":false,
                    "parameter_keys":["type"],
                    "parameter_schema":{"type":"object"},
                    "metadata":{"mcp_server":"knowledge"},
                    "warning_count":0
                  }],
                  "warnings":[{
                    "index":0,
                    "path":"tools[0].function.parameters.properties.query.anyOf",
                    "code":"schema_feature_may_be_ignored",
                    "message":"JSON Schema keyword may not be supported by every agent client."
                  }],
                  "boundary":{
                    "validation_only":true,
                    "tool_execution":false,
                    "tool_authorization":false
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentToolValidationView view = client.validateToolsView(Map.of("tools", List.of()));

        assertEquals("gollek.tool_contract_validation", view.object());
        assertTrue(view.valid());
        assertFalse(view.modelInvoked());
        assertTrue(view.validationOnly());
        assertFalse(view.toolExecutionEnabled());
        assertFalse(view.toolAuthorizationEnabled());
        assertEquals(2, view.toolCount());
        assertEquals(List.of("lookup_context", "mcp_knowledge_search"), view.toolNames());
        assertTrue(view.hasTool("lookup_context"));
        assertTrue(view.hasMcpTools());
        assertTrue(view.hasWarnings());
        assertEquals("schema_feature_may_be_ignored", view.warnings().get(0).code());
        assertEquals("anyOf", ((List<?>) view.tools().get(0).parameterSchema()
                .get("unsupported_keywords")).get(0));
        assertEquals("knowledge", view.tools().get(1).metadata().get("mcp_server"));
        assertEquals(URI.create("http://localhost:8080/v1/agent/tools/validate"),
                httpClient.lastRequest().uri());
    }

    @Test
    void listsMcpToolsAsOpenAiDefinitions() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"available":true,"compat":"openai","enabled_only":false,"tools":[]}
                """);
        AgentIntegrationClient client = client(httpClient);

        Map<String, Object> response = client.mcpTools(true, false);

        assertEquals("openai", response.get("compat"));
        assertEquals(URI.create("http://localhost:8080/v1/mcp/tools?compat=openai&enabledOnly=false"),
                httpClient.lastRequest().uri());
    }

    @Test
    void readsMcpServerDiscoveryView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "available":true,
                  "registry_path":"/home/user/.gollek/mcp.json",
                  "servers":[
                    {"name":"knowledge","enabled":true},
                    {"name":"files","enabled":false}
                  ],
                  "boundary":{
                    "role":"discovery_only",
                    "gollek_exposes":["registered_servers","tool_schemas"],
                    "agent_orchestrator_owns":["tool_authorization","tool_execution","tool_result_loop"],
                    "tool_execution":false
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentMcpDiscoveryView view = client.mcpServersView();

        assertTrue(view.available());
        assertEquals("/home/user/.gollek/mcp.json", view.registryPath());
        assertTrue(view.discoveryOnly());
        assertFalse(view.toolExecutionEnabled());
        assertEquals(List.of("registered_servers", "tool_schemas"), view.gollekExposes());
        assertEquals(List.of("tool_authorization", "tool_execution", "tool_result_loop"), view.orchestratorOwns());
        assertEquals(List.of("knowledge", "files"), view.serverNames());
        assertTrue(view.hasServer("knowledge"));
        assertTrue(view.servers().get(0).enabled());
        assertFalse(view.servers().get(1).enabled());
        assertEquals(URI.create("http://localhost:8080/v1/mcp/servers"), httpClient.lastRequest().uri());
    }

    @Test
    void readsMcpToolsDiscoveryViewAsOpenAiDefinitions() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "available":true,
                  "registry_path":"/home/user/.gollek/mcp.json",
                  "compat":"openai",
                  "enabled_only":true,
                  "tools":[{
                    "type":"function",
                    "function":{
                      "name":"mcp_knowledge_search",
                      "description":"Search knowledge",
                      "parameters":{
                        "type":"object",
                        "properties":{"query":{"type":"string"}},
                        "required":["query"]
                      }
                    },
                    "x_gollek":{
                      "mcp_server":"knowledge",
                      "mcp_tool_name":"search",
                      "tool_execution":false
                    }
                  }],
                  "boundary":{
                    "role":"discovery_only",
                    "tool_execution":false
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentMcpDiscoveryView view = client.mcpToolsView(true, true);

        assertTrue(view.available());
        assertEquals("openai", view.compatibility());
        assertTrue(view.enabledOnly());
        assertTrue(view.discoveryOnly());
        assertEquals(List.of("mcp_knowledge_search"), view.toolNames());
        assertTrue(view.hasTool("mcp_knowledge_search"));
        assertEquals(1, view.openAiToolDefinitions().size());
        AgentMcpDiscoveryView.Tool tool = view.tools().get(0);
        assertTrue(tool.openAiCompatible());
        assertTrue(tool.mcpTool());
        assertFalse(tool.executionEnabled());
        assertEquals("knowledge", tool.mcpServer());
        assertEquals("search", tool.mcpToolName());
        assertEquals("object", tool.inputSchema().get("type"));
        assertEquals(URI.create("http://localhost:8080/v1/mcp/tools?compat=openai&enabledOnly=true"),
                httpClient.lastRequest().uri());
    }

    @Test
    void encodesMcpServerNameInPath() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"available":true,"server":"knowledge/base","compat":"openai","tools":[]}
                """);
        AgentIntegrationClient client = client(httpClient);

        client.mcpServerTools("knowledge/base", true);

        assertEquals(URI.create("http://localhost:8080/v1/mcp/servers/knowledge%2Fbase/tools?compat=openai"),
                httpClient.lastRequest().uri());
    }

    @Test
    void readsMcpServerToolsDiscoveryViewAsNativeTools() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "available":true,
                  "server":"knowledge/base",
                  "compat":"mcp",
                  "tools":[{
                    "type":"mcp_tool",
                    "server":"knowledge/base",
                    "name":"search",
                    "description":"Search knowledge",
                    "input_schema":{"type":"object"},
                    "execution":false
                  }],
                  "boundary":{
                    "role":"discovery_only",
                    "tool_execution":false
                  }
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentMcpDiscoveryView view = client.mcpServerToolsView("knowledge/base", false);

        assertEquals("knowledge/base", view.serverName());
        assertEquals("mcp", view.compatibility());
        assertTrue(view.discoveryOnly());
        assertEquals(List.of("search"), view.toolNames());
        AgentMcpDiscoveryView.Tool tool = view.tools().get(0);
        assertEquals("knowledge/base", tool.server());
        assertEquals("mcp_tool", tool.type());
        assertTrue(tool.mcpTool());
        assertFalse(tool.openAiCompatible());
        assertFalse(tool.executionEnabled());
        assertEquals(URI.create("http://localhost:8080/v1/mcp/servers/knowledge%2Fbase/tools?compat=mcp"),
                httpClient.lastRequest().uri());
    }

    @Test
    void createsNonStreamingChatCompletion() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hello"}}]}
                """);
        AgentIntegrationClient client = client(httpClient);

        Map<String, Object> response = client.createChatCompletion(Map.of(
                "model", "demo",
                "stream", true));

        assertEquals("chatcmpl-1", response.get("id"));
        assertEquals("POST", httpClient.lastRequest().method());
        assertEquals(URI.create("http://localhost:8080/v1/chat/completions"), httpClient.lastRequest().uri());

        Map<String, Object> requestBody = new ObjectMapper().readValue(
                requestBody(httpClient.lastRequest()),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
                });
        assertEquals(false, requestBody.get("stream"));
    }

    @Test
    void createsChatCompletionView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "id":"chatcmpl-1",
                  "object":"chat.completion",
                  "model":"demo",
                  "choices":[{
                    "message":{
                      "role":"assistant",
                      "content":"I will search.",
                      "tool_calls":[{
                        "id":"call_search",
                        "type":"function",
                        "function":{"name":"knowledge_search","arguments":"{\\"query\\":\\"gollek\\"}"}
                      }]
                    },
                    "finish_reason":"tool_calls"
                  }],
                  "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentResponseView view = client.createChatCompletionView(Map.of(
                "model", "demo",
                "messages", java.util.List.of(Map.of("role", "user", "content", "search"))));

        assertEquals(AgentStreamEvent.Surface.CHAT_COMPLETIONS, view.surface());
        assertEquals("I will search.", view.outputText());
        assertEquals("tool_calls", view.finishReason());
        assertEquals(3, view.usage().totalTokens());
        assertTrue(view.hasToolCalls());
        AgentStreamEvent.ToolCall call = view.toolCalls().get(0);
        assertEquals("call_search", call.id());
        assertEquals("knowledge_search", call.name());
        assertEquals("gollek", call.argumentsMap().get("query"));
    }

    @Test
    void createsResponseView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "id":"resp-1",
                  "object":"response",
                  "model":"demo",
                  "output_text":"I will search.",
                  "output":[{
                    "id":"fc_search",
                    "type":"function_call",
                    "name":"knowledge_search",
                    "arguments":"{\\"query\\":\\"gollek\\"}",
                    "call_id":"call_search",
                    "status":"completed"
                  }]
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentResponseView view = client.createResponseView(Map.of("model", "demo", "input", "search"));

        assertEquals(AgentStreamEvent.Surface.RESPONSES, view.surface());
        assertEquals("I will search.", view.outputText());
        assertTrue(view.hasToolCalls());
        AgentStreamEvent.ToolCall call = view.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals("call_search", call.callId());
        assertEquals("completed", call.status());
        assertEquals("knowledge_search", call.name());
        assertEquals("gollek", call.argumentsMap().get("query"));
    }

    @Test
    void createsEmbeddingView() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {
                  "object":"list",
                  "model":"demo-embed",
                  "data":[
                    {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}
                  ],
                  "usage":{"prompt_tokens":4,"total_tokens":4},
                  "metadata":{"gollek_trace":{"request_id":"req-embed","trace_id":"trace-embed"}}
                }
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentEmbeddingView view = client.createEmbeddingView(Map.of(
                "model", "demo-embed",
                "input", List.of("Gollek serves embeddings for RAG.")));

        assertEquals("demo-embed", view.model());
        assertEquals(1, view.count());
        assertEquals(3, view.dimensions());
        assertEquals(List.of(0.1, 0.2, 0.3), view.firstVector());
        assertEquals(4, view.usage().promptTokens());
        assertEquals("req-embed", view.trace().get("request_id"));
        assertEquals(URI.create("http://localhost:8080/v1/embeddings"), httpClient.lastRequest().uri());
    }

    @Test
    void streamsChatCompletionEventsAndReturnsSnapshot() throws Exception {
        FakeHttpClient httpClient = FakeHttpClient.stream(200, """
                data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}

                data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"lo"},"finish_reason":null}]}

                data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

                data: [DONE]
                """);
        AgentIntegrationClient client = client(httpClient);
        List<AgentStreamEvent> events = new ArrayList<>();

        AgentStreamAccumulator.Snapshot snapshot = client.streamChatCompletion(
                Map.of("model", "demo", "stream", false),
                events::add);

        assertEquals(4, events.size());
        assertEquals("Hello", snapshot.outputText());
        assertEquals("stop", snapshot.finishReason());
        assertEquals(3, snapshot.usage().totalTokens());
        assertTrue(snapshot.completed());
        assertEquals("POST", httpClient.lastRequest().method());
        assertEquals(URI.create("http://localhost:8080/v1/chat/completions"), httpClient.lastRequest().uri());

        Map<String, Object> requestBody = new ObjectMapper().readValue(
                requestBody(httpClient.lastRequest()),
                new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {
                });
        assertEquals(true, requestBody.get("stream"));
    }

    @Test
    void streamsResponseToolCallsAndReturnsSnapshot() throws Exception {
        FakeHttpClient httpClient = FakeHttpClient.stream(200, """
                data: {"type":"response.function_call_arguments.delta","sequence_number":1,"item_id":"fc_search","output_index":1,"delta":"{\\"query\\":\\"gol","call_id":"call_search"}

                data: {"type":"response.function_call_arguments.done","sequence_number":2,"item_id":"fc_search","output_index":1,"arguments":"{\\"query\\":\\"gollek\\"}","call_id":"call_search"}

                data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp-1","output_text":"I will search.","metadata":{"gollek_stream":{"finish_reason":"tool_calls"}},"usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6}}}

                data: [DONE]
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentStreamAccumulator.Snapshot snapshot = client.streamResponse(
                Map.of("model", "demo", "input", "search"),
                null);

        assertEquals("I will search.", snapshot.outputText());
        assertEquals("tool_calls", snapshot.finishReason());
        assertEquals(6, snapshot.usage().totalTokens());
        assertTrue(snapshot.completed());
        assertEquals(1, snapshot.toolCalls().size());
        AgentStreamEvent.ToolCall call = snapshot.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals("call_search", call.callId());
        assertEquals("gollek", call.argumentsMap().get("query"));
        assertEquals(URI.create("http://localhost:8080/v1/responses"), httpClient.lastRequest().uri());
    }

    @Test
    void propagatesAgentTraceHeaders() throws Exception {
        FakeHttpClient httpClient = new FakeHttpClient(200, """
                {"id":"resp-1","output_text":"hello"}
                """);
        AgentIntegrationClient client = client(httpClient);
        AgentRequestOptions options = AgentRequestOptions.builder()
                .requestId("req-123")
                .traceId("trace-456")
                .sessionId("session-789")
                .userId("user-abc")
                .header("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
                .build();

        client.createResponse(Map.of("model", "demo", "input", "hello"), options);

        HttpHeaders headers = httpClient.lastRequest().headers();
        assertEquals("req-123", headers.firstValue(AgentRequestOptions.REQUEST_ID_HEADER).orElseThrow());
        assertEquals("trace-456", headers.firstValue(AgentRequestOptions.TRACE_ID_HEADER).orElseThrow());
        assertEquals("session-789", headers.firstValue(AgentRequestOptions.SESSION_ID_HEADER).orElseThrow());
        assertEquals("user-abc", headers.firstValue(AgentRequestOptions.USER_ID_HEADER).orElseThrow());
        assertEquals("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00",
                headers.firstValue("traceparent").orElseThrow());
    }

    @Test
    void mapsHttpErrorsToAgentIntegrationException() {
        FakeHttpClient httpClient = new FakeHttpClient(400, """
                {"error":{"message":"invalid tools"}}
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentIntegrationException exception = assertThrows(
                AgentIntegrationException.class,
                () -> client.validateTools(Map.of("tools", "bad")));

        assertEquals("SDK_ERR_AGENT_INTEGRATION", exception.getErrorCode());
    }

    @Test
    void mapsStreamHttpErrorsToAgentIntegrationException() {
        FakeHttpClient httpClient = FakeHttpClient.stream(500, """
                {"error":{"message":"stream failed"}}
                """);
        AgentIntegrationClient client = client(httpClient);

        AgentIntegrationException exception = assertThrows(
                AgentIntegrationException.class,
                () -> client.streamChatCompletion(Map.of("model", "demo"), null));

        assertEquals("SDK_ERR_AGENT_INTEGRATION", exception.getErrorCode());
        assertTrue(exception.getMessage().contains("500"));
    }

    private AgentIntegrationClient client(FakeHttpClient httpClient) {
        return new AgentIntegrationClient(
                httpClient,
                new ObjectMapper(),
                "http://localhost:8080/",
                "test-key");
    }

    private static String capabilitiesJson() {
        return """
                {
                  "object":"gollek.agent_capabilities",
                  "version":"v1",
                  "contract_version":"v1",
                  "supported_contract_versions":["v1"],
                  "service_role":"inference_serving_engine",
                  "feature_negotiation":{
                    "mode":"feature_flags",
                    "feature_namespace":"gollek.agent.compatibility",
                    "contract_version":"v1",
                    "supported_contract_versions":["v1"],
                    "required_features":[
                      "openai_chat_completions",
                      "openai_chat_streaming",
                      "openai_responses",
                      "openai_responses_streaming",
                      "openai_embeddings",
                      "model_capability_matrix",
                      "agent_capabilities",
                      "agent_contract",
                      "agent_feature_negotiation",
                      "agent_readiness_issue_catalog",
                      "agent_preflight",
                      "agent_request_validation",
                      "agent_tool_contract_validation",
                      "request_trace_context",
                      "mcp_tool_discovery",
                      "rag_context"
                    ],
                    "optional_features":["stream_usage_reporting"]
                  },
                  "agent_boundary":{
                    "gollek_owns":["model serving"],
                    "agent_orchestrator_owns":["tool execution loops"]
                  },
                  "compatibility":[
                    "openai_chat_completions",
                    "openai_chat_streaming",
                    "openai_responses",
                    "openai_responses_streaming",
                    "openai_embeddings",
                    "model_capability_matrix",
                    "agent_capabilities",
                    "agent_contract",
                    "agent_feature_negotiation",
                    "agent_readiness_issue_catalog",
                    "agent_preflight",
                    "agent_request_validation",
                    "agent_tool_contract_validation",
                    "request_trace_context",
                    "mcp_tool_discovery",
                    "rag_context"
                  ],
                  "endpoints":{
                    "openai_chat_completions":"/v1/chat/completions",
                    "openai_responses":"/v1/responses",
                    "openai_embeddings":"/v1/embeddings",
                    "model_capabilities":"/v1/models/{id}/capabilities",
                    "agent_contract":"/v1/agent/contract",
                    "agent_readiness_issues":"/v1/agent/readiness/issues",
                    "agent_preflight":"/v1/agent/preflight",
                    "agent_validation":"/v1/agent/validate",
                    "agent_tool_validation":"/v1/agent/tools/validate",
                    "mcp_tools":"/v1/mcp/tools"
                  },
                  "auth":{
                    "x_api_key_header":"X-API-Key",
                    "authorization_header":"Bearer token"
                  },
                  "traceability":{"request_id_header":"X-Gollek-Request-Id"}
                }
                """;
    }

    private static String readinessIssueCatalogJson() {
        return """
                {
                  "object":"gollek.agent_readiness_issue_catalog",
                  "version":"v1",
                  "service_role":"inference_serving_engine",
                  "boundary":{
                    "validation_only":true,
                    "model_invoked":false,
                    "tool_execution":false,
                    "retrieval_execution":false,
                    "tool_authorization":false
                  },
                  "count":2,
                  "items":[{
                    "code":"TOOL_DEFINITIONS_INVALID",
                    "area":"tool_validation",
                    "default_severity":"blocking",
                    "summary":"Tool definitions are invalid.",
                    "remediation":"Fix the OpenAI-compatible tool schema before serving the route."
                  },{
                    "code":"AGENT_REQUEST_INVALID",
                    "area":"request_validation",
                    "default_severity":"blocking",
                    "summary":"The agent request is invalid.",
                    "remediation":"Fix the request model, messages/input, tools, or RAG context before serving."
                  }],
                  "by_code":{},
                  "by_area":{}
                }
                """;
    }

    private static String modelCapabilitiesJson() {
        return """
                {
                  "model_id":"demo-model",
                  "available":true,
                  "api_contract":{
                    "chat_completions":true,
                    "responses":true,
                    "system_prompt":true,
                    "tools_request_schema":true,
                    "rag_context_injection":true
                  },
                  "openai_compatibility":{
                    "chat_completions":true,
                    "responses":true
                  },
                  "inference":{
                    "completion":true,
                    "system_prompt":true
                  },
                  "tooling":{
                    "tool_definitions":true,
                    "tool_execution":false
                  },
                  "rag":{
                    "context_injection":true,
                    "retrieval_policy":false,
                    "vector_store_ownership":false
                  }
                }
                """;
    }

    private static String embeddingModelCapabilitiesJson() {
        return """
                {
                  "model_id":"demo-embed",
                  "available":true,
                  "api_contract":{
                    "embeddings_endpoint":true
                  },
                  "openai_compatibility":{
                    "embeddings":true
                  },
                  "embeddings":{
                    "generation":true,
                    "endpoint":"/v1/embeddings",
                    "openai_compatible":true,
                    "dimensions":768,
                    "encoding_formats":["float","base64"],
                    "input_aliases":["input","inputs"],
                    "batch_inputs":true,
                    "metadata_passthrough":true,
                    "retrieval_policy":false,
                    "vector_store_ownership":false
                  },
                  "modalities":{
                    "embeddings":true
                  },
                  "rag":{
                    "retrieval_policy":false,
                    "vector_store_ownership":false
                  },
                  "tooling":{
                    "tool_execution":false
                  }
                }
                """;
    }

    private static String mcpToolsJson() {
        return """
                {
                  "available":true,
                  "boundary":{"role":"discovery_only","tool_execution":false},
                  "tools":[{
                    "type":"function",
                    "execution":false,
                    "function":{"name":"mcp_lookup","parameters":{"type":"object"}}
                  }]
                }
                """;
    }

    private static String toolValidationJson() {
        return """
                {
                  "object":"gollek.tool_contract_validation",
                  "valid":true,
                  "model_invoked":false,
                  "normalized":[],
                  "warnings":[],
                  "boundary":{
                    "validation_only":true,
                    "tool_execution":false,
                    "tool_authorization":false
                  }
                }
                """;
    }

    private static String embeddingRequestValidationJson() {
        return """
                {
                  "object":"gollek.agent_validation",
                  "surface":"embeddings",
                  "valid":true,
                  "model_invoked":false,
                  "normalized":{
                    "model":"demo-embed",
                    "input_count":1,
                    "input_lengths":[19],
                    "requested_dimensions":768,
                    "encoding_format":"float",
                    "parameter_keys":["dimensions","encoding_format","metadata"],
                    "metadata":{"tenant":"agent-project"},
                    "rag":{
                      "embedding_generation":true,
                      "retrieval_execution":false,
                      "retrieval_policy_owned_by":"agent_orchestrator",
                      "vector_store_owned_by":"agent_orchestrator"
                    }
                  },
                  "boundary":{
                    "validation_only":true,
                    "tool_execution":false,
                    "retrieval_execution":false
                  }
                }
                """;
    }

    private static String requestValidationJson() {
        return """
                {
                  "object":"gollek.agent_validation",
                  "surface":"chat",
                  "valid":true,
                  "model_invoked":false,
                  "normalized":{
                    "model":"demo-model",
                    "tool_contract":{"valid":true}
                  },
                  "boundary":{
                    "validation_only":true,
                    "tool_execution":false,
                    "retrieval_execution":false
                  }
                }
                """;
    }

    private static String servingContractJson() {
        return """
                {
                  "object":"gollek.agent_contract",
                  "version":"v1",
                  "contract_version":"v1",
                  "supported_contract_versions":["v1"],
                  "service_role":"inference_serving_engine",
                  "compatibility":[
                    "openai_chat_completions",
                    "openai_chat_streaming",
                    "openai_responses",
                    "openai_responses_streaming",
                    "openai_embeddings",
                    "model_capability_matrix",
                    "agent_capabilities",
                    "agent_contract",
                    "agent_feature_negotiation",
                    "agent_readiness_issue_catalog",
                    "agent_preflight",
                    "agent_request_validation",
                    "agent_tool_contract_validation",
                    "request_trace_context",
                    "mcp_tool_discovery",
                    "rag_context"
                  ],
                  "feature_negotiation":{
                    "mode":"feature_flags",
                    "feature_namespace":"gollek.agent.compatibility",
                    "contract_version":"v1",
                    "supported_contract_versions":["v1"],
                    "required_features":[
                      "openai_chat_completions",
                      "openai_chat_streaming",
                      "openai_responses",
                      "openai_responses_streaming",
                      "openai_embeddings",
                      "model_capability_matrix",
                      "agent_capabilities",
                      "agent_contract",
                      "agent_feature_negotiation",
                      "agent_readiness_issue_catalog",
                      "agent_preflight",
                      "agent_request_validation",
                      "agent_tool_contract_validation",
                      "request_trace_context",
                      "mcp_tool_discovery",
                      "rag_context"
                    ],
                    "optional_features":["stream_usage_reporting"]
                  },
                  "boundary":{
                    "gollek_owns":[
                      "model_serving",
                      "provider_routing",
                      "system_prompt_mapping",
                      "tool_schema_ingestion",
                      "tool_contract_validation",
                      "mcp_registry_discovery",
                      "embedding_generation",
                      "rag_context_injection"
                    ],
                    "agent_orchestrator_owns":[
                      "planning",
                      "memory_policy",
                      "retrieval_policy",
                      "vector_store_ownership",
                      "tool_authorization",
                      "tool_execution",
                      "tool_result_loop",
                      "workflow_state"
                    ],
                    "tool_execution":false,
                    "retrieval_execution":false
                  },
                  "streaming":{
                    "done_sentinel":"[DONE]",
                    "chat_completions_events":["chat.completion.chunk","[DONE]"],
                    "responses_events":["response.output_text.delta","response.completed","[DONE]"]
                  },
                  "endpoints":{
                    "chat_completions":{"method":"POST","path":"/v1/chat/completions"},
                    "responses":{"method":"POST","path":"/v1/responses"},
                    "embeddings":{"method":"POST","path":"/v1/embeddings"},
                    "model_capabilities":{"method":"GET","path":"/v1/models/{id}/capabilities"},
                    "mcp_tools":{"method":"GET","path":"/v1/mcp/tools?compat=openai"},
                    "agent_capabilities":{"method":"GET","path":"/v1/agent/capabilities"},
                    "agent_contract":{"method":"GET","path":"/v1/agent/contract"},
                    "agent_readiness_issues":{"method":"GET","path":"/v1/agent/readiness/issues"},
                    "agent_preflight":{"method":"POST","path":"/v1/agent/preflight"},
                    "agent_validation":{"method":"POST","path":"/v1/agent/validate?surface={surface}"},
                    "agent_tool_validation":{"method":"POST","path":"/v1/agent/tools/validate"}
                  },
                  "schemas":{
                    "chat_completions_request":{
                      "required":["model","messages"],
                      "properties":{"model":{},"messages":{},"tools":{},"rag_context":{}}
                    },
                    "responses_request":{
                      "required":["model","input"],
                      "properties":{"model":{},"input":{},"tools":{},"rag_context":{}}
                    }
                  }
                }
                """;
    }

    private static String requestBody(HttpRequest request) throws Exception {
        BodyCaptureSubscriber subscriber = new BodyCaptureSubscriber();
        request.bodyPublisher().orElseThrow().subscribe(subscriber);
        return subscriber.body();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    private static final class FakeHttpClient extends HttpClient {
        private final List<FakeExchange> exchanges;
        private final List<HttpRequest> requests = new ArrayList<>();
        private int nextExchange;
        private HttpRequest lastRequest;

        private FakeHttpClient(int statusCode, String body) {
            this(statusCode, body, false);
        }

        private FakeHttpClient(int statusCode, String body, boolean streamResponse) {
            this(List.of(new FakeExchange(statusCode, body, streamResponse)));
        }

        private FakeHttpClient(List<FakeExchange> exchanges) {
            this.exchanges = List.copyOf(exchanges);
        }

        private static FakeHttpClient stream(int statusCode, String body) {
            return new FakeHttpClient(statusCode, body, true);
        }

        private static FakeHttpClient sequence(FakeExchange... exchanges) {
            return new FakeHttpClient(List.of(exchanges));
        }

        private static FakeExchange json(int statusCode, String body) {
            return new FakeExchange(statusCode, body, false);
        }

        private HttpRequest lastRequest() {
            return lastRequest;
        }

        private List<HttpRequest> requests() {
            return List.copyOf(requests);
        }

        @Override
        public Optional<CookieHandler> cookieHandler() {
            return Optional.empty();
        }

        @Override
        public Optional<Duration> connectTimeout() {
            return Optional.empty();
        }

        @Override
        public Redirect followRedirects() {
            return Redirect.NEVER;
        }

        @Override
        public Optional<ProxySelector> proxy() {
            return Optional.empty();
        }

        @Override
        public SSLContext sslContext() {
            return null;
        }

        @Override
        public SSLParameters sslParameters() {
            return null;
        }

        @Override
        public Optional<Authenticator> authenticator() {
            return Optional.empty();
        }

        @Override
        public Version version() {
            return Version.HTTP_1_1;
        }

        @Override
        public Optional<Executor> executor() {
            return Optional.empty();
        }

        @Override
        @SuppressWarnings("unchecked")
        public <T> HttpResponse<T> send(HttpRequest request, HttpResponse.BodyHandler<T> responseBodyHandler)
                throws IOException, InterruptedException {
            if (nextExchange >= exchanges.size()) {
                throw new AssertionError("Unexpected request: " + request.uri());
            }
            FakeExchange exchange = exchanges.get(nextExchange++);
            lastRequest = request;
            requests.add(request);
            Object responseBody = exchange.streamResponse() ? exchange.body().lines() : exchange.body();
            return new FakeResponse<>(request, exchange.statusCode(), (T) responseBody);
        }

        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(
                HttpRequest request,
                HttpResponse.BodyHandler<T> responseBodyHandler) {
            throw new UnsupportedOperationException();
        }

        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(
                HttpRequest request,
                HttpResponse.BodyHandler<T> responseBodyHandler,
                HttpResponse.PushPromiseHandler<T> pushPromiseHandler) {
            throw new UnsupportedOperationException();
        }
    }

    private record FakeExchange(int statusCode, String body, boolean streamResponse) {
    }

    private record FakeResponse<T>(HttpRequest request, int statusCode, T body) implements HttpResponse<T> {
        @Override
        public Optional<HttpResponse<T>> previousResponse() {
            return Optional.empty();
        }

        @Override
        public HttpHeaders headers() {
            return HttpHeaders.of(Map.of(), (name, value) -> true);
        }

        @Override
        public Optional<SSLSession> sslSession() {
            return Optional.empty();
        }

        @Override
        public URI uri() {
            return request.uri();
        }

        @Override
        public HttpClient.Version version() {
            return HttpClient.Version.HTTP_1_1;
        }
    }

    private static final class BodyCaptureSubscriber implements Flow.Subscriber<ByteBuffer> {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private final CountDownLatch done = new CountDownLatch(1);
        private Throwable error;

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            subscription.request(Long.MAX_VALUE);
        }

        @Override
        public void onNext(ByteBuffer item) {
            byte[] bytes = new byte[item.remaining()];
            item.get(bytes);
            out.writeBytes(bytes);
        }

        @Override
        public void onError(Throwable throwable) {
            error = throwable;
            done.countDown();
        }

        @Override
        public void onComplete() {
            done.countDown();
        }

        private String body() throws Exception {
            if (!done.await(2, TimeUnit.SECONDS)) {
                throw new AssertionError("Timed out waiting for request body");
            }
            if (error != null) {
                throw new AssertionError("Failed to read request body", error);
            }
            return out.toString(StandardCharsets.UTF_8);
        }
    }
}
