package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentCommandTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void capabilitiesCallsServerEndpointAndCanRequireServingBoundary() throws Exception {
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startGetServer("/v1/agent/capabilities", capabilitiesResponse(), requestHeaders);

        AgentCommand.Capabilities command = new AgentCommand.Capabilities();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requestId = "req-capabilities";
        command.requireServingBoundary = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent capabilities"));
        assertTrue(captured.out().contains("OpenAI surfaces: chat=yes, responses=yes, embeddings=yes"));
        assertTrue(captured.out().contains(
                "Agent surfaces: model_capabilities=yes, issue_catalog=yes, mcp_tool_discovery=yes, rag_context=yes"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
        assertEquals("Bearer test-key", requestHeaders.get().getFirst("Authorization"));
        assertEquals("req-capabilities", requestHeaders.get().getFirst("X-Gollek-Request-Id"));
    }

    @Test
    void issueCatalogCallsServerEndpointAndPrintsRemediation() throws Exception {
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startGetServer("/v1/agent/readiness/issues", issueCatalogResponse(), requestHeaders);

        AgentCommand.IssueCatalog command = new AgentCommand.IssueCatalog();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.area = "tool_validation";
        command.requireServingBoundary = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent readiness issue catalog"));
        assertTrue(captured.out().contains("Endpoint: " + baseUrl + "/v1/agent/readiness/issues"));
        assertTrue(captured.out().contains("Entries: total=2, shown=1"));
        assertTrue(captured.out().contains("Boundary: validation_only=yes, model_invoked=no, tool_execution=no"));
        assertTrue(captured.out().contains("TOOL_DEFINITIONS_INVALID [tool_validation, blocking]"));
        assertTrue(captured.out().contains(
                "remediation: Fix the OpenAI-compatible tool schema before serving the route."));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
        assertEquals("Bearer test-key", requestHeaders.get().getFirst("Authorization"));
    }

    @Test
    void contractCanFailCiWhenServingBoundaryIsMissing() throws Exception {
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startGetServer("/v1/agent/contract", invalidContractResponse(), requestHeaders);

        AgentCommand.Contract command = new AgentCommand.Contract();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireServingBoundary = true;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent contract"));
        assertTrue(captured.out().contains("Serving contract issues:"));
        assertTrue(captured.out().contains("service_role must be inference_serving_engine"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
    }

    @Test
    void modelCapabilitiesCallsModelRouteAndCanRequireServingRoute() throws Exception {
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startGetServer("/v1/models/demo-model/capabilities", modelCapabilitiesResponse(), requestHeaders);

        AgentCommand.ModelCapabilities command = new AgentCommand.ModelCapabilities();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.requireServingBoundary = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent model capabilities"));
        assertTrue(captured.out().contains("Model: demo-model"));
        assertTrue(captured.out().contains("OpenAI surfaces: chat=yes, chat_streaming=yes, responses=yes"));
        assertTrue(captured.out().contains("Tools: definitions=yes"));
        assertTrue(captured.out().contains("RAG: context_injection=yes"));
        assertTrue(captured.out().contains("Embeddings: generation=yes, endpoint=/v1/embeddings, dimensions=768"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
        assertEquals("Bearer test-key", requestHeaders.get().getFirst("Authorization"));
    }

    @Test
    void toolsValidatePostsInlineToolsAndPrintsBoundary() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/tools/validate",
                toolValidationResponse(false),
                requestBody,
                requestHeaders);

        AgentCommand.ToolsValidate command = new AgentCommand.ToolsValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireServingBoundary = true;
        command.toolsJson = """
                {
                  "type": "function",
                  "function": {
                    "name": "search_context",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "query": {"type": "string"}
                      },
                      "required": ["query"]
                    }
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent tool validation: valid"));
        assertTrue(captured.out().contains("Tools: count=1, names=search_context"));
        assertTrue(captured.out().contains("Boundary: validation_only=yes, model_invoked=no, tool_execution=no"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertTrue(sent.path("tools").isArray());
        assertEquals("search_context", sent.path("tools").get(0).path("function").path("name").asText());
    }

    @Test
    void toolsValidateCanFailCiOnWarnings() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/tools/validate",
                toolValidationResponse(true),
                requestBody,
                new AtomicReference<>());

        AgentCommand.ToolsValidate command = new AgentCommand.ToolsValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireClean = true;
        command.toolsJson = """
                [
                  {
                    "type": "function",
                    "function": {
                      "name": "search_context",
                      "parameters": {"type": "object"}
                    }
                  }
                ]
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Warnings: 1"));
        assertTrue(captured.out().contains("schema_feature_may_be_ignored"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals(1, sent.path("tools").size());
    }

    @Test
    void requestValidatePostsGeneratedChatRequestAndPrintsBoundary() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/validate",
                requestValidationResponse(0),
                requestBody,
                requestHeaders);

        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireServingBoundary = true;
        command.modelId = "demo-model";
        command.prompt = "Validate this agent route.";
        command.toolFiles.add(writeTempJson("""
                [
                  {
                    "type": "function",
                    "function": {
                      "name": "search_context",
                      "parameters": {"type": "object"}
                    }
                  }
                ]
                """));
        command.ragContexts.add("Caller-owned context.");

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent request validation: valid"));
        assertTrue(captured.out().contains("Request: streaming=no, messages=2, inputs=0, tools=1"));
        assertTrue(captured.out().contains("RAG: injected=yes, items=1"));
        assertTrue(captured.out().contains("Boundary: validation_only=yes, model_invoked=no"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertTrue(sent.path("messages").isArray());
        assertEquals("search_context", sent.path("tools").get(0).path("function").path("name").asText());
        assertEquals("Caller-owned context.", sent.path("rag_context").get(0).path("text").asText());
    }

    @Test
    void requestValidateReportsEmbeddingDryRunBoundary() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/validate",
                embeddingValidationResponse(),
                requestBody,
                new AtomicReference<>());

        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireServingBoundary = true;
        command.requestJson = """
                {
                  "model": "demo-embed",
                  "input": ["Index this document"],
                  "dimensions": 384,
                  "encoding_format": "float",
                  "metadata": {"tenant": "agent-project"}
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Surface: embeddings"));
        assertTrue(captured.out().contains(
                "Embeddings: inputs=1, input_lengths=19, requested_dimensions=384, encoding=float, metadata_keys=tenant"));
        assertTrue(captured.out().contains(
                "Embedding boundary: generation=yes, retrieval_execution=no, retrieval_policy_owner=agent_orchestrator, vector_store_owner=agent_orchestrator"));
        assertTrue(captured.out().contains("Boundary: validation_only=yes, model_invoked=no"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-embed", sent.path("model").asText());
        assertEquals("Index this document", sent.path("input").get(0).asText());
        assertEquals(384, sent.path("dimensions").asInt());
        assertEquals("float", sent.path("encoding_format").asText());
        assertFalse(sent.has("messages"));
    }

    @Test
    void requestValidateRejectsConflictingTopLevelAndNestedRequestRouteBeforeHttpCall() throws Exception {
        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = "http://127.0.0.1:1";
        command.apiKey = "test-key";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responses",
                  "feature_profile": "chat_agent",
                  "request": {
                    "model": "other-model",
                    "surface": "chat",
                    "feature_profile": "embedding_rag",
                    "input": "Route should be consistent."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Agent request validation failed"));
        assertTrue(captured.err().contains("Agent request route conflict"));
        assertTrue(captured.err().contains("model top-level=demo-model request=other-model"));
        assertTrue(captured.err().contains("surface top-level=responses request=chat"));
        assertTrue(captured.err().contains("feature_profile top-level=chat_agent request=embedding_rag"));
    }

    @Test
    void requestValidatePrintsStructuredRouteConflictInJsonMode() throws Exception {
        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = "http://127.0.0.1:1";
        command.apiKey = "test-key";
        command.json = true;
        command.requestJson = conflictingRouteTemplate();

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertEquals("", captured.err());
        assertRouteConflictJson("gollek.agent_request_validation_error", captured.out());
    }

    @Test
    void requestValidateCliOverridesNormalizeNestedRequestRoute() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/validate",
                requestValidationResponse(1),
                requestBody,
                new AtomicReference<>());

        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.surface = "responses";
        command.featureProfile = "responses_agent";
        command.requestJson = """
                {
                  "model": "file-model",
                  "surface": "chat",
                  "featureProfile": "chat_agent",
                  "request": {
                    "model": "request-model",
                    "surface": "chat",
                    "featureProfile": "chat_agent",
                    "input": "Override this route."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertEquals("responses", sent.path("surface").asText());
        assertEquals("responses_agent", sent.path("feature_profile").asText());
        assertEquals("Override this route.", sent.path("input").asText());
        assertFalse(sent.has("featureProfile"));
    }

    @Test
    void templateSchemaPrintsCanonicalJsonSchema() throws Exception {
        AgentTemplateSchemaCommand command = new AgentTemplateSchemaCommand();
        command.json = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode schema = MAPPER.readTree(captured.out());
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.path("$schema").asText());
        assertEquals("https://gollek-ai.github.io/schemas/agent-route-template.schema.json",
                schema.path("$id").asText());
        assertEquals("object", schema.path("type").asText());
        assertEquals("request", schema.path("required").get(0).asText());
        assertEquals("chat", schema.path("properties").path("surface").path("enum").get(0).asText());
        assertEquals("agent_serving",
                schema.path("properties").path("feature_profile").path("enum").get(0).asText());
        assertEquals("clean",
                schema.path("x-gollek").path("supported_preflight_policies").get(2).asText());
        assertTrue(schema.path("$defs").path("openai_request").path("additionalProperties").asBoolean());
        assertTrue(schema.path("x-gollek").path("boundary").path("local_only").asBoolean());
        assertFalse(schema.path("x-gollek").path("boundary").path("tool_execution").asBoolean());
        assertFalse(schema.path("x-gollek").path("boundary").path("retrieval_execution").asBoolean());
    }

    @Test
    void templateSchemaHumanReportPrintsBoundaryAndSupportedValues() throws Exception {
        AgentTemplateSchemaCommand command = new AgentTemplateSchemaCommand();

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent route template schema"));
        assertTrue(captured.out().contains("Surfaces: chat,responses,embeddings"));
        assertTrue(captured.out().contains(
                "Feature profiles: agent_serving,chat_agent,responses_agent,embedding_rag,mcp_tools"));
        assertTrue(captured.out().contains("Preflight policies: readiness_only,requested_route,clean"));
        assertTrue(captured.out().contains(
                "Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no"));
        assertTrue(captured.out().contains("Use --json to print the raw JSON Schema."));
    }

    @Test
    void templateExamplePrintsValidChatTemplateJson() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.json = true;
        command.modelId = "demo-model";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode template = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_route_template", template.path("object").asText());
        assertEquals("demo-model", template.path("model").asText());
        assertEquals("chat", template.path("surface").asText());
        assertEquals("chat_agent", template.path("feature_profile").asText());
        assertEquals("requested_route", template.path("preflight_policy").path("profile").asText());
        assertTrue(template.path("discover_mcp_tools").asBoolean());
        assertTrue(template.path("validate_tools").asBoolean());
        assertEquals(2, template.path("request").path("messages").size());
        assertEquals(1, template.path("request").path("tools").size());
        assertEquals(1, template.path("request").path("rag_context").size());
        assertTemplateCheckAccepts(captured.out(), "chat_agent");
    }

    @Test
    void templateExampleDefaultsEmbeddingRagProfileToEmbeddingsSurface() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.json = true;
        command.modelId = "demo-embed";
        command.featureProfile = "embedding_rag";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        JsonNode template = MAPPER.readTree(captured.out());
        assertEquals("demo-embed", template.path("model").asText());
        assertEquals("embeddings", template.path("surface").asText());
        assertEquals("embedding_rag", template.path("feature_profile").asText());
        assertFalse(template.path("discover_mcp_tools").asBoolean());
        assertFalse(template.path("validate_tools").asBoolean());
        assertEquals(2, template.path("request").path("input").size());
        assertEquals(768, template.path("request").path("dimensions").asInt());
        assertEquals("float", template.path("request").path("encoding_format").asText());
        assertFalse(template.path("request").has("tools"));
        assertTemplateCheckAccepts(captured.out(), "embedding_rag");
    }

    @Test
    void templateExampleIdGeneratesCatalogRoute() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.json = true;
        command.modelId = "demo-embed";
        command.exampleId = "embedding-rag";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode template = MAPPER.readTree(captured.out());
        assertEquals("demo-embed", template.path("model").asText());
        assertEquals("embeddings", template.path("surface").asText());
        assertEquals("embedding_rag", template.path("feature_profile").asText());
        assertFalse(template.path("discover_mcp_tools").asBoolean());
        assertFalse(template.path("validate_tools").asBoolean());
        assertTemplateCheckAccepts(captured.out(), "embedding_rag");
    }

    @Test
    void templateExampleIdRejectsConflictingOverrides() {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.exampleId = "embedding_rag";
        command.surface = "chat";

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, command::call);

        assertEquals("Example id embedding_rag requires surface embeddings, but --surface was chat",
                exception.getMessage());
    }

    @Test
    void templateExampleIdRejectsUnsupportedIdWithAllowedValues() {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.exampleId = "retrieval";

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, command::call);

        assertEquals("Unsupported example id: retrieval. Use one of: chat, responses, embedding_rag, mcp_tools",
                exception.getMessage());
    }

    @Test
    void templateExampleOutputWritesRouteTemplateJson() throws Exception {
        Path output = Files.createTempDirectory("gollek agent route ")
                .resolve("route template.json");
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.exampleId = "mcp_tools";
        command.outputPath = output;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent route template example"));
        Path normalizedOutput = output.toAbsolutePath().normalize();
        assertTrue(captured.out().contains("Output file: " + normalizedOutput));
        assertTrue(captured.out().contains("Template check: gollek agent template-check --request-file '"
                + normalizedOutput + "' --json"));
        String savedJson = Files.readString(output);
        JsonNode template = MAPPER.readTree(savedJson);
        assertEquals("chat", template.path("surface").asText());
        assertEquals("mcp_tools", template.path("feature_profile").asText());
        assertTrue(template.path("discover_mcp_tools").asBoolean());
        assertTemplateCheckAccepts(savedJson, "mcp_tools");
    }

    @Test
    void templateExampleOutputWritesCatalogJson() throws Exception {
        Path output = Files.createTempDirectory("gollek-agent-template-catalog-")
                .resolve("template-catalog.json");
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.list = true;
        command.outputPath = output;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent route template examples"));
        assertTrue(captured.out().contains("Output file: " + output.toAbsolutePath().normalize()));
        JsonNode catalog = MAPPER.readTree(Files.readString(output));
        assertEquals("gollek.agent_route_template_examples", catalog.path("object").asText());
        assertEquals(4, catalog.path("examples").size());
    }

    @Test
    void templateExampleOutputRejectsExistingFileWithoutOverwrite() throws Exception {
        Path output = Files.createTempFile("gollek-agent-route-existing-", ".json");
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.exampleId = "chat";
        command.outputPath = output;

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, command::call);

        assertEquals("Output file already exists: " + output.toAbsolutePath().normalize()
                        + ". Use --overwrite to replace it.",
                exception.getMessage());
    }

    @Test
    void templateExampleOutputCanOverwriteExistingFile() throws Exception {
        Path output = Files.createTempFile("gollek-agent-route-overwrite-", ".json");
        Files.writeString(output, "{\"old\":true}");
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.exampleId = "responses";
        command.outputPath = output;
        command.overwrite = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode template = MAPPER.readTree(Files.readString(output));
        assertEquals("responses", template.path("surface").asText());
        assertEquals("responses_agent", template.path("feature_profile").asText());
        assertTrue(captured.out().contains("Output file: " + output.toAbsolutePath().normalize()));
        assertTemplateCheckAccepts(Files.readString(output), "responses_agent");
    }

    @Test
    void templateExampleListPrintsCatalogJson() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.list = true;
        command.json = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode catalog = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_route_template_examples", catalog.path("object").asText());
        assertEquals("v1", catalog.path("version").asText());
        assertEquals(4, catalog.path("examples").size());
        assertEquals("chat", catalog.path("examples").get(0).path("id").asText());
        assertEquals("gollek agent template-example --id chat --json",
                catalog.path("examples").get(0).path("command").asText());
        assertEquals("embedding_rag", catalog.path("examples").get(2).path("feature_profile").asText());
        assertEquals("gollek agent template-check --request-file <file> --json",
                catalog.path("examples").get(2).path("template_check_command").asText());
        assertTrue(catalog.path("boundary").path("local_only").asBoolean());
        assertFalse(catalog.path("boundary").path("http_request").asBoolean());
        assertFalse(catalog.path("boundary").path("tool_execution").asBoolean());
        assertFalse(catalog.path("boundary").path("retrieval_execution").asBoolean());
    }

    @Test
    void templateExampleListHumanReportPrintsCatalog() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.list = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent route template examples"));
        assertTrue(captured.out().contains("- chat: surface=chat, feature_profile=chat_agent"));
        assertTrue(captured.out().contains("command: gollek agent template-example --id chat --json"));
        assertTrue(captured.out().contains("- embedding_rag: surface=embeddings, feature_profile=embedding_rag"));
        assertTrue(captured.out().contains(
                "Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no"));
        assertTrue(captured.out().contains("Use --list --json to print the raw catalog."));
    }

    @Test
    void templateExampleHumanReportPrintsRouteAndBoundary() throws Exception {
        AgentTemplateExampleCommand command = new AgentTemplateExampleCommand();
        command.surface = "responses";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent route template example"));
        assertTrue(captured.out().contains("Route: model=demo-model, surface=responses, feature_profile=responses_agent"));
        assertTrue(captured.out().contains("Preflight policy: requested_route"));
        assertTrue(captured.out().contains("Template check: gollek agent template-check --request-file <file> --json"));
        assertTrue(captured.out().contains(
                "Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no"));
        assertTrue(captured.out().contains("Use --json to print the route template."));
    }

    @Test
    void templateCheckAcceptsConsistentRouteTemplateWithoutHttp() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "chat",
                  "feature_profile": "chat_agent",
                  "preflight_policy": {
                    "profile": "requested_route"
                  },
                  "request": {
                    "model": "demo-model",
                    "surface": "chat",
                    "feature_profile": "chat_agent",
                    "messages": [
                      {"role": "system", "content": "Use supplied context."},
                      {"role": "user", "content": "Check this route."}
                    ],
                    "tools": [
                      {
                        "type": "function",
                        "function": {
                          "name": "search_context",
                          "parameters": {"type": "object"}
                        }
                      }
                    ],
                    "rag_context": [
                      {"source": "docs", "text": "Caller-owned context."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent template check: valid"));
        assertTrue(captured.out().contains("Route: model=demo-model, surface=chat, feature_profile=chat_agent"));
        assertTrue(captured.out().contains("Preflight policy: requested_route"));
        assertTrue(captured.out().contains("messages=2"));
        assertTrue(captured.out().contains("tools=1"));
        assertTrue(captured.out().contains("rag_context=1"));
        assertTrue(captured.out().contains(
                "Boundary: local_only=yes, model_invoked=no, tool_execution=no, retrieval_execution=no, workflow_state=no"));
    }

    @Test
    void templateCheckRejectsConflictingRouteTemplate() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.requestJson = conflictingRouteTemplate();

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent template check: invalid"));
        assertTrue(captured.out().contains("Route conflict: Agent request route conflict"));
        assertTrue(captured.out().contains("[code: AGENT_ROUTE_CONFLICT]"));
        assertTrue(captured.out().contains("model top-level=demo-model request=other-model"));
        assertTrue(captured.out().contains("surface top-level=responses request=chat"));
        assertTrue(captured.out().contains("feature_profile top-level=chat_agent request=embedding_rag"));
    }

    @Test
    void templateCheckPrintsStructuredRouteConflictInJsonMode() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.json = true;
        command.requestJson = conflictingRouteTemplate();

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertEquals("", captured.err());
        assertRouteConflictJson("gollek.agent_template_check", captured.out());
        JsonNode result = MAPPER.readTree(captured.out());
        assertFalse(result.path("valid").asBoolean());
        assertEquals("AGENT_ROUTE_CONFLICT", result.path("issue_details").get(0).path("code").asText());
        assertEquals("route", result.path("issue_details").get(0).path("field").asText());
        assertTrue(result.path("boundary").path("local_only").asBoolean());
        assertFalse(result.path("boundary").path("http_request").asBoolean());
    }

    @Test
    void templateCheckCliOverridesNormalizeNestedRequestRoute() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.json = true;
        command.modelId = "demo-model";
        command.surface = "responses";
        command.featureProfile = "responses_agent";
        command.requestJson = """
                {
                  "model": "file-model",
                  "surface": "chat",
                  "featureProfile": "chat_agent",
                  "request": {
                    "model": "request-model",
                    "surface": "chat",
                    "featureProfile": "chat_agent",
                    "input": "Override this route."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode result = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_template_check", result.path("object").asText());
        assertTrue(result.path("valid").asBoolean());
        assertEquals("demo-model", result.path("route").path("model").asText());
        assertEquals("responses", result.path("route").path("surface").asText());
        assertEquals("responses_agent", result.path("route").path("feature_profile").asText());
        assertEquals(0, result.path("issues").size());
        assertEquals(0, result.path("issue_details").size());
        assertEquals("readiness_only", result.path("preflight_policy").path("profile").asText());
        assertTrue(result.path("request").path("parameters").isArray());
    }

    @Test
    void templateCheckRejectsUnsupportedSurfaceFeatureProfileAndPolicy() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.json = true;
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responsses",
                  "feature_profile": "chat-agnt",
                  "preflight_policy": "cleen",
                  "request": {
                    "input": "Reject unsupported local template settings."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode result = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_template_check", result.path("object").asText());
        assertFalse(result.path("valid").asBoolean());
        assertEquals("AGENT_ROUTE_TEMPLATE_INVALID", result.path("code").asText());
        assertTrue(result.path("error").asText().contains("unsupported surface: responsses"));
        assertEquals(3, result.path("issues").size());
        assertTrue(result.path("issues").get(0).asText().contains("unsupported surface: responsses"));
        assertTrue(result.path("issues").get(1).asText().contains("unsupported feature profile: chat_agnt"));
        assertTrue(result.path("issues").get(2).asText().contains("Unsupported preflight policy profile: cleen"));
        JsonNode issueDetails = result.path("issue_details");
        assertEquals(3, issueDetails.size());
        assertEquals("AGENT_TEMPLATE_SURFACE_UNSUPPORTED", issueDetails.get(0).path("code").asText());
        assertEquals("surface", issueDetails.get(0).path("field").asText());
        assertEquals("responsses", issueDetails.get(0).path("value").asText());
        assertEquals("responses", issueDetails.get(0).path("suggested_value").asText());
        assertEquals("chat", issueDetails.get(0).path("allowed_values").get(0).asText());
        assertEquals("AGENT_TEMPLATE_FEATURE_PROFILE_UNSUPPORTED", issueDetails.get(1).path("code").asText());
        assertEquals("feature_profile", issueDetails.get(1).path("field").asText());
        assertEquals("chat_agnt", issueDetails.get(1).path("value").asText());
        assertEquals("chat_agent", issueDetails.get(1).path("suggested_value").asText());
        assertEquals("AGENT_TEMPLATE_PREFLIGHT_POLICY_UNSUPPORTED", issueDetails.get(2).path("code").asText());
        assertEquals("preflight_policy", issueDetails.get(2).path("field").asText());
        assertEquals("cleen", issueDetails.get(2).path("value").asText());
        assertEquals("clean", issueDetails.get(2).path("suggested_value").asText());
        assertEquals("readiness_only", issueDetails.get(2).path("allowed_values").get(0).asText());
        assertTrue(result.path("preflight_policy").isObject());
        assertEquals(0, result.path("preflight_policy").size());
    }

    @Test
    void templateCheckHumanReportPrintsIssueCodes() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responsses",
                  "feature_profile": "chat-agnt",
                  "preflight_policy": "cleen",
                  "request": {
                    "input": "Reject unsupported local template settings."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertEquals("", captured.err());
        assertTrue(captured.out().contains("Gollek agent template check: invalid"));
        assertTrue(captured.out().contains("Template issues:"));
        assertTrue(captured.out().contains("[code: AGENT_TEMPLATE_SURFACE_UNSUPPORTED]"));
        assertTrue(captured.out().contains(
                "[field: surface, value: responsses, suggested: responses, allowed: chat|responses|embeddings]"));
        assertTrue(captured.out().contains("[code: AGENT_TEMPLATE_FEATURE_PROFILE_UNSUPPORTED]"));
        assertTrue(captured.out().contains(
                "field: feature_profile, value: chat_agnt, suggested: chat_agent, allowed: agent_serving|chat_agent|responses_agent|embedding_rag|mcp_tools"));
        assertTrue(captured.out().contains("[code: AGENT_TEMPLATE_PREFLIGHT_POLICY_UNSUPPORTED]"));
        assertTrue(captured.out().contains(
                "field: preflight_policy, value: cleen, suggested: clean, allowed: readiness_only|requested_route|clean"));
    }

    @Test
    void templateCheckReportsMissingModelIssueDetail() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();
        command.json = true;
        command.requestJson = """
                {
                  "surface": "chat",
                  "feature_profile": "chat_agent",
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Missing model should be structured."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        JsonNode result = MAPPER.readTree(captured.out());
        assertFalse(result.path("valid").asBoolean());
        assertTrue(result.path("error").asText().contains("missing model"));
        assertEquals(1, result.path("issues").size());
        JsonNode issue = result.path("issue_details").get(0);
        assertEquals("AGENT_TEMPLATE_MODEL_MISSING", issue.path("code").asText());
        assertEquals("model", issue.path("field").asText());
        assertFalse(issue.has("value"));
        assertFalse(issue.has("allowed_values"));
    }

    @Test
    void templateCheckRequiresSavedTemplateInput() throws Exception {
        AgentTemplateCheckCommand command = new AgentTemplateCheckCommand();

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertEquals("", captured.out());
        assertTrue(captured.err().contains("Agent template check failed"));
        assertTrue(captured.err().contains("Provide --request-json or --request-file"));
    }

    @Test
    void requestValidateCanFailCiOnNestedToolWarnings() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startPostServer(
                "/v1/agent/validate",
                requestValidationResponse(1),
                requestBody,
                new AtomicReference<>());

        AgentCommand.RequestValidate command = new AgentCommand.RequestValidate();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireClean = true;
        command.requestJson = """
                {
                  "model": "demo-model",
                  "input": "Validate this Responses payload.",
                  "tools": [
                    {
                      "type": "function",
                      "function": {
                        "name": "search_context",
                        "parameters": {"type": "object"}
                      }
                    }
                  ]
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Surface: responses"));
        assertTrue(captured.out().contains("Tool contract: valid=yes, warnings=1"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("Validate this Responses payload.", sent.path("input").asText());
    }

    @Test
    void mcpToolsDiscoversOpenAiToolsAndCanRequireBoundary() throws Exception {
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startGetServer("/v1/mcp/tools", mcpToolsResponse(false), requestHeaders);

        AgentCommand.McpTools command = new AgentCommand.McpTools();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireServingBoundary = true;
        command.requireTools = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent MCP tool discovery"));
        assertTrue(captured.out().contains("Available: yes, compat=openai, enabled_only=yes"));
        assertTrue(captured.out().contains("Boundary: discovery_only=yes, tool_execution=no"));
        assertTrue(captured.out().contains("Tools: count=1, openai_compatible=1, names=knowledge_search"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
    }

    @Test
    void mcpToolsCanFailCiWhenRequiredToolsAreMissing() throws Exception {
        String baseUrl = startGetServer("/v1/mcp/tools", mcpToolsResponse(true), new AtomicReference<>());

        AgentCommand.McpTools command = new AgentCommand.McpTools();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requireTools = true;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Tools: count=0"));
        assertTrue(captured.out().contains("MCP discovery returned no tools"));
    }

    @Test
    void readinessRunsFullChecklistAndPrintsJsonReport() throws Exception {
        AtomicReference<String> toolValidationBody = new AtomicReference<>();
        AtomicReference<String> requestValidationBody = new AtomicReference<>();
        AtomicReference<String> preflightBody = new AtomicReference<>();
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startReadinessServer(
                readyResponse(),
                toolValidationBody,
                requestValidationBody,
                preflightBody,
                requestHeaders);

        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.featureProfile = "chat_agent";
        command.json = true;
        command.requireServingBoundary = true;
        command.requireTools = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        JsonNode report = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_readiness_report", report.path("object").asText());
        assertEquals("ready", report.path("status").asText());
        assertTrue(report.path("ready").asBoolean());
        assertEquals("chat_agent", report.path("feature_profile").asText());
        assertTrue(report.path("checks").has("capabilities"));
        assertTrue(report.path("checks").has("contract"));
        assertTrue(report.path("checks").has("model_capabilities"));
        assertTrue(report.path("checks").has("mcp_tools"));
        assertTrue(report.path("checks").has("tool_validation"));
        assertTrue(report.path("checks").has("request_validation"));
        assertTrue(report.path("checks").has("preflight"));
        assertEquals(7, report.path("checked_areas").size());
        assertEquals("capabilities", report.path("checked_areas").get(0).asText());
        assertEquals("preflight", report.path("checked_areas").get(6).asText());
        assertEquals(0, report.path("issues_by_area").size());
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));

        JsonNode toolValidation = MAPPER.readTree(toolValidationBody.get());
        assertEquals("knowledge_search", toolValidation.path("tools").get(0).path("function").path("name").asText());

        JsonNode requestValidation = MAPPER.readTree(requestValidationBody.get());
        assertEquals("demo-model", requestValidation.path("model").asText());
        assertEquals("knowledge_search", requestValidation.path("tools").get(0).path("function").path("name").asText());

        JsonNode preflight = MAPPER.readTree(preflightBody.get());
        assertEquals("demo-model", preflight.path("model").asText());
        assertEquals("chat", preflight.path("surface").asText());
        assertEquals("chat_agent", preflight.path("feature_profile").asText());
        assertEquals("knowledge_search", preflight.path("request").path("tools").get(0).path("function").path("name").asText());
    }

    @Test
    void readinessCanFailWhenPreflightBlocks() throws Exception {
        String baseUrl = startReadinessServer(
                blockedResponse(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>());

        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.requireTools = true;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent readiness: blocked"));
        assertTrue(captured.out().contains("- preflight: blocked"));
        assertTrue(captured.out().contains("  blocking: agent request is not valid [code: AGENT_REQUEST_INVALID]"));
        assertTrue(captured.out().contains(
                "remediation: Fix the request model, messages/input, tools, or RAG context before serving."));
        assertTrue(captured.out().contains("  warning: mcp discovery unavailable [code: MCP_DISCOVERY_UNAVAILABLE]"));
        assertTrue(captured.out().contains(
                "remediation: Enable the MCP registry or mark MCP discovery optional for routes that do not need tools."));
        assertTrue(captured.out().contains("preflight: agent request is not valid [code: AGENT_REQUEST_INVALID]"));
    }

    @Test
    void readinessJsonGroupsIssuesByArea() throws Exception {
        String baseUrl = startReadinessServer(
                blockedResponse(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>());

        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.json = true;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        JsonNode report = MAPPER.readTree(captured.out());
        assertEquals("blocked", report.path("status").asText());
        assertEquals(7, report.path("checked_areas").size());
        assertEquals("preflight", report.path("checked_areas").get(6).asText());
        JsonNode preflightIssues = report.path("issues_by_area").path("preflight");
        assertEquals(2, preflightIssues.size());
        assertEquals("agent request is not valid", preflightIssues.get(0).asText());
        assertEquals("mcp discovery unavailable", preflightIssues.get(1).asText());
        assertEquals("AGENT_REQUEST_INVALID",
                report.path("checks").path("preflight").path("blocking_codes").get(0).asText());
        assertEquals("MCP_DISCOVERY_UNAVAILABLE",
                report.path("checks").path("preflight").path("warning_codes").get(0).asText());
        assertEquals("AGENT_REQUEST_INVALID", report.path("blocking_codes").get(0).asText());
        assertEquals("MCP_DISCOVERY_UNAVAILABLE", report.path("warning_codes").get(0).asText());
        assertEquals("AGENT_REQUEST_INVALID",
                report.path("checks").path("preflight").path("issue_hints").get(0).path("code").asText());
        assertEquals(
                "Fix the request model, messages/input, tools, or RAG context before serving.",
                report.path("issue_hints").get(0).path("remediation").asText());
    }

    @Test
    void readinessPrintsRequestValidationDetails() throws Exception {
        String baseUrl = startReadinessServer(
                readyResponseWithCheckResults(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>(),
                new AtomicReference<>());

        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Feature profile: agent_serving"));
        assertTrue(captured.out().contains("- request_validation: ready"));
        assertTrue(captured.out().contains("  details: surface=chat, model=demo-model"));
        assertTrue(captured.out().contains("  request: streaming=false, messages=2, inputs=0, tools=1"));
        assertTrue(captured.out().contains("  rag: injected=true, items=1, alias=rag_context"));
    }

    @Test
    void readinessRejectsConflictingTopLevelAndNestedRequestRouteBeforeHttpCall() throws Exception {
        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = "http://127.0.0.1:1";
        command.apiKey = "test-key";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responses",
                  "feature_profile": "chat_agent",
                  "request": {
                    "model": "other-model",
                    "surface": "chat",
                    "feature_profile": "embedding_rag",
                    "messages": [
                      {"role": "user", "content": "Route should be consistent."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Agent readiness failed"));
        assertTrue(captured.err().contains("Agent request route conflict"));
        assertTrue(captured.err().contains("model top-level=demo-model request=other-model"));
        assertTrue(captured.err().contains("surface top-level=responses request=chat"));
        assertTrue(captured.err().contains("feature_profile top-level=chat_agent request=embedding_rag"));
    }

    @Test
    void readinessPrintsStructuredRouteConflictInJsonMode() throws Exception {
        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = "http://127.0.0.1:1";
        command.apiKey = "test-key";
        command.json = true;
        command.requestJson = conflictingRouteTemplate();

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertEquals("", captured.err());
        assertRouteConflictJson("gollek.agent_readiness_error", captured.out());
    }

    @Test
    void readinessCliOverridesNormalizeNestedRequestRoute() throws Exception {
        AtomicReference<String> requestValidationBody = new AtomicReference<>();
        AtomicReference<String> preflightBody = new AtomicReference<>();
        String baseUrl = startReadinessServer(
                readyResponse(),
                new AtomicReference<>(),
                requestValidationBody,
                preflightBody,
                new AtomicReference<>());

        AgentCommand.Readiness command = new AgentCommand.Readiness();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.surface = "responses";
        command.featureProfile = "responses_agent";
        command.requestJson = """
                {
                  "model": "file-model",
                  "surface": "chat",
                  "featureProfile": "chat_agent",
                  "request": {
                    "model": "request-model",
                    "surface": "chat",
                    "featureProfile": "chat_agent",
                    "input": "Override this route."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        JsonNode requestValidation = MAPPER.readTree(requestValidationBody.get());
        assertEquals("demo-model", requestValidation.path("model").asText());
        assertEquals("responses", requestValidation.path("surface").asText());
        assertEquals("responses_agent", requestValidation.path("feature_profile").asText());
        assertFalse(requestValidation.has("featureProfile"));

        JsonNode preflight = MAPPER.readTree(preflightBody.get());
        assertEquals("demo-model", preflight.path("model").asText());
        assertEquals("responses", preflight.path("surface").asText());
        assertEquals("responses_agent", preflight.path("feature_profile").asText());
        assertEquals("demo-model", preflight.path("request").path("model").asText());
        assertEquals("responses", preflight.path("request").path("surface").asText());
        assertEquals("responses_agent", preflight.path("request").path("feature_profile").asText());
        assertFalse(preflight.path("request").has("featureProfile"));
    }

    @Test
    void preflightCallsServerEndpointAndPrintsBoundarySummary() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<Headers> requestHeaders = new AtomicReference<>();
        String baseUrl = startServer(readyResponse(), requestBody, requestHeaders);

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.featureProfile = "chat_agent";
        command.prompt = "Preflight this agent route.";
        command.noMcp = true;
        command.requestId = "req-cli";
        command.traceId = "trace-cli";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: ready"));
        assertTrue(captured.out().contains("Feature profile: chat_agent"));
        assertTrue(captured.out().contains("Selected route: model=demo-model, surface=chat, feature_profile=chat_agent"));
        assertTrue(captured.out().contains("Route comparison: match"));
        assertTrue(captured.out().contains("Boundary: validation_only=true, model_invoked=false"));
        assertEquals("test-key", requestHeaders.get().getFirst("X-API-Key"));
        assertEquals("Bearer test-key", requestHeaders.get().getFirst("Authorization"));
        assertEquals("req-cli", requestHeaders.get().getFirst("X-Gollek-Request-Id"));
        assertEquals("trace-cli", requestHeaders.get().getFirst("X-Gollek-Trace-Id"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertEquals("chat", sent.path("surface").asText());
        assertEquals("chat_agent", sent.path("feature_profile").asText());
        assertEquals(false, sent.path("discover_mcp_tools").asBoolean());
        assertEquals(false, sent.path("mcp_discovery_required").asBoolean());
        assertTrue(sent.path("request").path("messages").isArray());
    }

    @Test
    void preflightCanRequireSelectedRouteToMatchRequestedRoute() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(routeMismatchReadyResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.featureProfile = "chat_agent";
        command.noMcp = true;
        command.requireRouteMatch = true;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: ready"));
        assertTrue(captured.out().contains("Selected route: model=other-model, surface=chat, feature_profile=chat_agent"));
        assertTrue(captured.out().contains("Route comparison: mismatch fields=model"));
        assertTrue(captured.out().contains("  mismatch: model requested=demo-model selected=other-model"));
        assertTrue(captured.out().contains("Preflight gate: blocked"));
        assertTrue(captured.out().contains(
                "  blocking: selected route differs from requested route: model requested=demo-model selected=other-model"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertEquals("chat", sent.path("surface").asText());
    }

    @Test
    void preflightCanFailOnWarningsForCleanGate() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(warningReadyResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.featureProfile = "chat_agent";
        command.noMcp = true;
        command.preflightPolicy = "clean";

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: ready"));
        assertTrue(captured.out().contains("Preflight gate: blocked"));
        assertTrue(captured.out().contains("  blocking: preflight warnings are present"));
        assertTrue(captured.out().contains("Warnings:"));
        assertTrue(captured.out().contains("- mcp discovery unavailable [code: MCP_DISCOVERY_UNAVAILABLE]"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
    }

    @Test
    void preflightCanReadPolicyFromRequestJson() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(warningReadyResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "feature_profile": "chat_agent",
                  "preflight_policy": {
                    "profile": "clean"
                  },
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Validate policy from JSON."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: ready"));
        assertTrue(captured.out().contains("Preflight gate: blocked"));
        assertTrue(captured.out().contains("  blocking: preflight warnings are present"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertEquals("chat_agent", sent.path("feature_profile").asText());
        assertEquals(false, sent.path("discover_mcp_tools").asBoolean());
        assertFalse(sent.has("preflight_policy"));
    }

    @Test
    void preflightRejectsConflictingTopLevelAndNestedRequestRouteBeforeHttpCall() throws Exception {
        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.apiKey = "test-key";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responses",
                  "feature_profile": "chat_agent",
                  "request": {
                    "model": "other-model",
                    "surface": "chat",
                    "feature_profile": "embedding_rag",
                    "messages": [
                      {"role": "user", "content": "Route should be consistent."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Agent request route conflict"));
        assertTrue(captured.err().contains("model top-level=demo-model request=other-model"));
        assertTrue(captured.err().contains("surface top-level=responses request=chat"));
        assertTrue(captured.err().contains("feature_profile top-level=chat_agent request=embedding_rag"));
        assertTrue(captured.err().contains("--model, --surface, or --feature-profile"));
    }

    @Test
    void preflightPrintsStructuredRouteConflictInJsonMode() throws Exception {
        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.apiKey = "test-key";
        command.json = true;
        command.requestJson = conflictingRouteTemplate();

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertEquals("", captured.err());
        assertRouteConflictJson("gollek.agent_preflight_error", captured.out());
    }

    @Test
    void preflightCliModelOverrideNormalizesNestedRequestModel() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(readyResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "cli-model";
        command.noMcp = true;
        command.requestJson = """
                {
                  "model": "file-model",
                  "request": {
                    "model": "request-model",
                    "messages": [
                      {"role": "user", "content": "Override this route."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("cli-model", sent.path("model").asText());
        assertEquals("cli-model", sent.path("request").path("model").asText());
    }

    @Test
    void preflightFeatureProfileCanSelectEmbeddingSurface() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(embeddingReadyResponseWithCheckResults(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-embed";
        command.featureProfile = "embedding_rag";
        command.prompt = "Index this document.";
        command.noMcp = true;
        command.noToolValidation = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Surface: embeddings"));
        assertTrue(captured.out().contains("Feature profile: embedding_rag"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-embed", sent.path("model").asText());
        assertEquals("embeddings", sent.path("surface").asText());
        assertEquals("embedding_rag", sent.path("feature_profile").asText());
        assertTrue(sent.path("request").path("input").isArray());
        assertEquals("Index this document.", sent.path("request").path("input").get(0).asText());
    }

    @Test
    void preflightRequestShapeOverridesProfileDefaultSurface() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(readyResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "feature_profile": "embedding_rag",
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Validate this chat route."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Surface: chat"));
        assertTrue(captured.out().contains("Feature profile: embedding_rag"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("chat", sent.path("surface").asText());
        assertEquals("embedding_rag", sent.path("feature_profile").asText());
        assertTrue(sent.path("request").path("messages").isArray());
        assertFalse(sent.path("request").has("input"));
    }

    @Test
    void preflightPrintsServerSideCheckResults() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(readyResponseWithCheckResults(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.prompt = "Preflight this agent route.";

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: ready"));
        assertTrue(captured.out().contains("Checks: ready=5, blocked=0, skipped=1"));
        assertTrue(captured.out().contains("- capabilities: ready, requested=true"));
        assertTrue(captured.out().contains("- mcp_discovery: skipped, requested=false"));
        assertTrue(captured.out().contains("- request_validation: ready, requested=true"));
        assertTrue(captured.out().contains("  details: surface=chat, model=demo-model"));
        assertTrue(captured.out().contains("  request: streaming=false, messages=2, inputs=0, tools=1"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
    }

    @Test
    void preflightPrintsEmbeddingCheckDetails() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(embeddingReadyResponseWithCheckResults(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-embed";
        command.surface = "embeddings";
        command.prompt = "alpha";
        command.noMcp = true;
        command.noToolValidation = true;

        Captured captured = capture(command::call);

        assertEquals(0, captured.exitCode());
        assertTrue(captured.out().contains("Surface: embeddings"));
        assertTrue(captured.out().contains("- request_validation: ready, requested=true"));
        assertTrue(captured.out().contains("  details: surface=embeddings, model=demo-embed"));
        assertTrue(captured.out().contains(
                "  embeddings: inputs=2, input_lengths=5,4, requested_dimensions=768, encoding=float, metadata_keys=tenant"));
        assertTrue(captured.out().contains(
                "  embedding boundary: generation=true, retrieval_execution=false, retrieval_policy_owner=agent_orchestrator, vector_store_owner=agent_orchestrator"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-embed", sent.path("model").asText());
        assertEquals("embeddings", sent.path("surface").asText());
        assertTrue(sent.path("request").path("input").isArray());
    }

    @Test
    void preflightPrintsServerSideBlockedCheckMessages() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(blockedResponseWithCheckResults(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.modelId = "demo-model";
        command.prompt = "Preflight this blocked route.";

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        assertTrue(captured.out().contains("Gollek agent preflight: blocked"));
        assertTrue(captured.out().contains("Checks: ready=4, blocked=1, skipped=1"));
        assertTrue(captured.out().contains("- tool_validation: blocked, requested=true"));
        assertTrue(captured.out().contains("  blocking: tool definitions are not valid [code: TOOL_DEFINITIONS_INVALID]"));
        assertTrue(captured.out().contains(
                "remediation: Fix the OpenAI-compatible tool schema before serving the route."));
        assertTrue(captured.out().contains("  warning: schema portability warning [code: TOOL_SCHEMA_PORTABILITY_WARNING]"));
        assertTrue(captured.out().contains(
                "remediation: Simplify the tool schema or record the portability warning in the orchestrator."));
        assertTrue(captured.out().contains("Blocking issues:"));
        assertTrue(captured.out().contains("- tool_validation: tool definitions are not valid [code: TOOL_DEFINITIONS_INVALID]"));

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
    }

    @Test
    void preflightJsonModePrintsTypedResultAndReturnsBlockedExitCode() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        String baseUrl = startServer(blockedResponse(), requestBody, new AtomicReference<>());

        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = baseUrl;
        command.apiKey = "test-key";
        command.json = true;
        command.requestJson = """
                {
                  "model": "demo-model",
                  "surface": "responses",
                  "discover_mcp_tools": false,
                  "mcp_discovery_required": false,
                  "request": {
                    "input": "Validate this response route."
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(1, captured.exitCode());
        JsonNode result = MAPPER.readTree(captured.out());
        assertEquals("gollek.agent_preflight_result", result.path("object").asText());
        assertEquals("blocked", result.path("status").asText());
        assertFalse(result.path("ready").asBoolean());
        assertEquals("blocked", result.path("status_for_requested_route").asText());
        assertFalse(result.path("ready_for_requested_route").asBoolean());
        assertFalse(result.path("route_match_required").asBoolean());
        assertFalse(result.path("warnings_blocking").asBoolean());
        assertFalse(result.path("exit_ready").asBoolean());
        assertEquals("gollek.agent_preflight_gate", result.path("preflight_gate").path("object").asText());
        assertEquals("blocked", result.path("preflight_gate").path("status").asText());
        assertEquals("readiness_only", result.path("preflight_policy").path("profile").asText());
        assertEquals("responses", result.path("requested_route").path("surface").asText());
        assertEquals("demo-model", result.path("requested_route").path("model").asText());
        assertFalse(result.path("route_comparison").path("matches").asBoolean());
        assertTrue(result.path("route_comparison").path("mismatch_fields").isArray());
        assertEquals("agent request is not valid",
                result.path("readiness_report").path("blocking_messages").get(0).asText());

        JsonNode sent = MAPPER.readTree(requestBody.get());
        assertEquals("demo-model", sent.path("model").asText());
        assertEquals("responses", sent.path("surface").asText());
        assertEquals(false, sent.path("discover_mcp_tools").asBoolean());
        assertEquals("Validate this response route.", sent.path("request").path("input").asText());
    }

    @Test
    void preflightRequiresModelFromOptionOrRequest() throws Exception {
        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = "http://127.0.0.1:1";

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Provide --model"));
    }

    @Test
    void preflightRejectsUnknownPolicyProfileBeforeCallingServer() throws Exception {
        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = "http://127.0.0.1:1";
        command.modelId = "demo-model";
        command.preflightPolicy = "cleen";

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Unsupported preflight policy profile: cleen"));
        assertTrue(captured.err().contains("readiness_only"));
        assertTrue(captured.err().contains("requested_route"));
        assertTrue(captured.err().contains("clean"));
    }

    @Test
    void preflightRejectsUnknownPolicyProfileFromRequestJsonBeforeCallingServer() throws Exception {
        AgentCommand.Preflight command = new AgentCommand.Preflight();
        command.baseUrl = "http://127.0.0.1:1";
        command.requestJson = """
                {
                  "model": "demo-model",
                  "preflight_policy": "cleen",
                  "request": {
                    "messages": [
                      {"role": "user", "content": "Validate policy typo."}
                    ]
                  }
                }
                """;

        Captured captured = capture(command::call);

        assertEquals(2, captured.exitCode());
        assertTrue(captured.err().contains("Unsupported preflight policy profile: cleen"));
    }

    private static String conflictingRouteTemplate() {
        return """
                {
                  "model": "demo-model",
                  "surface": "responses",
                  "feature_profile": "chat_agent",
                  "request": {
                    "model": "other-model",
                    "surface": "chat",
                    "feature_profile": "embedding_rag",
                    "input": "Route should be consistent."
                  }
                }
                """;
    }

    private static void assertRouteConflictJson(String expectedObject, String json) throws Exception {
        JsonNode error = MAPPER.readTree(json);
        assertEquals(expectedObject, error.path("object").asText());
        assertEquals("AGENT_ROUTE_CONFLICT", error.path("code").asText());
        assertTrue(error.path("error").asText().contains("Agent request route conflict"));
        assertTrue(error.path("override_hint").asText().contains("--model"));
        JsonNode conflicts = error.path("conflicts");
        assertEquals(3, conflicts.size());
        assertEquals("model", conflicts.get(0).path("field").asText());
        assertEquals("demo-model", conflicts.get(0).path("top_level").asText());
        assertEquals("other-model", conflicts.get(0).path("request").asText());
        assertEquals("surface", conflicts.get(1).path("field").asText());
        assertEquals("responses", conflicts.get(1).path("top_level").asText());
        assertEquals("chat", conflicts.get(1).path("request").asText());
        assertEquals("feature_profile", conflicts.get(2).path("field").asText());
        assertEquals("chat_agent", conflicts.get(2).path("top_level").asText());
        assertEquals("embedding_rag", conflicts.get(2).path("request").asText());
    }

    private static void assertTemplateCheckAccepts(String requestJson, String expectedFeatureProfile) throws Exception {
        AgentTemplateCheckCommand check = new AgentTemplateCheckCommand();
        check.json = true;
        check.requestJson = requestJson;

        Captured captured = capture(check::call);

        assertEquals(0, captured.exitCode());
        assertEquals("", captured.err());
        JsonNode result = MAPPER.readTree(captured.out());
        assertTrue(result.path("valid").asBoolean());
        assertEquals(expectedFeatureProfile, result.path("route").path("feature_profile").asText());
        assertEquals(0, result.path("issues").size());
        assertEquals(0, result.path("issue_details").size());
    }

    private String startGetServer(
            String path,
            String response,
            AtomicReference<Headers> requestHeaders) throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(path, exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private String startReadinessServer(
            String preflightResponse,
            AtomicReference<String> toolValidationBody,
            AtomicReference<String> requestValidationBody,
            AtomicReference<String> preflightBody,
            AtomicReference<Headers> requestHeaders) throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/agent/capabilities", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            sendJson(exchange, capabilitiesResponse());
        });
        server.createContext("/v1/agent/contract", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            sendJson(exchange, validContractResponse());
        });
        server.createContext("/v1/models/demo-model/capabilities", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            sendJson(exchange, modelCapabilitiesResponse());
        });
        server.createContext("/v1/mcp/tools", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            sendJson(exchange, mcpToolsResponse(false));
        });
        server.createContext("/v1/agent/tools/validate", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            toolValidationBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            sendJson(exchange, toolValidationResponse(false));
        });
        server.createContext("/v1/agent/validate", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            requestValidationBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            sendJson(exchange, requestValidationResponse(0));
        });
        server.createContext("/v1/agent/preflight", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            preflightBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            sendJson(exchange, preflightResponse);
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private static void sendJson(com.sun.net.httpserver.HttpExchange exchange, String response)
            throws java.io.IOException {
        byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private String startPostServer(
            String path,
            String response,
            AtomicReference<String> requestBody,
            AtomicReference<Headers> requestHeaders) throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(path, exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private String startServer(
            String response,
            AtomicReference<String> requestBody,
            AtomicReference<Headers> requestHeaders) throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/agent/preflight", exchange -> {
            requestHeaders.set(exchange.getRequestHeaders());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private static String writeTempJson(String content) throws Exception {
        java.nio.file.Path path = java.nio.file.Files.createTempFile("gollek-agent-command-", ".json");
        java.nio.file.Files.writeString(path, content, StandardCharsets.UTF_8);
        return path.toString();
    }

    private static Captured capture(Callable<Integer> callable) throws Exception {
        PrintStream oldOut = System.out;
        PrintStream oldErr = System.err;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
            System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
            int exitCode = callable.call();
            return new Captured(
                    exitCode,
                    out.toString(StandardCharsets.UTF_8),
                    err.toString(StandardCharsets.UTF_8));
        } finally {
            System.setOut(oldOut);
            System.setErr(oldErr);
        }
    }

    private static String readyResponse() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "ready",
                  "ready": true,
                  "surface": "chat",
                  "model": "demo-model",
                  "feature_profile": "chat_agent",
                  "blocking_issue_count": 0,
                  "warning_count": 0,
                  "blocking_messages": [],
                  "warning_messages": [],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String warningReadyResponse() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "ready",
                  "ready": true,
                  "surface": "chat",
                  "model": "demo-model",
                  "feature_profile": "chat_agent",
                  "blocking_issue_count": 0,
                  "warning_count": 1,
                  "blocking_messages": [],
                  "warning_messages": ["mcp discovery unavailable"],
                  "warning_codes": ["MCP_DISCOVERY_UNAVAILABLE"],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String routeMismatchReadyResponse() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "ready",
                  "ready": true,
                  "surface": "chat",
                  "model": "other-model",
                  "feature_profile": "chat_agent",
                  "blocking_issue_count": 0,
                  "warning_count": 0,
                  "blocking_messages": [],
                  "warning_messages": [],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String readyResponseWithCheckResults() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "ready",
                  "ready": true,
                  "blocking_issue_count": 0,
                  "warning_count": 0,
                  "ready_check_count": 5,
                  "blocked_check_count": 0,
                  "skipped_check_count": 1,
                  "check_results": {
                    "capabilities": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "contract": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "model_route": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "mcp_discovery": {
                      "status": "skipped",
                      "ready": true,
                      "requested": false,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "tool_validation": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "request_validation": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": [],
                      "details": {
                        "surface": "chat",
                        "model": "demo-model",
                        "boundary": {
                          "validation_only": true,
                          "model_invoked": false,
                          "tool_execution": false,
                          "retrieval_execution": false
                        },
                        "request": {
                          "streaming": false,
                          "message_count": 2,
                          "input_count": 0,
                          "tool_count": 1,
                          "parameter_keys": ["model", "tools"],
                          "rag": {
                            "injected": true,
                            "items": 1,
                            "alias": "rag_context"
                          }
                        }
                      }
                    }
                  },
                  "blocking_messages": [],
                  "warning_messages": [],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String embeddingReadyResponseWithCheckResults() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "ready",
                  "ready": true,
                  "surface": "embeddings",
                  "model": "demo-embed",
                  "blocking_issue_count": 0,
                  "warning_count": 0,
                  "ready_check_count": 1,
                  "blocked_check_count": 0,
                  "skipped_check_count": 0,
                  "check_results": {
                    "request_validation": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": [],
                      "details": {
                        "surface": "embeddings",
                        "model": "demo-embed",
                        "boundary": {
                          "validation_only": true,
                          "model_invoked": false,
                          "tool_execution": false,
                          "retrieval_execution": false
                        },
                        "embedding": {
                          "input_count": 2,
                          "input_lengths": [5, 4],
                          "requested_dimensions": 768,
                          "encoding_format": "float",
                          "metadata_keys": ["tenant"],
                          "rag": {
                            "embedding_generation": true,
                            "retrieval_execution": false,
                            "retrieval_policy_owned_by": "agent_orchestrator",
                            "vector_store_owned_by": "agent_orchestrator",
                            "storage_owned_by_orchestrator": true
                          }
                        }
                      }
                    }
                  },
                  "blocking_messages": [],
                  "warning_messages": [],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String blockedResponseWithCheckResults() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "blocked",
                  "ready": false,
                  "blocking_issue_count": 1,
                  "warning_count": 1,
                  "ready_check_count": 4,
                  "blocked_check_count": 1,
                  "skipped_check_count": 1,
                  "check_results": {
                    "capabilities": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "contract": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "model_route": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "mcp_discovery": {
                      "status": "skipped",
                      "ready": true,
                      "requested": false,
                      "blocking_messages": [],
                      "warning_messages": []
                    },
                    "tool_validation": {
                      "status": "blocked",
                      "ready": false,
                      "requested": true,
                      "blocking_messages": ["tool definitions are not valid"],
                      "warning_messages": ["schema portability warning"],
                      "blocking_codes": ["TOOL_DEFINITIONS_INVALID"],
                      "warning_codes": ["TOOL_SCHEMA_PORTABILITY_WARNING"]
                    },
                    "request_validation": {
                      "status": "ready",
                      "ready": true,
                      "requested": true,
                      "blocking_messages": [],
                      "warning_messages": []
                    }
                  },
                  "blocking_messages": ["tool_validation: tool definitions are not valid"],
                  "warning_messages": ["tool_validation: schema portability warning"],
                  "blocking_codes": ["TOOL_DEFINITIONS_INVALID"],
                  "warning_codes": ["TOOL_SCHEMA_PORTABILITY_WARNING"],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String blockedResponse() {
        return """
                {
                  "object": "gollek.agent_preflight",
                  "status": "blocked",
                  "ready": false,
                  "blocking_issue_count": 1,
                  "warning_count": 1,
                  "blocking_messages": ["agent request is not valid"],
                  "warning_messages": ["mcp discovery unavailable"],
                  "blocking_codes": ["AGENT_REQUEST_INVALID"],
                  "warning_codes": ["MCP_DISCOVERY_UNAVAILABLE"],
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  }
                }
                """;
    }

    private static String toolValidationResponse(boolean withWarning) {
        String warnings = withWarning
                ? """
                  [
                    {
                      "index": 0,
                      "path": "$defs",
                      "code": "schema_feature_may_be_ignored",
                      "message": "JSON Schema keyword '$defs' may not be supported by every agent client."
                    }
                  ]
                  """
                : "[]";
        int warningCount = withWarning ? 1 : 0;
        return """
                {
                  "object": "gollek.tool_contract_validation",
                  "valid": true,
                  "model_invoked": false,
                  "tool_count": 1,
                  "normalized": [
                    {
                      "index": 0,
                      "name": "search_context",
                      "type": "function",
                      "strict": false,
                      "description_present": true,
                      "parameter_keys": ["properties", "required", "type"],
                      "parameter_schema": {
                        "type": "object",
                        "property_count": 1,
                        "required": ["query"]
                      },
                      "metadata": {},
                      "warning_count": %d
                    }
                  ],
                  "warnings": %s,
                  "boundary": {
                    "validation_only": true,
                    "tool_execution": false,
                    "tool_authorization": false
                  }
                }
                """.formatted(warningCount, warnings);
    }

    private static String requestValidationResponse(int toolWarningCount) {
        return """
                {
                  "object": "gollek.agent_validation",
                  "surface": "%s",
                  "valid": true,
                  "model_invoked": false,
                  "trace": {
                    "request_id": "req_validate_fixture",
                    "trace_id": "trace_validate_fixture"
                  },
                  "normalized": {
                    "request_id": "req_validate_fixture",
                    "trace_id": "trace_validate_fixture",
                    "model": "demo-model",
                    "streaming": false,
                    "stream_options": {
                      "include_usage": false
                    },
                    "message_count": %d,
                    "input_count": %d,
                    "tool_count": 1,
                    "tools": [
                      {
                        "name": "search_context",
                        "type": "function",
                        "strict": false,
                        "parameter_keys": ["type"]
                      }
                    ],
                    "tool_contract": {
                      "valid": true,
                      "tool_count": 1,
                      "warning_count": %d
                    },
                    "parameter_keys": ["model", "tools"],
                    "metadata": {},
                    "rag": {
                      "injected": true,
                      "items": 1,
                      "alias": "rag_context"
                    }
                  },
                  "boundary": {
                    "validation_only": true,
                    "tool_execution": false,
                    "retrieval_execution": false
                  }
                }
                """.formatted(
                toolWarningCount > 0 ? "responses" : "chat",
                toolWarningCount > 0 ? 0 : 2,
                toolWarningCount > 0 ? 1 : 0,
                toolWarningCount);
    }

    private static String embeddingValidationResponse() {
        return """
                {
                  "object": "gollek.agent_validation",
                  "surface": "embeddings",
                  "valid": true,
                  "model_invoked": false,
                  "trace": {
                    "request_id": "req_embed_validate",
                    "trace_id": "trace_embed_validate"
                  },
                  "normalized": {
                    "request_id": "req_embed_validate",
                    "trace_id": "trace_embed_validate",
                    "model": "demo-embed",
                    "input_count": 1,
                    "input_lengths": [19],
                    "requested_dimensions": 384,
                    "encoding_format": "float",
                    "parameter_keys": ["dimensions", "encoding_format", "metadata"],
                    "metadata": {"tenant": "agent-project"},
                    "rag": {
                      "embedding_generation": true,
                      "retrieval_execution": false,
                      "retrieval_policy_owned_by": "agent_orchestrator",
                      "vector_store_owned_by": "agent_orchestrator"
                    }
                  },
                  "boundary": {
                    "validation_only": true,
                    "tool_execution": false,
                    "retrieval_execution": false
                  }
                }
                """;
    }

    private static String mcpToolsResponse(boolean emptyTools) {
        String tools = emptyTools
                ? "[]"
                : """
                  [
                    {
                      "type": "function",
                      "function": {
                        "name": "knowledge_search",
                        "description": "Search caller-owned knowledge",
                        "parameters": {
                          "type": "object",
                          "properties": {
                            "query": {"type": "string"}
                          }
                        }
                      },
                      "x_gollek": {
                        "mcp_server": "knowledge",
                        "mcp_tool_name": "search"
                      }
                    }
                  ]
                  """;
        return """
                {
                  "available": true,
                  "registry_path": "/tmp/gollek/mcp/servers.json",
                  "compat": "openai",
                  "enabled_only": true,
                  "tools": %s,
                  "boundary": {
                    "role": "discovery_only",
                    "tool_execution": false,
                    "gollek_exposes": ["registered_servers", "tool_schemas"],
                    "agent_orchestrator_owns": ["tool_authorization", "tool_execution"]
                  }
                }
                """.formatted(tools);
    }

	    private static String capabilitiesResponse() {
	        return """
	                {
	                  "object": "gollek.agent_capabilities",
	                  "version": "v1",
	                  "contract_version": "v1",
	                  "supported_contract_versions": ["v1"],
	                  "service_role": "inference_serving_engine",
	                  "compatibility": [
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
	                  "feature_negotiation": {
	                    "mode": "feature_flags",
	                    "feature_namespace": "gollek.agent.compatibility",
	                    "contract_version": "v1",
	                    "minimum_client_contract_version": "v1",
	                    "supported_contract_versions": ["v1"],
	                    "default_feature_profile": "agent_serving",
	                    "supported_feature_profiles": [
	                      "agent_serving",
	                      "chat_agent",
	                      "responses_agent",
	                      "embedding_rag",
	                      "mcp_tools"
	                    ],
	                    "required_features": [
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
	                    "optional_features": [
	                      "system_prompt",
	                      "tool_choice",
	                      "tool_calls",
	                      "mcp_registry",
	                      "mcp_tool_definitions",
	                      "embeddings"
	                    ]
	                  },
	                  "endpoints": {
	                    "openai_chat_completions": "/v1/chat/completions",
	                    "openai_responses": "/v1/responses",
                    "openai_embeddings": "/v1/embeddings",
                    "model_capabilities": "/v1/models/{id}/capabilities",
                    "agent_contract": "/v1/agent/contract",
                    "agent_readiness_issues": "/v1/agent/readiness/issues",
                    "agent_preflight": "/v1/agent/preflight",
                    "agent_validation": "/v1/agent/validate",
                    "agent_tool_validation": "/v1/agent/tools/validate",
                    "mcp_tools": "/v1/mcp/tools"
                  },
                  "agent_boundary": {
                    "gollek_owns": [
                      "model serving",
                      "provider routing",
                      "system prompts",
                      "tool-call request and response schema",
                      "embedding generation",
                      "RAG context injection",
                      "MCP server registry and tool definitions"
                    ],
                    "agent_orchestrator_owns": [
                      "planning",
                      "memory policy",
                      "tool authorization",
                      "tool execution loops",
                      "workflow state"
                    ]
                  },
                  "auth": {
                    "x_api_key_header": "X-API-Key",
                    "authorization_header": "Bearer token"
                  },
                  "traceability": {
                    "request_id_header": "X-Gollek-Request-Id",
                    "trace_id_header": "X-Gollek-Trace-Id",
                    "session_id_header": "X-Gollek-Session-Id",
                    "user_id_header": "X-Gollek-User-Id",
                    "metadata_key": "gollek_trace"
                  }
                }
                """;
    }

    private static String issueCatalogResponse() {
        return """
                {
                  "object": "gollek.agent_readiness_issue_catalog",
                  "version": "v1",
                  "service_role": "inference_serving_engine",
                  "boundary": {
                    "validation_only": true,
                    "model_invoked": false,
                    "tool_execution": false,
                    "retrieval_execution": false,
                    "tool_authorization": false
                  },
                  "count": 2,
                  "items": [
                    {
                      "code": "TOOL_DEFINITIONS_INVALID",
                      "area": "tool_validation",
                      "default_severity": "blocking",
                      "summary": "Tool definitions are invalid.",
                      "remediation": "Fix the OpenAI-compatible tool schema before serving the route."
                    },
                    {
                      "code": "MCP_DISCOVERY_UNAVAILABLE",
                      "area": "mcp_discovery",
                      "default_severity": "warning",
                      "summary": "MCP discovery did not return usable tools.",
                      "remediation": "Check the MCP registry or mark discovery optional for this route."
                    }
                  ],
                  "by_code": {
                    "TOOL_DEFINITIONS_INVALID": {
                      "code": "TOOL_DEFINITIONS_INVALID",
                      "area": "tool_validation",
                      "default_severity": "blocking",
                      "summary": "Tool definitions are invalid.",
                      "remediation": "Fix the OpenAI-compatible tool schema before serving the route."
                    },
                    "MCP_DISCOVERY_UNAVAILABLE": {
                      "code": "MCP_DISCOVERY_UNAVAILABLE",
                      "area": "mcp_discovery",
                      "default_severity": "warning",
                      "summary": "MCP discovery did not return usable tools.",
                      "remediation": "Check the MCP registry or mark discovery optional for this route."
                    }
                  }
                }
                """;
    }

    private static String invalidContractResponse() {
        return """
                {
                  "object": "gollek.agent_contract",
                  "version": "v1",
                  "service_role": "agent_runtime",
                  "boundary": {
                    "gollek_owns": ["model_serving"],
                    "agent_orchestrator_owns": ["planning"],
                    "tool_execution": true,
                    "retrieval_execution": true
                  },
                  "endpoints": {},
                  "streaming": {}
                }
                """;
    }

    private static String validContractResponse() {
        return """
                {
                  "object": "gollek.agent_contract",
                  "version": "v1",
                  "service_role": "inference_serving_engine",
                  "boundary": {
                    "gollek_owns": [
                      "model_serving",
                      "provider_routing",
                      "system_prompt_mapping",
                      "tool_schema_ingestion",
                      "tool_contract_validation",
                      "mcp_registry_discovery",
                      "embedding_generation",
                      "rag_context_injection"
                    ],
                    "agent_orchestrator_owns": [
                      "planning",
                      "memory_policy",
                      "retrieval_policy",
                      "vector_store_ownership",
                      "tool_authorization",
                      "tool_execution",
                      "tool_result_loop",
                      "workflow_state"
                    ],
                    "tool_execution": false,
                    "retrieval_execution": false
                  },
                  "streaming": {
                    "done_sentinel": "[DONE]",
                    "chat_completions_events": ["chat.completion.chunk", "[DONE]"],
                    "responses_events": ["response.output_text.delta", "response.completed", "[DONE]"]
                  },
                  "endpoints": {
                    "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                    "responses": {"method": "POST", "path": "/v1/responses"},
                    "embeddings": {"method": "POST", "path": "/v1/embeddings"},
                    "model_capabilities": {"method": "GET", "path": "/v1/models/{id}/capabilities"},
                    "mcp_tools": {"method": "GET", "path": "/v1/mcp/tools?compat=openai"},
                    "agent_capabilities": {"method": "GET", "path": "/v1/agent/capabilities"},
                    "agent_contract": {"method": "GET", "path": "/v1/agent/contract"},
                    "agent_readiness_issues": {"method": "GET", "path": "/v1/agent/readiness/issues"},
                    "agent_preflight": {"method": "POST", "path": "/v1/agent/preflight"},
                    "agent_validation": {"method": "POST", "path": "/v1/agent/validate?surface={surface}"},
                    "agent_tool_validation": {"method": "POST", "path": "/v1/agent/tools/validate"}
                  }
                }
                """;
    }

    private static String modelCapabilitiesResponse() {
        return """
                {
                  "model_id": "demo-model",
                  "known": true,
                  "available": true,
                  "preferred_provider": "fixture-provider",
                  "format": "SAFETENSORS",
                  "architecture": "fixture-transformer",
                  "limits": {
                    "context_tokens": 8192,
                    "input_tokens": 4096,
                    "output_tokens": 1024,
                    "embedding_dimensions": 768
                  },
                  "api_contract": {
                    "chat_completions": true,
                    "chat_streaming": true,
                    "responses": true,
                    "responses_streaming": true,
                    "system_prompt": true,
                    "tools_request_schema": true,
                    "mcp_tool_definitions": true,
                    "rag_context_injection": true
                  },
                  "openai_compatibility": {
                    "chat_completions": true,
                    "chat_streaming": true,
                    "responses": true,
                    "responses_streaming": true,
                    "embeddings": true
                  },
                  "inference": {
                    "completion": true,
                    "streaming": true,
                    "system_prompt": true,
                    "json_mode": true,
                    "structured_outputs": true
                  },
                  "tooling": {
                    "tool_definitions": true,
                    "tool_choice": true,
                    "model_tool_calling": true,
                    "mcp_tool_definitions": true,
                    "tool_execution": false
                  },
                  "rag": {
                    "context_injection": true,
                    "retrieval_policy": false,
                    "vector_store_ownership": false
                  },
                  "embeddings": {
                    "generation": true,
                    "endpoint": "/v1/embeddings",
                    "openai_compatible": true,
                    "dimensions": 768,
                    "encoding_formats": ["float", "base64"],
                    "input_aliases": ["input", "inputs"],
                    "batch_inputs": true,
                    "metadata_passthrough": true,
                    "retrieval_policy": false,
                    "vector_store_ownership": false
                  },
                  "modalities": {
                    "text": true,
                    "embeddings": true
                  },
                  "provider_candidates": [
                    {
                      "id": "fixture-provider",
                      "name": "Fixture Provider",
                      "health": "HEALTHY",
                      "supports_model": true,
                      "capabilities": {
                        "streaming": true,
                        "tool_calling": true
                      }
                    }
                  ],
                  "metadata": {}
                }
                """;
    }

    private record Captured(int exitCode, String out, String err) {
    }
}
