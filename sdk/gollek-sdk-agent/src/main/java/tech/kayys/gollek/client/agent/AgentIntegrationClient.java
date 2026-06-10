package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Agent-facing discovery and validation client for a Gollek serving endpoint.
 *
 * <p>This client intentionally exposes contracts, validation, and MCP tool
 * schemas only. Planning, authorization, tool execution, and workflow state stay
 * with the caller's agent runtime.
 */
public final class AgentIntegrationClient {
    private static final String HEADER_API_KEY = "X-API-Key";
    private static final String HEADER_AUTHORIZATION = "Authorization";
    private static final String AUTHORIZATION_SCHEME = "Bearer";
    private static final String COMMUNITY_API_KEY = "community";
    private static final TypeReference<LinkedHashMap<String, Object>> MAP_TYPE = new TypeReference<>() {
    };

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;

    public AgentIntegrationClient(HttpClient httpClient, ObjectMapper objectMapper, String baseUrl, String apiKey) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper != null ? objectMapper : new ObjectMapper();
        this.baseUrl = normalizeBaseUrl(baseUrl);
        this.apiKey = normalizeApiKey(apiKey);
    }

    /**
     * Returns Gollek's agent-facing serving capabilities and endpoint links.
     */
    public Map<String, Object> capabilities() throws AgentIntegrationException {
        return capabilities(AgentRequestOptions.empty());
    }

    /**
     * Returns Gollek's agent-facing serving capabilities and endpoint links.
     */
    public Map<String, Object> capabilities(AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/agent/capabilities", options);
    }

    /**
     * Returns Gollek's agent-facing serving capabilities as a typed discovery view.
     */
    public AgentCapabilitiesView capabilitiesView() throws AgentIntegrationException {
        return capabilitiesView(AgentRequestOptions.empty());
    }

    /**
     * Returns Gollek's agent-facing serving capabilities as a typed discovery view.
     */
    public AgentCapabilitiesView capabilitiesView(AgentRequestOptions options) throws AgentIntegrationException {
        return AgentCapabilitiesView.from(capabilities(options), objectMapper);
    }

    /**
     * Returns the machine-readable agent integration contract.
     */
    public Map<String, Object> contract() throws AgentIntegrationException {
        return contract(AgentRequestOptions.empty());
    }

    /**
     * Returns the machine-readable agent integration contract.
     */
    public Map<String, Object> contract(AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/agent/contract", options);
    }

    /**
     * Returns the machine-readable agent integration contract as a typed serving view.
     */
    public AgentServingContract contractView() throws AgentIntegrationException {
        return contractView(AgentRequestOptions.empty());
    }

    /**
     * Returns the machine-readable agent integration contract as a typed serving view.
     */
    public AgentServingContract contractView(AgentRequestOptions options) throws AgentIntegrationException {
        return AgentServingContract.from(contract(options), objectMapper);
    }

    /**
     * Returns stable readiness issue codes with summaries and remediation text.
     */
    public Map<String, Object> readinessIssueCatalog() throws AgentIntegrationException {
        return readinessIssueCatalog(AgentRequestOptions.empty());
    }

    /**
     * Returns stable readiness issue codes with summaries and remediation text.
     */
    public Map<String, Object> readinessIssueCatalog(AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/agent/readiness/issues", options);
    }

    /**
     * Returns stable readiness issue codes as a typed catalog view.
     */
    public AgentReadinessIssueCatalogView readinessIssueCatalogView() throws AgentIntegrationException {
        return readinessIssueCatalogView(AgentRequestOptions.empty());
    }

    /**
     * Returns stable readiness issue codes as a typed catalog view.
     */
    public AgentReadinessIssueCatalogView readinessIssueCatalogView(AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentReadinessIssueCatalogView.from(readinessIssueCatalog(options), objectMapper);
    }

    /**
     * Returns one model's agent-facing capability matrix.
     */
    public Map<String, Object> modelCapabilities(String modelId) throws AgentIntegrationException {
        return modelCapabilities(modelId, AgentRequestOptions.empty());
    }

    /**
     * Returns one model's agent-facing capability matrix.
     */
    public Map<String, Object> modelCapabilities(String modelId, AgentRequestOptions options)
            throws AgentIntegrationException {
        requireName(modelId, "modelId");
        return get("/v1/models/" + encode(modelId) + "/capabilities", options);
    }

    /**
     * Returns one model's agent-facing capability matrix as a typed route view.
     */
    public AgentModelCapabilitiesView modelCapabilitiesView(String modelId) throws AgentIntegrationException {
        return modelCapabilitiesView(modelId, AgentRequestOptions.empty());
    }

    /**
     * Returns one model's agent-facing capability matrix as a typed route view.
     */
    public AgentModelCapabilitiesView modelCapabilitiesView(String modelId, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentModelCapabilitiesView.from(modelCapabilities(modelId, options), objectMapper);
    }

    /**
     * Runs a serving-only agent preflight and returns a grouped readiness report.
     *
     * <p>The preflight fetches discovery views, validates MCP/tool schemas, and
     * validates the request dry-run. It does not invoke a model, execute tools,
     * run retrieval, update memory, or choose the next agent step.
     */
    public AgentServingReadinessReport servingReadiness(
            String modelId,
            String surface,
            Map<String, Object> request) throws AgentIntegrationException {
        return servingReadiness(modelId, surface, request, AgentRequestOptions.empty());
    }

    /**
     * Runs a serving-only agent preflight and returns a grouped readiness report.
     *
     * <p>The preflight fetches discovery views, validates MCP/tool schemas, and
     * validates the request dry-run. It does not invoke a model, execute tools,
     * run retrieval, update memory, or choose the next agent step.
     */
    public AgentServingReadinessReport servingReadiness(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return servingReadiness(AgentServingPreflightRequest.builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .requestOptions(options)
                .build());
    }

    /**
     * Runs a serving-only agent preflight from a typed preflight request.
     *
     * <p>This overload keeps future preflight options explicit while preserving
     * the serving boundary: it never invokes a model, executes tools, runs
     * retrieval, updates memory, or chooses the next agent step.
     */
    public AgentServingReadinessReport servingReadiness(AgentServingPreflightRequest preflight)
            throws AgentIntegrationException {
        if (preflight == null) {
            throw new IllegalArgumentException("preflight must not be null");
        }
        AgentRequestOptions effectiveOptions = preflight.requestOptions();
        AgentCapabilitiesView capabilities = capabilitiesView(effectiveOptions);
        AgentServingContract contract = contractView(effectiveOptions);
        AgentModelCapabilitiesView modelRoute = modelCapabilitiesView(preflight.modelId(), effectiveOptions);
        AgentMcpDiscoveryView mcpTools = preflight.discoverMcpTools()
                ? mcpToolsView(preflight.openAiToolCompatibility(), preflight.enabledOnly(), effectiveOptions)
                : null;
        AgentServingReadinessReport.Builder report = AgentServingReadinessReport.builder()
                .capabilities(capabilities)
                .contract(contract)
                .featureNegotiation(
                        capabilities.featureNegotiation(),
                        preflight.featureProfile(),
                        preflight.requiredContractVersion(),
                        preflight.requiredFeatures(),
                        preflight.optionalFeatures())
                .modelRoute(preflight.surface(), modelRoute)
                .mcpDiscovery(mcpTools, preflight.mcpDiscoveryRequired());
        if (preflight.validateTools()) {
            AgentToolValidationView toolValidation = validateToolsView(
                    toolValidationRequest(preflight.request(), mcpTools),
                    effectiveOptions);
            report.toolValidation(toolValidation, preflight.toolValidationRequired());
        }
        if (preflight.validateRequest()) {
            AgentValidationView requestValidation = validateRequestView(
                    preflight.surface(),
                    preflight.request(),
                    effectiveOptions);
            report.requestValidation(requestValidation, preflight.requestValidationRequired());
        }
        return report.build();
    }

    /**
     * Calls Gollek's server-side agent preflight endpoint and returns the raw
     * readiness payload.
     *
     * <p>This is the one-call variant of {@link #servingReadiness(AgentServingPreflightRequest)}.
     * It keeps the same serving boundary and relies on the server to compose
     * capability, contract, model route, MCP, tool, and request-validation
     * checks.
     */
    public Map<String, Object> servingPreflight(
            String modelId,
            String surface,
            Map<String, Object> request) throws AgentIntegrationException {
        return servingPreflight(modelId, surface, request, AgentRequestOptions.empty());
    }

    /**
     * Calls Gollek's server-side agent preflight endpoint and returns the raw
     * readiness payload.
     */
    public Map<String, Object> servingPreflight(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return servingPreflight(AgentServingPreflightRequest.builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .requestOptions(options)
                .build());
    }

    /**
     * Calls Gollek's server-side agent preflight endpoint and returns the raw
     * readiness payload.
     */
    public Map<String, Object> servingPreflight(AgentServingPreflightRequest preflight)
            throws AgentIntegrationException {
        if (preflight == null) {
            throw new IllegalArgumentException("preflight must not be null");
        }
        return post("/v1/agent/preflight", preflightBody(preflight), preflight.requestOptions());
    }

    /**
     * Calls Gollek's server-side preflight endpoint and maps the response to the
     * same typed readiness report used by the multi-call preflight flow.
     */
    public AgentServingReadinessReport servingPreflightReadiness(
            String modelId,
            String surface,
            Map<String, Object> request) throws AgentIntegrationException {
        return servingPreflightReadiness(modelId, surface, request, AgentRequestOptions.empty());
    }

    /**
     * Calls Gollek's server-side preflight endpoint and maps the response to the
     * same typed readiness report used by the multi-call preflight flow.
     */
    public AgentServingReadinessReport servingPreflightReadiness(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return servingPreflightReadiness(AgentServingPreflightRequest.builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .requestOptions(options)
                .build());
    }

    /**
     * Calls Gollek's server-side preflight endpoint and maps the response to the
     * same typed readiness report used by the multi-call preflight flow.
     */
    public AgentServingReadinessReport servingPreflightReadiness(AgentServingPreflightRequest preflight)
            throws AgentIntegrationException {
        return AgentServingReadinessReport.fromPreflightResponse(servingPreflight(preflight));
    }

    /**
     * Calls Gollek's server-side preflight endpoint and returns the request,
     * raw payload, typed readiness report, and route comparison together.
     */
    public AgentServingPreflightResult servingPreflightResult(
            String modelId,
            String surface,
            Map<String, Object> request) throws AgentIntegrationException {
        return servingPreflightResult(modelId, surface, request, AgentRequestOptions.empty());
    }

    /**
     * Calls Gollek's server-side preflight endpoint and returns the request,
     * raw payload, typed readiness report, and route comparison together.
     */
    public AgentServingPreflightResult servingPreflightResult(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return servingPreflightResult(AgentServingPreflightRequest.builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .requestOptions(options)
                .build());
    }

    /**
     * Calls Gollek's server-side preflight endpoint and returns the request,
     * raw payload, typed readiness report, and route comparison together.
     */
    public AgentServingPreflightResult servingPreflightResult(AgentServingPreflightRequest preflight)
            throws AgentIntegrationException {
        return AgentServingPreflightResult.from(preflight, servingPreflight(preflight));
    }

    /**
     * Calls Gollek's server-side preflight endpoint and evaluates it against
     * the default readiness-only gate policy.
     */
    public AgentServingPreflightGate servingPreflightGate(
            String modelId,
            String surface,
            Map<String, Object> request) throws AgentIntegrationException {
        return servingPreflightGate(modelId, surface, request, AgentRequestOptions.empty(), null);
    }

    /**
     * Calls Gollek's server-side preflight endpoint and evaluates it against
     * caller-owned gate policy.
     */
    public AgentServingPreflightGate servingPreflightGate(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentServingPreflightPolicy policy) throws AgentIntegrationException {
        return servingPreflightGate(modelId, surface, request, AgentRequestOptions.empty(), policy);
    }

    /**
     * Calls Gollek's server-side preflight endpoint and evaluates it against
     * caller-owned gate policy.
     */
    public AgentServingPreflightGate servingPreflightGate(
            String modelId,
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options,
            AgentServingPreflightPolicy policy) throws AgentIntegrationException {
        return servingPreflightGate(AgentServingPreflightRequest.builder()
                .modelId(modelId)
                .surface(surface)
                .request(request)
                .requestOptions(options)
                .build(), policy);
    }

    /**
     * Calls Gollek's server-side preflight endpoint and evaluates it against
     * the default readiness-only gate policy.
     */
    public AgentServingPreflightGate servingPreflightGate(AgentServingPreflightRequest preflight)
            throws AgentIntegrationException {
        return servingPreflightGate(preflight, null);
    }

    /**
     * Calls Gollek's server-side preflight endpoint and evaluates it against
     * caller-owned gate policy.
     */
    public AgentServingPreflightGate servingPreflightGate(
            AgentServingPreflightRequest preflight,
            AgentServingPreflightPolicy policy) throws AgentIntegrationException {
        return servingPreflightResult(preflight).gate(policy);
    }

    /**
     * Validates a chat, Responses, or embedding payload without invoking a model.
     */
    public Map<String, Object> validateRequest(Map<String, Object> request) throws AgentIntegrationException {
        return validateRequest(null, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a chat, Responses, or embedding payload without invoking a model.
     */
    public Map<String, Object> validateRequest(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return validateRequest(null, request, options);
    }

    /**
     * Validates a payload for a specific compatibility surface without invoking a model.
     */
    public Map<String, Object> validateRequest(String surface, Map<String, Object> request) throws AgentIntegrationException {
        return validateRequest(surface, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a payload for a specific compatibility surface without invoking a model.
     */
    public Map<String, Object> validateRequest(
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        String path = "/v1/agent/validate";
        if (surface != null && !surface.isBlank()) {
            path += "?surface=" + encode(surface);
        }
        return post(path, body(request), options);
    }

    /**
     * Validates a chat, Responses, or embedding payload and returns a typed dry-run view.
     */
    public AgentValidationView validateRequestView(Map<String, Object> request) throws AgentIntegrationException {
        return validateRequestView(null, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a chat, Responses, or embedding payload and returns a typed dry-run view.
     */
    public AgentValidationView validateRequestView(
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return validateRequestView(null, request, options);
    }

    /**
     * Validates a payload for a specific compatibility surface and returns a typed dry-run view.
     */
    public AgentValidationView validateRequestView(String surface, Map<String, Object> request)
            throws AgentIntegrationException {
        return validateRequestView(surface, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a payload for a specific compatibility surface and returns a typed dry-run view.
     */
    public AgentValidationView validateRequestView(
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws AgentIntegrationException {
        return AgentValidationView.from(validateRequest(surface, request, options), objectMapper);
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions without executing tools.
     */
    public Map<String, Object> validateTools(Map<String, Object> request) throws AgentIntegrationException {
        return validateTools(request, AgentRequestOptions.empty());
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions without executing tools.
     */
    public Map<String, Object> validateTools(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return post("/v1/agent/tools/validate", body(request), options);
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions and returns a typed view.
     */
    public AgentToolValidationView validateToolsView(Map<String, Object> request) throws AgentIntegrationException {
        return validateToolsView(request, AgentRequestOptions.empty());
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions and returns a typed view.
     */
    public AgentToolValidationView validateToolsView(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentToolValidationView.from(validateTools(request, options), objectMapper);
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request.
     */
    public Map<String, Object> createChatCompletion(Map<String, Object> request) throws AgentIntegrationException {
        return createChatCompletion(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request.
     */
    public Map<String, Object> createChatCompletion(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return post("/v1/chat/completions", nonStreamingBody(request), options);
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request and returns a typed response view.
     */
    public AgentResponseView createChatCompletionView(Map<String, Object> request) throws AgentIntegrationException {
        return createChatCompletionView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request and returns a typed response view.
     */
    public AgentResponseView createChatCompletionView(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentResponseView.from(createChatCompletion(request, options), objectMapper);
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request.
     */
    public Map<String, Object> createResponse(Map<String, Object> request) throws AgentIntegrationException {
        return createResponse(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request.
     */
    public Map<String, Object> createResponse(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return post("/v1/responses", nonStreamingBody(request), options);
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request and returns a typed response view.
     */
    public AgentResponseView createResponseView(Map<String, Object> request) throws AgentIntegrationException {
        return createResponseView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request and returns a typed response view.
     */
    public AgentResponseView createResponseView(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentResponseView.from(createResponse(request, options), objectMapper);
    }

    /**
     * Streams an OpenAI-compatible chat completion request, emits each parsed
     * event to the callback, and returns the final accumulated stream state.
     */
    public AgentStreamAccumulator.Snapshot streamChatCompletion(
            Map<String, Object> request,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        return streamChatCompletion(request, AgentRequestOptions.empty(), eventConsumer);
    }

    /**
     * Streams an OpenAI-compatible chat completion request, emits each parsed
     * event to the callback, and returns the final accumulated stream state.
     */
    public AgentStreamAccumulator.Snapshot streamChatCompletion(
            Map<String, Object> request,
            AgentRequestOptions options,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        return stream("/v1/chat/completions", streamingBody(request), options, eventConsumer);
    }

    /**
     * Streams an OpenAI-compatible Responses request, emits each parsed event to
     * the callback, and returns the final accumulated stream state.
     */
    public AgentStreamAccumulator.Snapshot streamResponse(
            Map<String, Object> request,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        return streamResponse(request, AgentRequestOptions.empty(), eventConsumer);
    }

    /**
     * Streams an OpenAI-compatible Responses request, emits each parsed event to
     * the callback, and returns the final accumulated stream state.
     */
    public AgentStreamAccumulator.Snapshot streamResponse(
            Map<String, Object> request,
            AgentRequestOptions options,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        return stream("/v1/responses", streamingBody(request), options, eventConsumer);
    }

    /**
     * Sends an OpenAI-compatible embedding request.
     */
    public Map<String, Object> createEmbedding(Map<String, Object> request) throws AgentIntegrationException {
        return createEmbedding(request, AgentRequestOptions.empty());
    }

    /**
     * Sends an OpenAI-compatible embedding request.
     */
    public Map<String, Object> createEmbedding(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return post("/v1/embeddings", body(request), options);
    }

    /**
     * Sends an OpenAI-compatible embedding request and returns a typed embedding view.
     */
    public AgentEmbeddingView createEmbeddingView(Map<String, Object> request) throws AgentIntegrationException {
        return createEmbeddingView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends an OpenAI-compatible embedding request and returns a typed embedding view.
     */
    public AgentEmbeddingView createEmbeddingView(Map<String, Object> request, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentEmbeddingView.from(createEmbedding(request, options), objectMapper);
    }

    /**
     * Lists registered MCP servers, if the server runtime exposes a registry.
     */
    public Map<String, Object> mcpServers() throws AgentIntegrationException {
        return mcpServers(AgentRequestOptions.empty());
    }

    /**
     * Lists registered MCP servers, if the server runtime exposes a registry.
     */
    public Map<String, Object> mcpServers(AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/mcp/servers", options);
    }

    /**
     * Lists registered MCP servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServersView() throws AgentIntegrationException {
        return mcpServersView(AgentRequestOptions.empty());
    }

    /**
     * Lists registered MCP servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServersView(AgentRequestOptions options) throws AgentIntegrationException {
        return AgentMcpDiscoveryView.from(mcpServers(options), objectMapper);
    }

    /**
     * Returns one redacted MCP server registration.
     */
    public Map<String, Object> mcpServer(String serverName) throws AgentIntegrationException {
        return mcpServer(serverName, AgentRequestOptions.empty());
    }

    /**
     * Returns one redacted MCP server registration.
     */
    public Map<String, Object> mcpServer(String serverName, AgentRequestOptions options) throws AgentIntegrationException {
        requireName(serverName, "serverName");
        return get("/v1/mcp/servers/" + encode(serverName), options);
    }

    /**
     * Returns one redacted MCP server registration as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServerView(String serverName) throws AgentIntegrationException {
        return mcpServerView(serverName, AgentRequestOptions.empty());
    }

    /**
     * Returns one redacted MCP server registration as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServerView(String serverName, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentMcpDiscoveryView.from(mcpServer(serverName, options), objectMapper);
    }

    /**
     * Lists MCP tools across enabled servers using the server default representation.
     */
    public Map<String, Object> mcpTools() throws AgentIntegrationException {
        return mcpTools(AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across enabled servers using the server default representation.
     */
    public Map<String, Object> mcpTools(AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/mcp/tools", options);
    }

    /**
     * Lists MCP tools across enabled servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpToolsView() throws AgentIntegrationException {
        return mcpToolsView(AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across enabled servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpToolsView(AgentRequestOptions options) throws AgentIntegrationException {
        return AgentMcpDiscoveryView.from(mcpTools(options), objectMapper);
    }

    /**
     * Lists MCP tools across servers, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpTools(boolean openAiCompat, boolean enabledOnly) throws AgentIntegrationException {
        return mcpTools(openAiCompat, enabledOnly, AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across servers, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpTools(
            boolean openAiCompat,
            boolean enabledOnly,
            AgentRequestOptions options) throws AgentIntegrationException {
        return get("/v1/mcp/tools?compat=" + compat(openAiCompat) + "&enabledOnly=" + enabledOnly, options);
    }

    /**
     * Lists MCP tools across servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpToolsView(boolean openAiCompat, boolean enabledOnly)
            throws AgentIntegrationException {
        return mcpToolsView(openAiCompat, enabledOnly, AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across servers as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpToolsView(boolean openAiCompat, boolean enabledOnly, AgentRequestOptions options)
            throws AgentIntegrationException {
        return AgentMcpDiscoveryView.from(mcpTools(openAiCompat, enabledOnly, options), objectMapper);
    }

    /**
     * Lists tools for one MCP server, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpServerTools(String serverName, boolean openAiCompat) throws AgentIntegrationException {
        return mcpServerTools(serverName, openAiCompat, AgentRequestOptions.empty());
    }

    /**
     * Lists tools for one MCP server, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpServerTools(
            String serverName,
            boolean openAiCompat,
            AgentRequestOptions options) throws AgentIntegrationException {
        requireName(serverName, "serverName");
        return get("/v1/mcp/servers/" + encode(serverName) + "/tools?compat=" + compat(openAiCompat), options);
    }

    /**
     * Lists tools for one MCP server as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServerToolsView(String serverName, boolean openAiCompat)
            throws AgentIntegrationException {
        return mcpServerToolsView(serverName, openAiCompat, AgentRequestOptions.empty());
    }

    /**
     * Lists tools for one MCP server as a typed discovery-only view.
     */
    public AgentMcpDiscoveryView mcpServerToolsView(
            String serverName,
            boolean openAiCompat,
            AgentRequestOptions options) throws AgentIntegrationException {
        return AgentMcpDiscoveryView.from(mcpServerTools(serverName, openAiCompat, options), objectMapper);
    }

    private Map<String, Object> get(String path, AgentRequestOptions options) throws AgentIntegrationException {
        HttpRequest request = request(path, options)
                .GET()
                .build();
        return send(request);
    }

    private Map<String, Object> post(String path, Map<String, Object> body, AgentRequestOptions options)
            throws AgentIntegrationException {
        try {
            HttpRequest request = request(path, options)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                    .build();
            return send(request);
        } catch (IOException e) {
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Failed to encode agent integration request", e);
        }
    }

    private AgentStreamAccumulator.Snapshot stream(
            String path,
            Map<String, Object> body,
            AgentRequestOptions options,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        try {
            HttpRequest request = request(path, options)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                    .build();
            HttpResponse<Stream<String>> response = httpClient.send(request, HttpResponse.BodyHandlers.ofLines());
            return handleStreamResponse(response, eventConsumer);
        } catch (IOException e) {
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Agent integration stream failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Agent integration stream interrupted", e);
        }
    }

    private HttpRequest.Builder request(String path, AgentRequestOptions options) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("Accept", "application/json")
                .header(HEADER_API_KEY, apiKey)
                .header(HEADER_AUTHORIZATION, authorizationValue(apiKey))
                .timeout(Duration.ofSeconds(30));
        applyHeaders(builder, options);
        return builder;
    }

    private void applyHeaders(HttpRequest.Builder builder, AgentRequestOptions options) {
        AgentRequestOptions effective = options == null ? AgentRequestOptions.empty() : options;
        effective.toHeaders().forEach(builder::setHeader);
    }

    private Map<String, Object> send(HttpRequest request) throws AgentIntegrationException {
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return handleResponse(response);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Agent integration request interrupted", e);
        } catch (IOException e) {
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Agent integration request failed", e);
        }
    }

    private Map<String, Object> handleResponse(HttpResponse<String> response) throws AgentIntegrationException {
        String responseBody = response.body();
        if (response.statusCode() >= 200 && response.statusCode() < 300) {
            if (responseBody == null || responseBody.isBlank()) {
                return new LinkedHashMap<>();
            }
            try {
                return objectMapper.readValue(responseBody, MAP_TYPE);
            } catch (IOException e) {
                throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Failed to parse agent integration response", e);
            }
        }

        throw new AgentIntegrationException(
                "SDK_ERR_AGENT_INTEGRATION",
                "Agent integration request failed with status " + response.statusCode() + ": "
                        + (responseBody == null ? "" : responseBody));
    }

    private AgentStreamAccumulator.Snapshot handleStreamResponse(
            HttpResponse<Stream<String>> response,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        Stream<String> body = response.body();
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            String responseBody;
            if (body == null) {
                responseBody = "";
            } else {
                try (body) {
                    responseBody = body.collect(Collectors.joining("\n"));
                }
            }
            throw new AgentIntegrationException(
                    "SDK_ERR_AGENT_INTEGRATION",
                    "Agent integration stream failed with status " + response.statusCode() + ": " + responseBody);
        }
        if (body == null) {
            return new AgentStreamAccumulator().snapshot();
        }
        return consumeStream(body, eventConsumer);
    }

    private AgentStreamAccumulator.Snapshot consumeStream(
            Stream<String> lines,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        AgentStreamEventParser parser = new AgentStreamEventParser(objectMapper);
        AgentStreamAccumulator accumulator = new AgentStreamAccumulator();
        StringBuilder eventData = new StringBuilder();
        try (lines) {
            Iterator<String> iterator = lines.iterator();
            while (iterator.hasNext()) {
                String line = iterator.next();
                if (line == null || line.isBlank()) {
                    flushStreamEvent(eventData, parser, accumulator, eventConsumer);
                    continue;
                }
                String trimmed = line.trim();
                if (trimmed.startsWith("data:")) {
                    appendEventData(eventData, trimmed.substring("data:".length()).trim());
                } else if (looksLikeStandaloneData(trimmed)) {
                    appendEventData(eventData, trimmed);
                    flushStreamEvent(eventData, parser, accumulator, eventConsumer);
                }
            }
            flushStreamEvent(eventData, parser, accumulator, eventConsumer);
            return accumulator.snapshot();
        }
    }

    private void appendEventData(StringBuilder eventData, String data) {
        if (eventData.length() > 0) {
            eventData.append('\n');
        }
        eventData.append(data);
    }

    private void flushStreamEvent(
            StringBuilder eventData,
            AgentStreamEventParser parser,
            AgentStreamAccumulator accumulator,
            Consumer<AgentStreamEvent> eventConsumer) throws AgentIntegrationException {
        if (eventData.isEmpty()) {
            return;
        }
        String data = eventData.toString();
        eventData.setLength(0);
        try {
            AgentStreamEvent event = parser.parse(data);
            accumulator.accept(event);
            if (eventConsumer != null) {
                eventConsumer.accept(event);
            }
        } catch (JsonProcessingException e) {
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Failed to parse agent integration stream", e);
        } catch (RuntimeException e) {
            throw new AgentIntegrationException("SDK_ERR_AGENT_INTEGRATION", "Agent integration stream consumer failed", e);
        }
    }

    private Map<String, Object> body(Map<String, Object> request) {
        return request == null ? Map.of() : request;
    }

    private Map<String, Object> nonStreamingBody(Map<String, Object> request) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (request != null) {
            out.putAll(request);
        }
        out.put("stream", false);
        return out;
    }

    private Map<String, Object> streamingBody(Map<String, Object> request) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (request != null) {
            out.putAll(request);
        }
        out.put("stream", true);
        return out;
    }

    private Map<String, Object> toolValidationRequest(Map<String, Object> request, AgentMcpDiscoveryView mcpTools) {
        Object requestTools = request == null ? null : request.get("tools");
        if (requestTools != null) {
            return Map.of("tools", requestTools);
        }
        List<Map<String, Object>> discoveredTools = mcpTools == null ? List.of() : mcpTools.openAiToolDefinitions();
        return Map.of("tools", discoveredTools);
    }

    private Map<String, Object> preflightBody(AgentServingPreflightRequest preflight) {
        Map<String, Object> out = new LinkedHashMap<>();
        putIfPresent(out, "model", preflight.modelId());
        putIfPresent(out, "surface", preflight.surface());
        out.put("request", preflight.request());
        out.put("discover_mcp_tools", preflight.discoverMcpTools());
        out.put("mcp_discovery_required", preflight.mcpDiscoveryRequired());
        out.put("validate_tools", preflight.validateTools());
        out.put("tool_validation_required", preflight.toolValidationRequired());
        out.put("validate_request", preflight.validateRequest());
        out.put("request_validation_required", preflight.requestValidationRequired());
        out.put("openai_tool_compatibility", preflight.openAiToolCompatibility());
        out.put("enabled_only", preflight.enabledOnly());
        out.put("feature_profile", preflight.featureProfile());
        out.put("required_contract_version", preflight.requiredContractVersion());
        out.put("required_features", preflight.requiredFeatures());
        out.put("optional_features", preflight.optionalFeatures());
        return out;
    }

    private void putIfPresent(Map<String, Object> out, String key, String value) {
        if (value != null && !value.isBlank()) {
            out.put(key, value);
        }
    }

    private boolean looksLikeStandaloneData(String value) {
        return AgentStreamEventParser.DONE.equals(value)
                || value.startsWith("{")
                || value.startsWith("[");
    }

    private String compat(boolean openAiCompat) {
        return openAiCompat ? "openai" : "mcp";
    }

    private void requireName(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8).replace("+", "%20");
    }

    private static String normalizeBaseUrl(String value) {
        String out = value == null || value.isBlank() ? "http://localhost:8080" : value.trim();
        while (out.endsWith("/")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    private static String normalizeApiKey(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            return COMMUNITY_API_KEY;
        }
        return apiKey;
    }

    private static String authorizationValue(String apiKey) {
        return AUTHORIZATION_SCHEME + " " + apiKey;
    }
}
