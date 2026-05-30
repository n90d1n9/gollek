package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Agent-facing discovery and validation client for a Gollek serving endpoint.
 *
 * <p>This client intentionally exposes contracts, validation, and MCP tool
 * schemas only. Planning, authorization, tool execution, and workflow state stay
 * with the caller's agent runtime.
 */
public final class AgentIntegrationClient {
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
    public Map<String, Object> capabilities() throws SdkException {
        return capabilities(AgentRequestOptions.empty());
    }

    /**
     * Returns Gollek's agent-facing serving capabilities and endpoint links.
     */
    public Map<String, Object> capabilities(AgentRequestOptions options) throws SdkException {
        return get("/v1/agent/capabilities", options);
    }

    /**
     * Returns the machine-readable agent integration contract.
     */
    public Map<String, Object> contract() throws SdkException {
        return contract(AgentRequestOptions.empty());
    }

    /**
     * Returns the machine-readable agent integration contract.
     */
    public Map<String, Object> contract(AgentRequestOptions options) throws SdkException {
        return get("/v1/agent/contract", options);
    }

    /**
     * Validates a chat, Responses, or embedding payload without invoking a model.
     */
    public Map<String, Object> validateRequest(Map<String, Object> request) throws SdkException {
        return validateRequest(null, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a chat, Responses, or embedding payload without invoking a model.
     */
    public Map<String, Object> validateRequest(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return validateRequest(null, request, options);
    }

    /**
     * Validates a payload for a specific compatibility surface without invoking a model.
     */
    public Map<String, Object> validateRequest(String surface, Map<String, Object> request) throws SdkException {
        return validateRequest(surface, request, AgentRequestOptions.empty());
    }

    /**
     * Validates a payload for a specific compatibility surface without invoking a model.
     */
    public Map<String, Object> validateRequest(
            String surface,
            Map<String, Object> request,
            AgentRequestOptions options) throws SdkException {
        String path = "/v1/agent/validate";
        if (surface != null && !surface.isBlank()) {
            path += "?surface=" + encode(surface);
        }
        return post(path, body(request), options);
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions without executing tools.
     */
    public Map<String, Object> validateTools(Map<String, Object> request) throws SdkException {
        return validateTools(request, AgentRequestOptions.empty());
    }

    /**
     * Validates OpenAI-compatible function and MCP tool definitions without executing tools.
     */
    public Map<String, Object> validateTools(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return post("/v1/agent/tools/validate", body(request), options);
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request.
     */
    public Map<String, Object> createChatCompletion(Map<String, Object> request) throws SdkException {
        return createChatCompletion(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request.
     */
    public Map<String, Object> createChatCompletion(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return post("/v1/chat/completions", nonStreamingBody(request), options);
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request and returns a typed response view.
     */
    public AgentResponseView createChatCompletionView(Map<String, Object> request) throws SdkException {
        return createChatCompletionView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible chat completion request and returns a typed response view.
     */
    public AgentResponseView createChatCompletionView(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return AgentResponseView.from(createChatCompletion(request, options), objectMapper);
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request.
     */
    public Map<String, Object> createResponse(Map<String, Object> request) throws SdkException {
        return createResponse(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request.
     */
    public Map<String, Object> createResponse(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return post("/v1/responses", nonStreamingBody(request), options);
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request and returns a typed response view.
     */
    public AgentResponseView createResponseView(Map<String, Object> request) throws SdkException {
        return createResponseView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends a non-streaming OpenAI-compatible Responses request and returns a typed response view.
     */
    public AgentResponseView createResponseView(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return AgentResponseView.from(createResponse(request, options), objectMapper);
    }

    /**
     * Sends an OpenAI-compatible embedding request.
     */
    public Map<String, Object> createEmbedding(Map<String, Object> request) throws SdkException {
        return createEmbedding(request, AgentRequestOptions.empty());
    }

    /**
     * Sends an OpenAI-compatible embedding request.
     */
    public Map<String, Object> createEmbedding(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return post("/v1/embeddings", body(request), options);
    }

    /**
     * Sends an OpenAI-compatible embedding request and returns a typed embedding view.
     */
    public AgentEmbeddingView createEmbeddingView(Map<String, Object> request) throws SdkException {
        return createEmbeddingView(request, AgentRequestOptions.empty());
    }

    /**
     * Sends an OpenAI-compatible embedding request and returns a typed embedding view.
     */
    public AgentEmbeddingView createEmbeddingView(Map<String, Object> request, AgentRequestOptions options)
            throws SdkException {
        return AgentEmbeddingView.from(createEmbedding(request, options), objectMapper);
    }

    /**
     * Lists registered MCP servers, if the server runtime exposes a registry.
     */
    public Map<String, Object> mcpServers() throws SdkException {
        return mcpServers(AgentRequestOptions.empty());
    }

    /**
     * Lists registered MCP servers, if the server runtime exposes a registry.
     */
    public Map<String, Object> mcpServers(AgentRequestOptions options) throws SdkException {
        return get("/v1/mcp/servers", options);
    }

    /**
     * Returns one redacted MCP server registration.
     */
    public Map<String, Object> mcpServer(String serverName) throws SdkException {
        return mcpServer(serverName, AgentRequestOptions.empty());
    }

    /**
     * Returns one redacted MCP server registration.
     */
    public Map<String, Object> mcpServer(String serverName, AgentRequestOptions options) throws SdkException {
        requireName(serverName, "serverName");
        return get("/v1/mcp/servers/" + encode(serverName), options);
    }

    /**
     * Lists MCP tools across enabled servers using the server default representation.
     */
    public Map<String, Object> mcpTools() throws SdkException {
        return mcpTools(AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across enabled servers using the server default representation.
     */
    public Map<String, Object> mcpTools(AgentRequestOptions options) throws SdkException {
        return get("/v1/mcp/tools", options);
    }

    /**
     * Lists MCP tools across servers, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpTools(boolean openAiCompat, boolean enabledOnly) throws SdkException {
        return mcpTools(openAiCompat, enabledOnly, AgentRequestOptions.empty());
    }

    /**
     * Lists MCP tools across servers, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpTools(
            boolean openAiCompat,
            boolean enabledOnly,
            AgentRequestOptions options) throws SdkException {
        return get("/v1/mcp/tools?compat=" + compat(openAiCompat) + "&enabledOnly=" + enabledOnly, options);
    }

    /**
     * Lists tools for one MCP server, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpServerTools(String serverName, boolean openAiCompat) throws SdkException {
        return mcpServerTools(serverName, openAiCompat, AgentRequestOptions.empty());
    }

    /**
     * Lists tools for one MCP server, optionally converted to OpenAI function-tool definitions.
     */
    public Map<String, Object> mcpServerTools(
            String serverName,
            boolean openAiCompat,
            AgentRequestOptions options) throws SdkException {
        requireName(serverName, "serverName");
        return get("/v1/mcp/servers/" + encode(serverName) + "/tools?compat=" + compat(openAiCompat), options);
    }

    private Map<String, Object> get(String path, AgentRequestOptions options) throws SdkException {
        HttpRequest request = request(path, options)
                .GET()
                .build();
        return send(request);
    }

    private Map<String, Object> post(String path, Map<String, Object> body, AgentRequestOptions options)
            throws SdkException {
        try {
            HttpRequest request = request(path, options)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)))
                    .build();
            return send(request);
        } catch (IOException e) {
            throw new SdkException("SDK_ERR_AGENT_INTEGRATION", "Failed to encode agent integration request", e);
        }
    }

    private HttpRequest.Builder request(String path, AgentRequestOptions options) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("Accept", "application/json")
                .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                .timeout(Duration.ofSeconds(30));
        applyHeaders(builder, options);
        return builder;
    }

    private void applyHeaders(HttpRequest.Builder builder, AgentRequestOptions options) {
        AgentRequestOptions effective = options == null ? AgentRequestOptions.empty() : options;
        effective.toHeaders().forEach(builder::setHeader);
    }

    private Map<String, Object> send(HttpRequest request) throws SdkException {
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return handleResponse(response);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new SdkException("SDK_ERR_AGENT_INTEGRATION", "Agent integration request interrupted", e);
        } catch (IOException e) {
            throw new SdkException("SDK_ERR_AGENT_INTEGRATION", "Agent integration request failed", e);
        }
    }

    private Map<String, Object> handleResponse(HttpResponse<String> response) throws SdkException {
        String responseBody = response.body();
        if (response.statusCode() >= 200 && response.statusCode() < 300) {
            if (responseBody == null || responseBody.isBlank()) {
                return new LinkedHashMap<>();
            }
            try {
                return objectMapper.readValue(responseBody, MAP_TYPE);
            } catch (IOException e) {
                throw new SdkException("SDK_ERR_AGENT_INTEGRATION", "Failed to parse agent integration response", e);
            }
        }

        throw new SdkException(
                "SDK_ERR_AGENT_INTEGRATION",
                "Agent integration request failed with status " + response.statusCode() + ": "
                        + (responseBody == null ? "" : responseBody));
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
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
