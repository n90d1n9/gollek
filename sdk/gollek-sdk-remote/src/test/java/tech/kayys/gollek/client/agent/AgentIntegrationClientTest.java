package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;

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
                .firstValue(ApiKeyConstants.HEADER_API_KEY).orElseThrow());
        assertEquals("ApiKey test-key", httpClient.lastRequest().headers()
                .firstValue(ApiKeyConstants.HEADER_AUTHORIZATION).orElseThrow());
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
    void mapsHttpErrorsToSdkException() {
        FakeHttpClient httpClient = new FakeHttpClient(400, """
                {"error":{"message":"invalid tools"}}
                """);
        AgentIntegrationClient client = client(httpClient);

        SdkException exception = assertThrows(
                SdkException.class,
                () -> client.validateTools(Map.of("tools", "bad")));

        assertEquals("SDK_ERR_AGENT_INTEGRATION", exception.getErrorCode());
    }

    private AgentIntegrationClient client(FakeHttpClient httpClient) {
        return new AgentIntegrationClient(
                httpClient,
                new ObjectMapper(),
                "http://localhost:8080/",
                "test-key");
    }

    private static String requestBody(HttpRequest request) throws Exception {
        BodyCaptureSubscriber subscriber = new BodyCaptureSubscriber();
        request.bodyPublisher().orElseThrow().subscribe(subscriber);
        return subscriber.body();
    }

    private static final class FakeHttpClient extends HttpClient {
        private final int statusCode;
        private final String body;
        private HttpRequest lastRequest;

        private FakeHttpClient(int statusCode, String body) {
            this.statusCode = statusCode;
            this.body = body;
        }

        private HttpRequest lastRequest() {
            return lastRequest;
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
            lastRequest = request;
            return (HttpResponse<T>) new FakeResponse(request, statusCode, body);
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

    private record FakeResponse(HttpRequest request, int statusCode, String body) implements HttpResponse<String> {
        @Override
        public Optional<HttpResponse<String>> previousResponse() {
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
