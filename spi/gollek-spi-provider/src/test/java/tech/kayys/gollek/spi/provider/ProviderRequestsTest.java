package tech.kayys.gollek.spi.provider;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.time.Duration;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ProviderRequests} — the fluent CopyBuilder utility
 * for creating modified copies of {@link ProviderRequest}.
 */
class ProviderRequestsTest {

        @Test
        void withModelReplacesModelOnly() {
                ProviderRequest original = ProviderRequest.builder()
                                .model("original-model")
                                .message(Message.user("Hello"))
                                .parameter("temperature", 0.7)
                                .streaming(true)
                                .userId("user1")
                                .build();

                ProviderRequest patched = ProviderRequests.withModel(original, "/path/to/resolved.gguf");

                // Model should be replaced
                assertEquals("/path/to/resolved.gguf", patched.getModel());

                // All other fields should be preserved
                assertEquals(original.getRequestId(), patched.getRequestId());
                assertEquals(original.getMessages().size(), patched.getMessages().size());
                assertEquals(original.getMessages().get(0).getContent(), patched.getMessages().get(0).getContent());
                assertEquals(original.getParameters().get("temperature"), patched.getParameters().get("temperature"));
                assertEquals(original.isStreaming(), patched.isStreaming());
                assertEquals(original.getUserId(), patched.getUserId());
        }

        @Test
        void withModelNullSourceThrows() {
                assertThrows(NullPointerException.class,
                                () -> ProviderRequests.withModel(null, "/path"));
        }

        @Test
        void withModelNullNewModelThrows() {
                ProviderRequest src = ProviderRequest.builder()
                                .model("test")
                                .message(Message.user("hi"))
                                .build();

                assertThrows(NullPointerException.class,
                                () -> ProviderRequests.withModel(src, null));
        }

        @Test
        void copyBuilderCanModifyMultipleFields() {
                ProviderRequest original = ProviderRequest.builder()
                                .model("test-model")
                                .message(Message.user("Hello"))
                                .streaming(false)
                                .timeout(Duration.ofSeconds(30))
                                .userId("user-a")
                                .build();

                ProviderRequest patched = ProviderRequests.copyOf(original)
                                .model("new-model")
                                .streaming(true)
                                .timeout(Duration.ofMinutes(2))
                                .userId("user-b")
                                .build();

                assertEquals("new-model", patched.getModel());
                assertTrue(patched.isStreaming());
                assertEquals(Duration.ofMinutes(2), patched.getTimeout());
                assertEquals("user-b", patched.getUserId().orElse(null));

                // Unchanged fields preserved
                assertEquals(original.getRequestId(), patched.getRequestId());
                assertEquals(original.getMessages(), patched.getMessages());
        }

        @Test
        void copyOfNullThrows() {
                assertThrows(NullPointerException.class,
                                () -> ProviderRequests.copyOf(null));
        }

        @Test
        void copyPreservesMetadata() {
                ProviderRequest original = ProviderRequest.builder()
                                .model("test")
                                .message(Message.user("hello"))
                                .metadata("tenantId", "tenant-1")
                                .metadata("region", "us-east-1")
                                .build();

                ProviderRequest copy = ProviderRequests.withModel(original, "new-model");

                assertEquals("tenant-1", copy.getMetadata().get("tenantId"));
                assertEquals("us-east-1", copy.getMetadata().get("region"));
        }

        @Test
        void copyPreservesSessionAndTraceIds() {
                ProviderRequest original = ProviderRequest.builder()
                                .model("test")
                                .message(Message.user("hello"))
                                .sessionId("session-123")
                                .traceId("trace-abc")
                                .apiKey("my-key")
                                .build();

                ProviderRequest copy = ProviderRequests.withModel(original, "new");

                assertEquals("session-123", copy.getSessionId().orElse(null));
                assertEquals("trace-abc", copy.getTraceId().orElse(null));
                assertEquals("my-key", copy.getApiKey().orElse(null));
        }

        @Test
        void fromInferenceRequestPreservesToolsToolChoiceAndRagMetadata() {
                ToolDefinition tool = ToolDefinition.builder()
                                .name("searchDocs")
                                .description("Search project docs")
                                .parameters(Map.of("type", "object"))
                                .build();
                InferenceRequest inference = InferenceRequest.builder()
                                .model("user-model")
                                .message(Message.system("Use tools when needed."))
                                .message(Message.user("Find the install profile docs."))
                                .tool(tool)
                                .toolChoice("auto")
                                .metadata("rag_enabled", true)
                                .parameter("embedding_model", "embed-model")
                                .build();

                ProviderRequest provider = ProviderRequests.fromInferenceRequest(
                                inference,
                                "provider-model",
                                true,
                                Duration.ofSeconds(15),
                                "native",
                                Map.of(),
                                Map.of());

                assertEquals("provider-model", provider.getModel());
                assertTrue(provider.isStreaming());
                assertEquals(1, provider.getTools().size());
                assertEquals("searchDocs", provider.getTools().get(0).getName());
                assertEquals("auto", provider.getToolChoice());
                assertEquals(true, provider.getMetadata().get("rag_enabled"));
                assertEquals("embed-model", provider.getParameters().get("embedding_model"));
        }
}
