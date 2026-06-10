package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentEmbeddingViewTest {

    @Test
    void extractsOpenAiEmbeddingVectorsUsageAndTrace() throws Exception {
        AgentEmbeddingView view = AgentEmbeddingView.fromJson("""
                {
                  "object": "list",
                  "model": "demo-embed",
                  "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
                  ],
                  "usage": {
                    "prompt_tokens": 8,
                    "total_tokens": 8
                  },
                  "metadata": {
                    "tenant": "agent-project",
                    "gollek_trace": {
                      "request_id": "req_embed_123",
                      "trace_id": "trace_embed_123"
                    }
                  }
                }
                """);

        assertEquals("list", view.object());
        assertEquals("demo-embed", view.model());
        assertTrue(view.hasEmbeddings());
        assertEquals(2, view.count());
        assertEquals(3, view.dimensions());
        assertEquals(List.of(0.1, 0.2, 0.3), view.firstVector());
        assertEquals(List.of(0.4, 0.5, 0.6), view.embeddings().get(1).vector());
        assertEquals(8, view.usage().totalTokens());
        assertEquals("agent-project", view.metadata().get("tenant"));
        assertEquals("req_embed_123", view.trace().get("request_id"));
        assertEquals("trace_embed_123", view.trace().get("trace_id"));
    }

    @Test
    void extractsNativeEmbeddingResponseShape() throws Exception {
        AgentEmbeddingView view = AgentEmbeddingView.fromJson("""
                {
                  "requestId": "req-native",
                  "model": "native-embed",
                  "embeddings": [
                    [1.0, 2.0, 3.0]
                  ],
                  "dimension": 3,
                  "metadata": {"source": "native"}
                }
                """);

        assertEquals("req-native", view.requestId());
        assertEquals("native-embed", view.model());
        assertEquals(1, view.count());
        assertEquals(3, view.dimensions());
        assertEquals(List.of(1.0, 2.0, 3.0), view.vectors().get(0));
        assertEquals("native", view.metadata().get("source"));
    }
}
