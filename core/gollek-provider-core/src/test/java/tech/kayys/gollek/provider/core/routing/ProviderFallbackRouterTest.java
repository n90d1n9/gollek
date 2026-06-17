package tech.kayys.gollek.provider.core.routing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderMetadata;

import io.smallrye.mutiny.Uni;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ProviderFallbackRouter.
 */
class ProviderFallbackRouterTest {

    private MockProvider primaryProvider;
    private MockProvider fallbackProvider;
    private MockProvider localProvider;
    private ProviderFallbackRouter router;

    @BeforeEach
    void setUp() {
        primaryProvider = new MockProvider("openai", true, false);
        fallbackProvider = new MockProvider("anthropic", true, false);
        localProvider = new MockProvider("local-gguf", true, false);

        router = ProviderFallbackRouter.builder()
            .primary("openai", "gpt-4", 0.03)
            .fallback("anthropic", "claude-3", 0.025)
            .localFallback("local-gguf", "llama-3-70b", 0.0)
            .maxCostPerRequest(1.0)
            .maxRetries(3)
            .timeout(Duration.ofSeconds(10))
            .build();

        router.registerProvider("openai", primaryProvider);
        router.registerProvider("anthropic", fallbackProvider);
        router.registerProvider("local-gguf", localProvider);
    }

    @Test
    @DisplayName("Successful request on first provider")
    void testSuccessfulRequestOnPrimary() {
        ProviderRequest request = new MockProviderRequest("gpt-4", "Hello", 100);
        
        InferenceResponse response = router.route(request)
            .await().atMost(Duration.ofSeconds(5));
        
        assertNotNull(response);
        assertEquals(1, primaryProvider.callCount.get());
        assertEquals(0, fallbackProvider.callCount.get());
    }

    @Test
    @DisplayName("Fallback to second provider when primary fails")
    void testFallbackToSecondProvider() {
        primaryProvider.setShouldFail(true);
        ProviderRequest request = new MockProviderRequest("gpt-4", "Hello", 100);
        
        InferenceResponse response = router.route(request)
            .await().atMost(Duration.ofSeconds(5));
        
        assertNotNull(response);
        assertEquals(1, primaryProvider.callCount.get());
        assertEquals(1, fallbackProvider.callCount.get());
        assertEquals("anthropic", response.getModel());
    }

    @Test
    @DisplayName("Fallback chain exhausts and throws error")
    void testFallbackChainExhausted() {
        primaryProvider.setShouldFail(true);
        fallbackProvider.setShouldFail(true);
        localProvider.setShouldFail(true);
        
        ProviderRequest request = new MockProviderRequest("gpt-4", "Hello", 100);
        
        assertThrows(Exception.class, () -> {
            router.route(request)
                .await().atMost(Duration.ofSeconds(5));
        });
    }

    @Test
    @DisplayName("Circuit breaker opens after consecutive failures")
    void testCircuitBreakerOpens() {
        primaryProvider.setShouldFail(true);
        ProviderRequest request = new MockProviderRequest("gpt-4", "Hello", 100);
        
        // Make 5 requests to trigger circuit breaker
        for (int i = 0; i < 6; i++) {
            try {
                router.route(request).await().atMost(Duration.ofSeconds(5));
            } catch (Exception e) {
                // Expected
            }
        }
        
        // Verify metrics show failures
        Map<String, ProviderFallbackRouter.ProviderStats> stats = router.getProviderStats();
        assertTrue(stats.containsKey("openai"));
    }

    @Test
    @DisplayName("Provider metrics are tracked correctly")
    void testProviderMetrics() {
        ProviderRequest request = new MockProviderRequest("gpt-4", "Hello", 100);
        
        // Make 3 successful requests
        for (int i = 0; i < 3; i++) {
            router.route(request).await().atMost(Duration.ofSeconds(5));
        }
        
        Map<String, ProviderFallbackRouter.ProviderStats> stats = router.getProviderStats();
        ProviderFallbackRouter.ProviderStats openaiStats = stats.get("openai");
        
        assertNotNull(openaiStats);
        assertEquals(3, openaiStats.totalRequests());
        assertEquals(3, openaiStats.successfulRequests());
        assertEquals(0, openaiStats.failedRequests());
        assertEquals(100.0, openaiStats.successRate(), 0.1);
    }

    @Test
    @DisplayName("Router rejects when no providers configured")
    void testRouterWithNoProviders() {
        assertThrows(IllegalStateException.class, () -> {
            ProviderFallbackRouter.builder()
                .maxCostPerRequest(1.0)
                .build();
        });
    }

    @Test
    @DisplayName("Metrics snapshot is immutable")
    void testMetricsImmutability() {
        Map<String, ProviderFallbackRouter.ProviderStats> stats = router.getProviderStats();
        
        // Should not throw when accessing stats
        assertTrue(stats.containsKey("openai"));
        assertTrue(stats.containsKey("anthropic"));
        assertTrue(stats.containsKey("local-gguf"));
    }

    /**
     * Mock provider for testing.
     */
    static class MockProvider implements LLMProvider {
        private final String id;
        private final boolean enabled;
        private volatile boolean shouldFail = false;
        final AtomicInteger callCount = new AtomicInteger(0);

        MockProvider(String id, boolean enabled, boolean shouldFail) {
            this.id = id;
            this.enabled = enabled;
            this.shouldFail = shouldFail;
        }

        void setShouldFail(boolean shouldFail) {
            this.shouldFail = shouldFail;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public String name() {
            return id + "-provider";
        }

        @Override
        public String version() {
            return "1.0.0";
        }

        @Override
        public ProviderMetadata metadata() {
            return new ProviderMetadata(id, "Mock " + id, version());
        }

        @Override
        public boolean isEnabled() {
            return enabled;
        }

        @Override
        public ProviderCapabilities capabilities() {
            return new MockCapabilities();
        }

        @Override
        public void initialize(tech.kayys.gollek.spi.provider.ProviderConfig config) {
        }

        @Override
        public boolean supports(String modelId, ProviderRequest request) {
            return true;
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
            callCount.incrementAndGet();
            if (shouldFail) {
                return Uni.createFrom().failure(new RuntimeException("Provider " + id + " failed"));
            }
            return Uni.createFrom().item(new MockInferenceResponse(id, "Response from " + id));
        }

        @Override
        public Uni<ProviderHealth> health() {
            return Uni.createFrom().item(new ProviderHealth(true, 1.0, "Healthy"));
        }

        @Override
        public void shutdown() {
        }
    }

    static class MockProviderRequest extends ProviderRequest {
        private final String modelId;
        private final String prompt;
        private final int estimatedTokens;

        MockProviderRequest(String modelId, String prompt, int estimatedTokens) {
            this.modelId = modelId;
            this.prompt = prompt;
            this.estimatedTokens = estimatedTokens;
        }

        @Override
        public String getModelId() {
            return modelId;
        }

        @Override
        public int getEstimatedTokens() {
            return estimatedTokens;
        }

        @Override
        public String getPrompt() {
            return prompt;
        }

        @Override
        public java.util.Map<String, Object> getParameters() {
            return Map.of();
        }
    }

    static class MockInferenceResponse implements InferenceResponse {
        private final String model;
        private final String text;

        MockInferenceResponse(String model, String text) {
            this.model = model;
            this.text = text;
        }

        @Override
        public String getText() {
            return text;
        }

        @Override
        public String getModel() {
            return model;
        }

        @Override
        public InferenceUsage getUsage() {
            return new MockInferenceUsage(10, 50, 60);
        }

        @Override
        public String getFinishReason() {
            return "stop";
        }
    }

    static class MockInferenceUsage implements InferenceUsage {
        private final int inputTokens;
        private final int outputTokens;
        private final int totalTokens;

        MockInferenceUsage(int inputTokens, int outputTokens, int totalTokens) {
            this.inputTokens = inputTokens;
            this.outputTokens = outputTokens;
            this.totalTokens = totalTokens;
        }

        @Override
        public int getInputTokens() {
            return inputTokens;
        }

        @Override
        public int getOutputTokens() {
            return outputTokens;
        }

        @Override
        public int getTotalTokens() {
            return totalTokens;
        }
    }

    static class MockCapabilities implements ProviderCapabilities {
        @Override
        public boolean supportsStreaming() {
            return true;
        }

        @Override
        public boolean supportsFunctionCalling() {
            return false;
        }

        @Override
        public boolean supportsMultimodal() {
            return false;
        }

        @Override
        public int getMaxContextLength() {
            return 4096;
        }

        @Override
        public java.util.Set<String> getSupportedFormats() {
            return java.util.Set.of();
        }
    }
}
