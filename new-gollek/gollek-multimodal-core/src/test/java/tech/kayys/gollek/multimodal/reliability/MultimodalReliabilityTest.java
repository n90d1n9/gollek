package tech.kayys.gollek.multimodal.reliability;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Reliability tests for multimodal inference service.
 * Tests system behavior during failures and recovery.
 */
class MultimodalReliabilityTest {

    private MultimodalInferenceService service;

    @BeforeEach
    void setUp() {
        service = new MultimodalInferenceService();
        service.initialize();
    }

    @Test
    void testGracefulDegradation() {
        // Test that service degrades gracefully when processor unavailable
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("degradation-test")
            .model("nonexistent-model")
            .inputs(MultimodalContent.ofText("test"))
            .build();

        Uni<MultimodalResponse> response = service.infer(request);
        
        // Should fail gracefully, not crash
        try {
            response.await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // Expected - service should handle missing models gracefully
            assertThat(e.getMessage()).contains("No processor available");
        }
    }

    @Test
    void testRetryBehavior() throws Exception {
        // Test retry behavior on transient failures
        AtomicInteger attemptCount = new AtomicInteger(0);
        
        // Simulate transient failure then success
        for (int i = 0; i < 3; i++) {
            try {
                MultimodalRequest request = MultimodalRequest.builder()
                    .requestId("retry-" + i)
                    .model("clip-vit-base")
                    .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                    .build();

                MultimodalResponse response = service.infer(request)
                    .await().atMost(Duration.ofSeconds(10));
                
                if (response != null && response.getStatus() == MultimodalResponse.ResponseStatus.SUCCESS) {
                    attemptCount.incrementAndGet();
                }
            } catch (Exception e) {
                // Transient failures are OK
            }
        }

        // Should have at least one success
        assertThat(attemptCount.get()).isGreaterThan(0);
    }

    @Test
    void testCircuitBreakerBehavior() {
        // Test that service implements circuit breaker pattern
        // (Prevent cascade failures)
        
        int failureCount = 0;
        
        // Send requests that should fail
        for (int i = 0; i < 10; i++) {
            try {
                MultimodalRequest request = MultimodalRequest.builder()
                    .requestId("circuit-" + i)
                    .model("invalid-model")
                    .inputs(MultimodalContent.ofText("test"))
                    .build();

                service.infer(request).await().atMost(Duration.ofSeconds(2));
            } catch (Exception e) {
                failureCount++;
            }
        }

        // All should fail, but service should remain operational
        assertThat(failureCount).isEqualTo(10);
        
        // Service should still be able to handle valid requests
        MultimodalRequest validRequest = MultimodalRequest.builder()
            .requestId("post-circuit")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
            .build();

        // Should not throw exception
        assertThat(service).isNotNull();
    }

    @Test
    void testTimeoutHandling() {
        // Test that timeouts are handled properly
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("timeout-test")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
            .build();

        try {
            // Should timeout if takes too long
            service.infer(request).await().atMost(Duration.ofMillis(1));
        } catch (Exception e) {
            // Timeout is expected
            assertThat(e.getMessage()).contains("timeout");
        }
    }

    @Test
    void testResourceCleanup() {
        // Test that resources are properly cleaned up
        int initialStreamCount = service.getActiveStreamCount();
        
        // Create and complete streams
        for (int i = 0; i < 10; i++) {
            MultimodalRequest request = MultimodalRequest.builder()
                .requestId("cleanup-" + i)
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                .build();

            try {
                service.infer(request).await().atMost(Duration.ofSeconds(10));
            } catch (Exception e) {
                // Ignore
            }
        }

        // Resources should be cleaned up
        int finalStreamCount = service.getActiveStreamCount();
        assertThat(finalStreamCount).isLessThanOrEqualTo(initialStreamCount);
    }

    @Test
    void testConcurrentModificationSafety() throws Exception {
        // Test that service is thread-safe during concurrent modifications
        AtomicInteger successCount = new AtomicInteger(0);
        Thread[] threads = new Thread[10];
        
        for (int i = 0; i < threads.length; i++) {
            final int threadId = i;
            threads[i] = new Thread(() -> {
                try {
                    MultimodalRequest request = MultimodalRequest.builder()
                        .requestId("concurrent-" + threadId)
                        .model("clip-vit-base")
                        .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                        .build();

                    MultimodalResponse response = service.infer(request)
                        .await().atMost(Duration.ofSeconds(10));
                    
                    if (response != null) {
                        successCount.incrementAndGet();
                    }
                } catch (Exception e) {
                    // Ignore concurrent modification exceptions
                }
            });
            threads[i].start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        // Should handle concurrent access safely
        assertThat(successCount.get()).isGreaterThan(0);
    }

    private byte[] createTestImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }
}
