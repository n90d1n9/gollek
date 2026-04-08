package tech.kayys.gollek.multimodal.stress;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Stress tests for multimodal inference service.
 * Tests system behavior under extreme load.
 */
class MultimodalStressTest {

    private MultimodalInferenceService service;
    private ExecutorService executorService;

    @BeforeEach
    void setUp() {
        service = new MultimodalInferenceService();
        service.initialize();
        executorService = Executors.newFixedThreadPool(10);
    }

    @Test
    void testHighConcurrencyLoad() throws Exception {
        // Test with 100 concurrent requests
        int concurrentRequests = 100;
        CountDownLatch latch = new CountDownLatch(concurrentRequests);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);
        AtomicLong totalDuration = new AtomicLong(0);

        for (int i = 0; i < concurrentRequests; i++) {
            final int requestId = i;
            executorService.submit(() -> {
                try {
                    long startTime = System.currentTimeMillis();
                    
                    MultimodalRequest request = MultimodalRequest.builder()
                        .requestId("stress-" + requestId)
                        .model("clip-vit-base")
                        .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                        .outputConfig(MultimodalRequest.OutputConfig.builder()
                            .outputModalities(ModalityType.EMBEDDING)
                            .build())
                        .build();

                    Uni<MultimodalResponse> response = service.infer(request);
                    MultimodalResponse result = response.await().atMost(Duration.ofSeconds(30));
                    
                    if (result != null && result.getStatus() == MultimodalResponse.ResponseStatus.SUCCESS) {
                        successCount.incrementAndGet();
                    } else {
                        failureCount.incrementAndGet();
                    }
                    
                    totalDuration.addAndGet(System.currentTimeMillis() - startTime);
                    
                } catch (Exception e) {
                    failureCount.incrementAndGet();
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all requests to complete
        boolean completed = latch.await(2, TimeUnit.MINUTES);
        assertThat(completed).isTrue();

        // Validate results
        System.out.println("=== High Concurrency Stress Test Results ===");
        System.out.println("Total Requests: " + concurrentRequests);
        System.out.println("Successful: " + successCount.get());
        System.out.println("Failed: " + failureCount.get());
        System.out.println("Average Duration: " + (totalDuration.get() / concurrentRequests) + "ms");
        System.out.println("Throughput: " + (concurrentRequests / (totalDuration.get() / 1000.0)) + " req/s");
        System.out.println("===========================================");

        // Expect at least 90% success rate under load
        assertThat(successCount.get()).isGreaterThan(concurrentRequests * 90 / 100);
    }

    @Test
    void testSustainedLoad() throws Exception {
        // Test sustained load for 5 minutes
        int durationMinutes = 5;
        int requestsPerSecond = 10;
        AtomicInteger totalRequests = new AtomicInteger(0);
        AtomicInteger failures = new AtomicInteger(0);
        
        CountDownLatch latch = new CountDownLatch(durationMinutes * 60);
        
        for (int minute = 0; minute < durationMinutes; minute++) {
            for (int second = 0; second < 60; second++) {
                for (int i = 0; i < requestsPerSecond; i++) {
                    executorService.submit(() -> {
                        try {
                            MultimodalRequest request = MultimodalRequest.builder()
                                .requestId("sustained-" + System.currentTimeMillis())
                                .model("clip-vit-base")
                                .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                                .build();

                            service.infer(request).await().atMost(Duration.ofSeconds(10));
                            totalRequests.incrementAndGet();
                            
                        } catch (Exception e) {
                            failures.incrementAndGet();
                        } finally {
                            latch.countDown();
                        }
                    });
                }
                
                // Wait 1 second between batches
                Thread.sleep(1000);
            }
        }

        boolean completed = latch.await(10, TimeUnit.MINUTES);
        assertThat(completed).isTrue();

        System.out.println("=== Sustained Load Test Results ===");
        System.out.println("Duration: " + durationMinutes + " minutes");
        System.out.println("Total Requests: " + totalRequests.get());
        System.out.println("Failures: " + failures.get());
        System.out.println("Success Rate: " + (100.0 - (failures.get() * 100.0 / totalRequests.get())) + "%");
        System.out.println("Avg Throughput: " + (totalRequests.get() / (durationMinutes * 60)) + " req/s");
        System.out.println("====================================");

        // Expect at least 95% success rate over sustained period
        assertThat(failures.get()).isLessThan(totalRequests.get() * 5 / 100);
    }

    @Test
    void testBatchStress() throws Exception {
        // Test batch processing under stress
        int batchSize = 50;
        int batchCount = 10;
        AtomicInteger totalProcessed = new AtomicInteger(0);
        
        List<MultimodalRequest> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            batch.add(MultimodalRequest.builder()
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                .build());
        }

        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < batchCount; i++) {
            Uni<List<MultimodalResponse>> response = service.inferBatch(batch);
            List<MultimodalResponse> results = response.await().atMost(Duration.ofMinutes(2));
            
            if (results != null && results.size() == batchSize) {
                totalProcessed.addAndGet(results.size());
            }
        }
        
        long totalDuration = System.currentTimeMillis() - startTime;

        System.out.println("=== Batch Stress Test Results ===");
        System.out.println("Batch Size: " + batchSize);
        System.out.println("Batch Count: " + batchCount);
        System.out.println("Total Processed: " + totalProcessed.get());
        System.out.println("Total Duration: " + totalDuration + "ms");
        System.out.println("Avg Batch Duration: " + (totalDuration / batchCount) + "ms");
        System.out.println("Throughput: " + (totalProcessed.get() / (totalDuration / 1000.0)) + " req/s");
        System.out.println("=================================");

        assertThat(totalProcessed.get()).isEqualTo(batchSize * batchCount);
    }

    @Test
    void testMemoryPressure() throws Exception {
        // Test behavior under memory pressure
        int requestCount = 200;
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger oomCount = new AtomicInteger(0);
        CountDownLatch latch = new CountDownLatch(requestCount);

        for (int i = 0; i < requestCount; i++) {
            final int requestId = i;
            executorService.submit(() -> {
                try {
                    MultimodalRequest request = MultimodalRequest.builder()
                        .requestId("memory-" + requestId)
                        .model("clip-vit-base")
                        .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                        .build();

                    service.infer(request).await().atMost(Duration.ofSeconds(10));
                    successCount.incrementAndGet();
                    
                } catch (OutOfMemoryError e) {
                    oomCount.incrementAndGet();
                } catch (Exception e) {
                    // Other exceptions are OK under memory pressure
                } finally {
                    latch.countDown();
                }
            });
        }

        boolean completed = latch.await(5, TimeUnit.MINUTES);
        assertThat(completed).isTrue();

        System.out.println("=== Memory Pressure Test Results ===");
        System.out.println("Total Requests: " + requestCount);
        System.out.println("Successful: " + successCount.get());
        System.out.println("OOM Errors: " + oomCount.get());
        System.out.println("====================================");

        // System should handle memory pressure gracefully (no OOM)
        assertThat(oomCount.get()).isZero();
    }

    @Test
    void testServiceRecovery() throws Exception {
        // Test service recovery after heavy load
        int heavyLoadRequests = 100;
        
        // Apply heavy load
        List<Uni<MultimodalResponse>> responses = new ArrayList<>();
        for (int i = 0; i < heavyLoadRequests; i++) {
            MultimodalRequest request = MultimodalRequest.builder()
                .requestId("recovery-" + i)
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                .build();
            responses.add(service.infer(request));
        }

        // Wait for all to complete
        for (Uni<MultimodalResponse> response : responses) {
            try {
                response.await().atMost(Duration.ofSeconds(30));
            } catch (Exception e) {
                // Ignore failures during heavy load
            }
        }

        // Verify service recovers and can handle normal requests
        MultimodalRequest normalRequest = MultimodalRequest.builder()
            .requestId("post-stress")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
            .build();

        MultimodalResponse response = service.infer(normalRequest)
            .await().atMost(Duration.ofSeconds(10));

        System.out.println("=== Service Recovery Test Results ===");
        System.out.println("Heavy Load Requests: " + heavyLoadRequests);
        System.out.println("Post-Stress Request: " + (response != null ? "SUCCESS" : "FAILED"));
        System.out.println("======================================");

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
    }

    private byte[] createTestImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }
}
