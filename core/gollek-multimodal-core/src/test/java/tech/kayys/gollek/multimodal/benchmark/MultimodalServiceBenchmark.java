package tech.kayys.gollek.multimodal.benchmark;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * End-to-end performance benchmarks for multimodal inference service.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 10, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 10, timeUnit = TimeUnit.SECONDS)
@Fork(2)
public class MultimodalServiceBenchmark {

    private MultimodalInferenceService service;
    private byte[] testImage;
    private List<MultimodalRequest> batchRequests;

    @Setup
    public void setUp() throws Exception {
        service = new MultimodalInferenceService();
        service.initialize();
        testImage = createTestImage();
        
        // Create batch of 20 requests
        batchRequests = IntStream.range(0, 20)
            .mapToObj(i -> MultimodalRequest.builder()
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                .outputConfig(MultimodalRequest.OutputConfig.builder()
                    .outputModalities(ModalityType.EMBEDDING)
                    .build())
                .requestId("batch-" + i)
                .build())
            .collect(Collectors.toList());
    }

    @Benchmark
    public void benchmarkSingleRequestThroughput(Blackhole bh) throws Exception {
        MultimodalRequest request = MultimodalRequest.builder()
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .outputConfig(MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.EMBEDDING)
                .build())
            .build();
        
        MultimodalResponse response = service.infer(request)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    @Benchmark
    public void benchmarkBatchThroughput(Blackhole bh) throws Exception {
        List<MultimodalResponse> responses = batchRequests.stream()
            .map(request -> {
                try {
                    return service.infer(request)
                        .await().atMost(java.time.Duration.ofSeconds(10));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            })
            .collect(Collectors.toList());
        
        bh.consume(responses);
    }

    @Benchmark
    @Threads(4)
    public void benchmarkConcurrentThroughput(Blackhole bh) throws Exception {
        MultimodalRequest request = MultimodalRequest.builder()
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .build();
        
        MultimodalResponse response = service.infer(request)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    @Benchmark
    @Threads(8)
    public void benchmarkHighConcurrencyThroughput(Blackhole bh) throws Exception {
        MultimodalRequest request = MultimodalRequest.builder()
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .build();
        
        MultimodalResponse response = service.infer(request)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    private byte[] createTestImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }
}
