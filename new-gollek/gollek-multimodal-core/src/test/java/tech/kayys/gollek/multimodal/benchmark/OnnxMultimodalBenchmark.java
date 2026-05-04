package tech.kayys.gollek.multimodal.benchmark;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.onnx.processor.OnnxMultimodalProcessor;

import java.util.concurrent.TimeUnit;

/**
 * Performance benchmarks for ONNX multimodal processor.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 5, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 5, timeUnit = TimeUnit.SECONDS)
@Fork(2)
@Threads(1)
public class OnnxMultimodalBenchmark {

    private OnnxMultimodalProcessor processor;
    private byte[] testImage;
    private MultimodalRequest embeddingRequest;
    private MultimodalRequest classificationRequest;

    @Setup
    public void setUp() throws Exception {
        processor = new OnnxMultimodalProcessor();
        testImage = createTestImage();
        
        // Embedding request (CLIP)
        embeddingRequest = MultimodalRequest.builder()
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .outputConfig(MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.EMBEDDING)
                .build())
            .build();
        
        // Classification request (ResNet)
        classificationRequest = MultimodalRequest.builder()
            .model("resnet50")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .build();
    }

    @Benchmark
    public void benchmarkClipEmbedding(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(embeddingRequest)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    @Benchmark
    public void benchmarkResNetClassification(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(classificationRequest)
            .await().atMost(java.time.Duration.ofSeconds(5));
        bh.consume(response);
    }

    @Benchmark
    @Threads(4)
    public void benchmarkConcurrentEmbedding(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(embeddingRequest)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    @Benchmark
    @Threads(8)
    public void benchmarkHighConcurrencyEmbedding(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(embeddingRequest)
            .await().atMost(java.time.Duration.ofSeconds(10));
        bh.consume(response);
    }

    @Benchmark
    public void benchmarkBatchEmbedding(Blackhole bh) throws Exception {
        // Process 10 images in sequence (simulates batch)
        for (int i = 0; i < 10; i++) {
            MultimodalResponse response = processor.process(embeddingRequest)
                .await().atMost(java.time.Duration.ofSeconds(10));
            bh.consume(response);
        }
    }

    private byte[] createTestImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }
}
