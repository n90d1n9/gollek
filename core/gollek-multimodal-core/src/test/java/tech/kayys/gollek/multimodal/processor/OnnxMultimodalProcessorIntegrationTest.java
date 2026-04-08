package tech.kayys.gollek.multimodal.processor;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.onnx.processor.OnnxMultimodalProcessor;
import org.assertj.core.data.Offset;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for ONNX multimodal processor with vision models.
 */
class OnnxMultimodalProcessorIntegrationTest {

    private OnnxMultimodalProcessor processor;

    @BeforeEach
    void setUp() {
        processor = new OnnxMultimodalProcessor();
    }

    @Test
    void testClipImageEmbedding() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
            Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "clip-vit-base.onnx")),
            "CLIP model not found, skipping test"
        );

        byte[] testImage = loadTestImage("test-dog.jpg");

        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("test-clip-001")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .outputConfig(MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.EMBEDDING)
                .build())
            .build();

        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(10));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()).hasSize(1);
        assertThat(response.getOutputs()[0].getEmbedding()).isNotNull();
        assertThat(response.getOutputs()[0].getEmbedding()).hasSize(512); // CLIP ViT-B embedding size
        
        // Check embedding is normalized
        float[] embedding = response.getOutputs()[0].getEmbedding();
        float norm = calculateNorm(embedding);
        assertThat(norm).isCloseTo(1.0f, within(0.01f));
        
        System.out.println("Embedding norm: " + norm);
    }

    @Test
    void testBlipImageCaptioning() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
            Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "blip-caption.onnx")),
            "BLIP model not found, skipping test"
        );

        byte[] testImage = loadTestImage("test-beach.jpg");

        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("test-blip-001")
            .model("blip-caption")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .build();

        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(10));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()[0].getText()).isNotBlank();
        assertThat(response.getOutputs()[0].getText().length()).isGreaterThan(10);
        
        System.out.println("Caption: " + response.getOutputs()[0].getText());
    }

    @Test
    void testResNetClassification() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
            Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "resnet50.onnx")),
            "ResNet model not found, skipping test"
        );

        byte[] testImage = loadTestImage("test-cat.jpg");

        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("test-resnet-001")
            .model("resnet50")
            .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
            .build();

        Uni<MultimodalResponse> responseUni = processor.process(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(5));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()[0].getText()).contains("Class");
        
        System.out.println("Classification: " + response.getOutputs()[0].getText());
    }

    @Test
    void testTaskTypeDetection() {
        // Test automatic task type detection
        processor.detectTaskType(
            new MultimodalContent[]{
                MultimodalContent.ofBase64Image(new byte[100], "image/jpeg")
            },
            "clip-vit-base"
        );
        
        // Should detect IMAGE_EMBEDDING for CLIP
        // This is a basic smoke test
    }

    @Test
    void testProcessorAvailability() {
        boolean available = processor.isAvailable();
        System.out.println("ONNX processor available: " + available);
        // Should be true if ONNX Runtime is available
    }

    @Test
    void testProcessorId() {
        assertThat(processor.getProcessorId()).isEqualTo("onnx-multimodal");
    }

    @Test
    void testEmbeddingSimilarity() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
            Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "clip-vit-base.onnx")),
            "CLIP model not found, skipping test"
        );

        // Load two similar images
        byte[] image1 = loadTestImage("test-dog1.jpg");
        byte[] image2 = loadTestImage("test-dog2.jpg");
        byte[] image3 = loadTestImage("test-car.jpg");

        // Get embeddings
        float[] embedding1 = getEmbedding(image1);
        float[] embedding2 = getEmbedding(image2);
        float[] embedding3 = getEmbedding(image3);

        // Similar images should have high cosine similarity
        float similarity12 = cosineSimilarity(embedding1, embedding2);
        float similarity13 = cosineSimilarity(embedding1, embedding3);

        System.out.println("Similarity (dog1, dog2): " + similarity12);
        System.out.println("Similarity (dog1, car): " + similarity13);

        // Similar images should have similarity > 0.5
        assertThat(similarity12).isGreaterThan(0.5f);
        // Different images should have lower similarity
        assertThat(similarity12).isGreaterThan(similarity13);
    }

    @Test
    void testConcurrentProcessing() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
            Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "clip-vit-base.onnx")),
            "CLIP model not found, skipping test"
        );

        byte[] testImage = loadTestImage("test-dog.jpg");
        
        // Create 20 concurrent requests
        java.util.List<Uni<MultimodalResponse>> requests = java.util.stream.IntStream.range(0, 20)
            .mapToObj(i -> MultimodalRequest.builder()
                .requestId("test-concurrent-" + i)
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                .outputConfig(MultimodalRequest.OutputConfig.builder()
                    .outputModalities(ModalityType.EMBEDDING)
                    .build())
                .build())
            .map(request -> processor.process(request))
            .collect(java.util.stream.Collectors.toList());

        // Execute all and validate
        for (int i = 0; i < requests.size(); i++) {
            MultimodalResponse response = requests.get(i).await().atMost(java.time.Duration.ofSeconds(10));
            assertThat(response).isNotNull();
            assertThat(response.getRequestId()).isEqualTo("test-concurrent-" + i);
            System.out.println("Request " + i + " completed in " + response.getDurationMs() + "ms");
        }
    }

    // Helper methods
    
    private byte[] loadTestImage(String filename) throws Exception {
        Path testImagePath = Path.of("src/test/resources/images", filename);
        if (Files.exists(testImagePath)) {
            return Files.readAllBytes(testImagePath);
        }
        return createPlaceholderImage();
    }

    private byte[] createPlaceholderImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }

    private float[] getEmbedding(byte[] imageBytes) throws Exception {
        MultimodalRequest request = MultimodalRequest.builder()
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(imageBytes, "image/jpeg"))
            .outputConfig(MultimodalRequest.OutputConfig.builder()
                .outputModalities(ModalityType.EMBEDDING)
                .build())
            .build();
        
        MultimodalResponse response = processor.process(request)
            .await().atMost(java.time.Duration.ofSeconds(10));
        
        return response.getOutputs()[0].getEmbedding();
    }

    private float calculateNorm(float[] vector) {
        float sum = 0.0f;
        for (float v : vector) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    private float cosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        return dotProduct / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    private Offset<Float> within(float delta) {
        return org.assertj.core.api.Assertions.within(delta);
    }
}
