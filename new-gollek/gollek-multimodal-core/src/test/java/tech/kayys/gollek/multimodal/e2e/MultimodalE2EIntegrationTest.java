package tech.kayys.gollek.multimodal.e2e;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End-to-end tests for multimodal inference pipeline.
 */
class MultimodalE2EIntegrationTest {

    private MultimodalInferenceService service;

    @BeforeEach
    void setUp() {
        service = new MultimodalInferenceService();
        service.initialize();
    }

    @Test
    void testVisionAssistantWorkflow() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "llava-13b.gguf")),
                "LLaVA model not found, skipping test");

        byte[] testImage = loadTestImage("test-office.jpg");

        // Simulate vision assistant workflow
        String[] questions = {
                "What objects are in this image?",
                "What color is the wall?",
                "Is there a computer visible?"
        };

        for (String question : questions) {
            MultimodalRequest request = MultimodalRequest.builder()
                    .model("llava-13b-gguf")
                    .inputs(
                            MultimodalContent.ofText(question),
                            MultimodalContent.ofBase64Image(testImage, "image/jpeg"))
                    .build();

            Uni<MultimodalResponse> responseUni = service.infer(request);
            MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(30));

            assertThat(response).isNotNull();
            assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
            assertThat(response.getOutputs()[0].getText()).isNotBlank();

            System.out.println("Q: " + question);
            System.out.println("A: " + response.getOutputs()[0].getText());
            System.out.println("---");
        }
    }

    @Test
    void testImageSearchWorkflow() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "clip-vit-base.onnx")),
                "CLIP model not found, skipping test");

        // Index multiple images
        byte[] image1 = loadTestImage("test-dog.jpg");
        byte[] image2 = loadTestImage("test-cat.jpg");
        byte[] image3 = loadTestImage("test-car.jpg");

        float[] embedding1 = getEmbedding(image1);
        float[] embedding2 = getEmbedding(image2);
        float[] embedding3 = getEmbedding(image3);

        // Query with a new image
        byte[] queryImage = loadTestImage("test-dog2.jpg");
        float[] queryEmbedding = getEmbedding(queryImage);

        // Find most similar
        float similarity1 = cosineSimilarity(queryEmbedding, embedding1);
        float similarity2 = cosineSimilarity(queryEmbedding, embedding2);
        float similarity3 = cosineSimilarity(queryEmbedding, embedding3);

        System.out.println("Similarity with image1 (dog): " + similarity1);
        System.out.println("Similarity with image2 (cat): " + similarity2);
        System.out.println("Similarity with image3 (car): " + similarity3);

        // Query image (dog) should be most similar to image1 (dog)
        assertThat(similarity1).isGreaterThan(similarity2);
        assertThat(similarity1).isGreaterThan(similarity3);
    }

    @Test
    void testDocumentQaWorkflow() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "claude-3-onnx")),
                "Claude model not found, skipping test");

        // Create a simple test document
        byte[] docBytes = "This is a test document about machine learning.".getBytes();

        MultimodalRequest request = MultimodalRequest.builder()
                .model("claude-3-onnx")
                .inputs(
                        MultimodalContent.ofText("What is this document about?"),
                        MultimodalContent.ofDocument(docBytes, "txt", "text/plain"))
                .build();

        Uni<MultimodalResponse> responseUni = service.infer(request);
        MultimodalResponse response = responseUni.await().atMost(java.time.Duration.ofSeconds(30));

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        assertThat(response.getOutputs()[0].getText()).containsIgnoringCase("machine learning");

        System.out.println("Answer: " + response.getOutputs()[0].getText());
    }

    @Test
    void testBatchProcessingWorkflow() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(Path.of(System.getProperty("user.home"), ".gollek", "models", "clip-vit-base.onnx")),
                "CLIP model not found, skipping test");

        // Create batch of 50 images
        java.util.List<MultimodalRequest> requests = java.util.stream.IntStream.range(0, 50)
                .mapToObj(i -> {
                    try {
                        byte[] image = loadTestImage("test-image" + (i % 3) + ".jpg");
                        return MultimodalRequest.builder()
                                .model("clip-vit-base")
                                .inputs(MultimodalContent.ofBase64Image(image, "image/jpeg"))
                                .outputConfig(MultimodalRequest.OutputConfig.builder()
                                        .outputModalities(ModalityType.EMBEDDING)
                                        .build())
                                .build();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                })
                .collect(java.util.stream.Collectors.toList());

        long startTime = System.currentTimeMillis();

        // Process all requests
        java.util.List<MultimodalResponse> responses = requests.stream()
                .map(request -> service.infer(request).await().atMost(java.time.Duration.ofSeconds(10)))
                .collect(java.util.stream.Collectors.toList());

        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;

        // Validate all responses
        assertThat(responses).hasSize(50);
        for (MultimodalResponse response : responses) {
            assertThat(response.getStatus()).isEqualTo(MultimodalResponse.ResponseStatus.SUCCESS);
        }

        System.out.println("Processed 50 images in " + totalTime + "ms");
        System.out.println("Average: " + (totalTime / 50.0) + "ms per image");
    }

    @Test
    void testErrorHandling() {
        // Test with invalid model
        MultimodalRequest request = MultimodalRequest.builder()
                .model("nonexistent-model")
                .inputs(MultimodalContent.ofText("Test"))
                .build();

        Uni<MultimodalResponse> responseUni = service.infer(request);

        // Should fail gracefully
        try {
            responseUni.await().atMost(java.time.Duration.ofSeconds(5));
        } catch (Exception e) {
            // Expected
            System.out.println("Error handled correctly: " + e.getMessage());
        }
    }

    @Test
    void testServiceInitialization() {
        assertThat(service).isNotNull();
        assertThat(service.getAvailableProcessors()).isNotEmpty();
        System.out.println("Available processors: " + service.getAvailableProcessors());
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

        MultimodalResponse response = service.infer(request)
                .await().atMost(java.time.Duration.ofSeconds(10));

        return response.getOutputs()[0].getEmbedding();
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
}
