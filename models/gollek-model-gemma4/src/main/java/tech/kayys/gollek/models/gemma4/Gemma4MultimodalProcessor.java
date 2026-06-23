package tech.kayys.gollek.models.gemma4;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.jboss.logging.Logger;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.processor.MultimodalProcessor;

@ApplicationScoped
public class Gemma4MultimodalProcessor implements MultimodalProcessor {

    private static final Logger LOG = Logger.getLogger(Gemma4MultimodalProcessor.class);
    public static final String PROCESSOR_ID = "gemma4-multimodal";

    // Cache for identical images
    private final Map<String, AccelTensor> embeddingCache = new ConcurrentHashMap<>();

    @Override
    public String getProcessorId() {
        return PROCESSOR_ID;
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public Uni<MultimodalResponse> process(MultimodalRequest request) {
        return Uni.createFrom().deferred(() -> {
            try {
                StringBuilder combinedPrompt = new StringBuilder();
                Map<String, Object> parameters = new HashMap<>();

                for (MultimodalContent input : request.getInputs()) {
                    if (input.getModality() == ModalityType.TEXT) {
                        combinedPrompt.append(input.getText());
                    } else if (input.getModality() == ModalityType.IMAGE) {
                        String b64 = input.getBase64Data();
                        if (b64 != null) {
                            AccelTensor imageEmbeddings = processImage(b64);
                            String imageKey = "image_" + System.currentTimeMillis();
                            parameters.put(imageKey, imageEmbeddings); // Pass tensor directly
                            combinedPrompt.append("[IMAGE:").append(imageKey).append("]");
                        }
                    }
                }

                // Actually invoke safetensor engine here or return as prepared request.
                // For now, this is intercepted by SafetensorRunnerPlugin or similar.
                MultimodalContent output = MultimodalContent.ofText("Processed " + combinedPrompt.toString());
                
                MultimodalResponse response = MultimodalResponse.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .outputs(output)
                        .metadata(parameters)
                        .build();

                return Uni.createFrom().item(response);
            } catch (Exception e) {
                return Uni.createFrom().failure(e);
            }
        });
    }

    @Override
    public Multi<MultimodalResponse> processStream(MultimodalRequest request) {
        return Multi.createFrom().failure(new UnsupportedOperationException("Streaming not yet implemented for Gemma4MultimodalProcessor"));
    }

    /**
     * Processes base64 image into vision embeddings tensor.
     * Caches embeddings if the same image is seen again.
     */
    private AccelTensor processImage(String base64Data) throws Exception {
        // Simple hash/checksum for caching identical images
        String cacheKey = String.valueOf(base64Data.hashCode());
        if (embeddingCache.containsKey(cacheKey)) {
            LOG.infof("Using cached vision embeddings for image %s", cacheKey);
            return embeddingCache.get(cacheKey);
        }

        byte[] imageBytes = Base64.getDecoder().decode(base64Data);
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));

        if (img == null) {
            throw new IllegalArgumentException("Invalid image data");
        }

        // Resize & Normalize
        // Assuming dynamic resize up to a reasonable square bounds for SigLIP/ViT
        // SigLIP typically uses 224, 384, etc. Gemma 4 Vision configuration states patch_size = 16.
        int targetSize = 224; // Placeholder
        BufferedImage resized = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img.getScaledInstance(targetSize, targetSize, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        // Convert to tensor (mocked, as real implementation needs full Gemma4VisionTower)
        AccelTensor mockEmbeddings = AccelTensor.zeros(new long[]{1, 280, 768});
        
        embeddingCache.put(cacheKey, mockEmbeddings);
        return mockEmbeddings;
    }
}
