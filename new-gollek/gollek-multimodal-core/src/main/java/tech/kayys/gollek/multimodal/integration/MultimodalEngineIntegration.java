package tech.kayys.gollek.multimodal.integration;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.ModalityType;

import java.util.List;

/**
 * Integrates multimodal inference with the Gollek inference engine.
 * Converts between standard inference requests and multimodal requests.
 */
@ApplicationScoped
public class MultimodalEngineIntegration {

    private static final Logger log = Logger.getLogger(MultimodalEngineIntegration.class);

    @Inject
    MultimodalInferenceService multimodalService;

    /**
     * Convert a standard inference request to a multimodal request.
     *
     * @param request the standard inference request
     * @return multimodal request
     */
    public MultimodalRequest toMultimodalRequest(InferenceRequest request) {
        MultimodalRequest.Builder builder = MultimodalRequest.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .tenantId(request.getUserId().orElse(null))
                .timeoutMs(request.getTimeout().map(java.time.Duration::toMillis).orElse(30000L));

        // Convert prompt to text content
        MultimodalContent textContent = MultimodalContent.ofText(
                request.getMessages().stream()
                        .map(m -> m.getContent())
                        .reduce("", (a, b) -> a + "\n" + b));

        // Check for image attachments in parameters
        List<MultimodalContent> contents = new java.util.ArrayList<>();
        contents.add(textContent);

        // Add images if present
        if (request.getParameters().containsKey("images")) {
            @SuppressWarnings("unchecked")
            List<String> imageUrls = (List<String>) request.getParameters().get("images");
            for (String imageUrl : imageUrls) {
                if (imageUrl.startsWith("http")) {
                    contents.add(MultimodalContent.ofImageUri(imageUrl, "image/jpeg"));
                } else if (imageUrl.startsWith("data:")) {
                    // Parse data URI
                    String base64Data = imageUrl.substring(imageUrl.indexOf(",") + 1);
                    String mimeType = imageUrl.substring(5, imageUrl.indexOf(";"));
                    contents.add(MultimodalContent.builder(ModalityType.IMAGE)
                            .base64Data(base64Data)
                            .mimeType(mimeType)
                            .build());
                }
            }
        }

        builder.inputs(contents.toArray(new MultimodalContent[0]));

        // Convert parameters
        MultimodalRequest.OutputConfig.Builder configBuilder = MultimodalRequest.OutputConfig.builder();

        if (request.getParameters().containsKey("max_tokens")) {
            configBuilder.maxTokens((Integer) request.getParameters().get("max_tokens"));
        }
        if (request.getParameters().containsKey("temperature")) {
            configBuilder.temperature(((Number) request.getParameters().get("temperature")).doubleValue());
        }
        if (request.getParameters().containsKey("top_p")) {
            configBuilder.topP(((Number) request.getParameters().get("top_p")).doubleValue());
        }
        if (request.getParameters().containsKey("stream")) {
            configBuilder.stream((Boolean) request.getParameters().get("stream"));
        }

        builder.outputConfig(configBuilder.build());

        return builder.build();
    }

    /**
     * Convert a multimodal response to a standard inference response.
     *
     * @param multimodalResponse the multimodal response
     * @param request            the original request
     * @return inference response
     */
    public InferenceResponse toInferenceResponse(
            MultimodalResponse multimodalResponse,
            InferenceRequest request) {

        // Extract text from multimodal output
        StringBuilder content = new StringBuilder();
        for (MultimodalContent output : multimodalResponse.getOutputs()) {
            if (output.getModality() == ModalityType.TEXT) {
                content.append(output.getText());
            }
        }

        InferenceResponse.Builder builder = InferenceResponse.builder()
                .requestId(multimodalResponse.getRequestId())
                .content(content.toString())
                .model(multimodalResponse.getModel())
                .durationMs(multimodalResponse.getDurationMs())
                .metadata("multimodal", "true")
                .metadata("processor", multimodalResponse.getMetadata().getOrDefault("processor", "unknown"));

        // Add usage if available
        if (multimodalResponse.getUsage() != null) {
            builder.inputTokens(multimodalResponse.getUsage().getInputTokens())
                    .outputTokens(multimodalResponse.getUsage().getOutputTokens())
                    .tokensUsed(multimodalResponse.getUsage().getTotalTokens());
        }

        return builder.build();
    }

    /**
     * Execute multimodal inference from a standard request.
     *
     * @param request the standard inference request
     * @return Uni containing inference response
     */
    public Uni<InferenceResponse> executeMultimodal(InferenceRequest request) {
        log.infof("Executing multimodal inference for request: %s", request.getRequestId());

        // Convert to multimodal request
        MultimodalRequest multimodalRequest = toMultimodalRequest(request);

        // Execute multimodal inference
        return multimodalService.infer(multimodalRequest)
                .onItem().transform(response -> toInferenceResponse(response, request))
                .onFailure().recoverWithItem(error -> {
                    log.errorf("Multimodal inference failed: %s", error.getMessage());
                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content("Error: " + error.getMessage())
                            .model(request.getModel())
                            .metadata("error", "true")
                            .build();
                });
    }

    /**
     * Execute streaming multimodal inference.
     *
     * @param request the standard inference request
     * @return Multi containing streaming chunks
     */
    public Multi<StreamingInferenceChunk> executeMultimodalStream(InferenceRequest request) {
        log.infof("Executing streaming multimodal inference for request: %s", request.getRequestId());

        MultimodalRequest multimodalRequest = toMultimodalRequest(request);

        return multimodalService.inferStream(multimodalRequest)
                .onItem().transformToMultiAndConcatenate(response -> {
                    String content = "";
                    for (MultimodalContent output : response.getOutputs()) {
                        if (output.getModality() == ModalityType.TEXT) {
                            content = output.getText();
                        }
                    }

                    boolean isFinal = response.getStatus() != MultimodalResponse.ResponseStatus.SUCCESS;
                    if (isFinal) {
                        return Multi.createFrom().item(StreamingInferenceChunk.finalChunk(
                                response.getRequestId(), 0, content));
                    } else {
                        return Multi.createFrom().item(StreamingInferenceChunk.of(
                                response.getRequestId(), 0, content));
                    }
                });
    }

    /**
     * Check if multimodal inference is available for the given model.
     *
     * @param model the model name
     * @return true if available
     */
    public boolean isMultimodalAvailable(String model) {
        return multimodalService.getAvailableProcessors().stream()
                .anyMatch(id -> id.toLowerCase().contains(model.toLowerCase()) ||
                        model.toLowerCase().contains(id.toLowerCase()));
    }
}
