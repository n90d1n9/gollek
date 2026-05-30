package tech.kayys.gollek.spi.pipeline;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModalityType;

import java.time.Instant;

/**
 * Extension point for model-specific feature pipelines.
 *
 * <p>Implement this interface in an external feature project when a capability
 * needs orchestration beyond a single backend call. Examples include OCR/VL
 * pipelines, text-to-speech pipelines, ASR pipelines, diffusion pipelines, or
 * benchmark/evaluation pipelines.</p>
 *
 * <p>Feature jars can be discovered as CDI beans or by Java {@link java.util.ServiceLoader}.
 * For ServiceLoader discovery, add this file to the feature jar:</p>
 *
 * <pre>
 * META-INF/services/tech.kayys.gollek.spi.pipeline.ModelPipeline
 * </pre>
 */
public interface ModelPipeline {

    /**
     * Stable pipeline id, such as {@code paddleocr-vl}.
     */
    String id();

    /**
     * Human-readable metadata for user interfaces and diagnostics.
     */
    PipelineDescriptor descriptor();

    /**
     * Larger numbers win when more than one pipeline supports the same request.
     */
    default int priority() {
        return 0;
    }

    /**
     * Return true when this pipeline can handle the inspected model/request.
     */
    boolean supports(ModelPipelineRequest request);

    /**
     * Execute the complete pipeline.
     */
    Uni<InferenceResponse> infer(ModelPipelineRequest request);

    /**
     * Execute the pipeline as a stream. The default wraps the non-streaming result
     * into one content chunk and one final chunk.
     */
    default Multi<StreamingInferenceChunk> inferStream(ModelPipelineRequest request) {
        return infer(request).onItem().transformToMulti(response -> Multi.createFrom().items(
                new StreamingInferenceChunk(
                        request.request().getRequestId(),
                        0,
                        ModalityType.TEXT,
                        response.getContent(),
                        null,
                        false,
                        null,
                        null,
                        Instant.now(),
                        response.getMetadata()),
                new StreamingInferenceChunk(
                        request.request().getRequestId(),
                        1,
                        ModalityType.TEXT,
                        "",
                        null,
                        true,
                        response.getFinishReason().name().toLowerCase(),
                        new StreamingInferenceChunk.ChunkUsage(
                                response.getInputTokens(),
                                response.getOutputTokens(),
                                response.getDurationMs()),
                        Instant.now(),
                        response.getMetadata())));
    }
}
