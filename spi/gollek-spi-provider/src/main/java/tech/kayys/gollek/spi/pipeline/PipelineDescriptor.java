package tech.kayys.gollek.spi.pipeline;

import tech.kayys.gollek.spi.model.ModalityType;

import java.util.List;
import java.util.Map;

/**
 * Stable metadata for a feature-level pipeline.
 *
 * <p>Pipelines are higher-level orchestration adapters that sit above a backend
 * provider. A backend usually knows how to execute one graph or model format;
 * a pipeline knows how to compose preprocessing, multiple backend sessions,
 * decoding, and postprocessing for one capability such as OCR, TTS, ASR, or
 * image generation.</p>
 */
public record PipelineDescriptor(
        String id,
        String name,
        String version,
        String family,
        List<ModalityType> inputModalities,
        List<ModalityType> outputModalities,
        List<String> tags,
        Map<String, Object> metadata) {

    public PipelineDescriptor {
        inputModalities = inputModalities == null ? List.of() : List.copyOf(inputModalities);
        outputModalities = outputModalities == null ? List.of() : List.copyOf(outputModalities);
        tags = tags == null ? List.of() : List.copyOf(tags);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
