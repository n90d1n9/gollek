package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable mini-batch for mixed-modality trainer pipelines.
 *
 * <p>The batch keeps each sample's parts grouped together while exposing
 * modality summaries that trainer/model adapters can use for routing image,
 * text, audio, document, embedding, or time-series towers.</p>
 */
public record MultimodalBatch(
        List<List<MultimodalContent>> samples,
        Map<ModalityType, Integer> modalityCounts) {
    public MultimodalBatch(List<List<MultimodalContent>> samples) {
        this(samples, Map.of());
    }

    public MultimodalBatch {
        Objects.requireNonNull(samples, "samples must not be null");
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }
        List<List<MultimodalContent>> copy = new ArrayList<>(samples.size());
        for (int i = 0; i < samples.size(); i++) {
            copy.add(MultimodalDatasetSupport.immutableSample(samples.get(i), "batch sample " + i));
        }
        samples = Collections.unmodifiableList(copy);
        modalityCounts = MultimodalDatasetSupport.modalityCounts(samples);
    }

    public int sampleCount() {
        return samples.size();
    }

    public int partCount() {
        return modalityCounts.values().stream().mapToInt(Integer::intValue).sum();
    }

    public boolean contains(ModalityType modality) {
        return modalityCounts.containsKey(Objects.requireNonNull(modality, "modality must not be null"));
    }

    public boolean sampleContains(int sampleIndex, ModalityType modality) {
        Objects.requireNonNull(modality, "modality must not be null");
        return sampleModalities(sampleIndex).contains(modality);
    }

    public Set<ModalityType> sampleModalities(int sampleIndex) {
        return MultimodalDatasetSupport.modalitySet(samples.get(sampleIndex));
    }
}
