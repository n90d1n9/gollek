package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

final class MultimodalDatasetSupport {
    private MultimodalDatasetSupport() {
    }

    static List<MultimodalContent> immutableSample(List<MultimodalContent> sample, String owner) {
        Objects.requireNonNull(sample, owner + " must not be null");
        if (sample.isEmpty()) {
            throw new IllegalArgumentException(owner + " must contain at least one multimodal part");
        }
        List<MultimodalContent> copy = new ArrayList<>(sample.size());
        for (int i = 0; i < sample.size(); i++) {
            MultimodalContent content = Objects.requireNonNull(
                    sample.get(i),
                    owner + " part " + i + " must not be null");
            if (content.getModality() == null) {
                throw new IllegalArgumentException(owner + " part " + i + " must declare a modality");
            }
            copy.add(content);
        }
        return Collections.unmodifiableList(copy);
    }

    static Set<ModalityType> modalitySet(List<MultimodalContent> sample) {
        Set<ModalityType> modalities = new LinkedHashSet<>();
        for (MultimodalContent content : sample) {
            modalities.add(content.getModality());
        }
        return Collections.unmodifiableSet(modalities);
    }

    static Map<ModalityType, Integer> modalityCounts(List<List<MultimodalContent>> samples) {
        EnumMap<ModalityType, Integer> counts = new EnumMap<>(ModalityType.class);
        for (List<MultimodalContent> sample : samples) {
            for (MultimodalContent content : sample) {
                counts.merge(content.getModality(), 1, Integer::sum);
            }
        }
        return Collections.unmodifiableMap(counts);
    }
}
