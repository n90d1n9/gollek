package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable data-quality report for multimodal datasets.
 */
public record MultimodalDatasetReport(
        int sampleCount,
        int partCount,
        Map<ModalityType, Integer> partCountsByModality,
        Map<ModalityType, Integer> sampleCountsByModality,
        Map<String, Integer> sampleSignatureCounts,
        Map<String, Integer> mimeTypeCounts,
        int textPartCount,
        int blankTextPartCount,
        int remoteAssetCount,
        int inlinedAssetCount,
        int unresolvedAssetCount,
        int sourcePathCount,
        List<String> duplicateSourcePaths,
        List<Integer> samplesWithBlankText,
        List<Integer> samplesWithoutText,
        List<Integer> samplesWithoutNonTextAsset) {
    public MultimodalDatasetReport {
        if (sampleCount < 0 || partCount < 0 || textPartCount < 0 || blankTextPartCount < 0
                || remoteAssetCount < 0 || inlinedAssetCount < 0 || unresolvedAssetCount < 0
                || sourcePathCount < 0) {
            throw new IllegalArgumentException("report counts must not be negative");
        }
        partCountsByModality = immutableEnumMap(partCountsByModality, "partCountsByModality");
        sampleCountsByModality = immutableEnumMap(sampleCountsByModality, "sampleCountsByModality");
        sampleSignatureCounts = immutableStringMap(sampleSignatureCounts, "sampleSignatureCounts");
        mimeTypeCounts = immutableStringMap(mimeTypeCounts, "mimeTypeCounts");
        duplicateSourcePaths = immutableList(duplicateSourcePaths, "duplicateSourcePaths");
        samplesWithBlankText = immutableList(samplesWithBlankText, "samplesWithBlankText");
        samplesWithoutText = immutableList(samplesWithoutText, "samplesWithoutText");
        samplesWithoutNonTextAsset = immutableList(samplesWithoutNonTextAsset, "samplesWithoutNonTextAsset");
    }

    public int partCount(ModalityType modality) {
        return partCountsByModality.getOrDefault(Objects.requireNonNull(modality, "modality must not be null"), 0);
    }

    public int sampleCount(ModalityType modality) {
        return sampleCountsByModality.getOrDefault(Objects.requireNonNull(modality, "modality must not be null"), 0);
    }

    public boolean hasBlankText() {
        return blankTextPartCount > 0;
    }

    public boolean hasDuplicateSourcePaths() {
        return !duplicateSourcePaths.isEmpty();
    }

    public boolean hasUnresolvedAssets() {
        return unresolvedAssetCount > 0;
    }

    public boolean hasEverySample(ModalityType modality) {
        return sampleCount > 0 && sampleCount(modality) == sampleCount;
    }

    public boolean isReadyForTextAssetTraining(ModalityType assetModality) {
        Objects.requireNonNull(assetModality, "assetModality must not be null");
        if (assetModality == ModalityType.TEXT) {
            throw new IllegalArgumentException("assetModality must be a non-text modality");
        }
        return sampleCount > 0
                && !hasBlankText()
                && samplesWithoutText.isEmpty()
                && hasEverySample(assetModality);
    }

    private static Map<ModalityType, Integer> immutableEnumMap(Map<ModalityType, Integer> map, String name) {
        Objects.requireNonNull(map, name + " must not be null");
        EnumMap<ModalityType, Integer> copy = new EnumMap<>(ModalityType.class);
        map.forEach((key, value) -> {
            Objects.requireNonNull(key, name + " key must not be null");
            if (value == null || value < 0) {
                throw new IllegalArgumentException(name + " values must not be null or negative");
            }
            copy.put(key, value);
        });
        return Collections.unmodifiableMap(copy);
    }

    private static Map<String, Integer> immutableStringMap(Map<String, Integer> map, String name) {
        Objects.requireNonNull(map, name + " must not be null");
        Map<String, Integer> copy = new LinkedHashMap<>();
        map.forEach((key, value) -> {
            String normalized = Objects.requireNonNull(key, name + " key must not be null").trim();
            if (normalized.isEmpty()) {
                throw new IllegalArgumentException(name + " keys must not be blank");
            }
            if (value == null || value < 0) {
                throw new IllegalArgumentException(name + " values must not be null or negative");
            }
            copy.put(normalized, value);
        });
        return Collections.unmodifiableMap(copy);
    }

    private static <T> List<T> immutableList(List<T> list, String name) {
        Objects.requireNonNull(list, name + " must not be null");
        return Collections.unmodifiableList(new ArrayList<>(list));
    }
}
