package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Dataset-level quality checks for multimodal training corpora.
 */
public final class MultimodalDatasetDiagnostics {
    private MultimodalDatasetDiagnostics() {
    }

    public static MultimodalDatasetReport inspect(Dataset<? extends List<MultimodalContent>> dataset) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        List<List<MultimodalContent>> samples = new ArrayList<>(dataset.size());
        for (int i = 0; i < dataset.size(); i++) {
            samples.add(dataset.get(i));
        }
        return inspect(samples);
    }

    public static MultimodalDatasetReport inspect(List<? extends List<MultimodalContent>> samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }

        EnumMap<ModalityType, Integer> partCounts = new EnumMap<>(ModalityType.class);
        EnumMap<ModalityType, Integer> sampleCounts = new EnumMap<>(ModalityType.class);
        Map<String, Integer> signatureCounts = new LinkedHashMap<>();
        Map<String, Integer> mimeTypeCounts = new LinkedHashMap<>();
        Map<String, Integer> sourcePathCounts = new LinkedHashMap<>();
        List<Integer> samplesWithBlankText = new ArrayList<>();
        List<Integer> samplesWithoutText = new ArrayList<>();
        List<Integer> samplesWithoutNonTextAsset = new ArrayList<>();

        int partCount = 0;
        int textPartCount = 0;
        int blankTextPartCount = 0;
        int remoteAssetCount = 0;
        int inlinedAssetCount = 0;
        int unresolvedAssetCount = 0;
        int sourcePathCount = 0;

        for (int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            List<MultimodalContent> sample = MultimodalDatasetSupport.immutableSample(
                    samples.get(sampleIndex),
                    "sample " + sampleIndex);
            Set<ModalityType> sampleModalities = new LinkedHashSet<>();
            boolean hasText = false;
            boolean hasBlankText = false;
            boolean hasNonTextAsset = false;

            for (MultimodalContent content : sample) {
                ModalityType modality = content.getModality();
                partCount++;
                partCounts.merge(modality, 1, Integer::sum);
                sampleModalities.add(modality);

                if (modality == ModalityType.TEXT) {
                    textPartCount++;
                    hasText = true;
                    String text = content.getText();
                    if (text == null || text.isBlank()) {
                        blankTextPartCount++;
                        hasBlankText = true;
                    }
                } else {
                    hasNonTextAsset = true;
                    String mimeType = normalizedMimeType(content.getMimeType());
                    mimeTypeCounts.merge(mimeType, 1, Integer::sum);
                    if (content.isRemote()) {
                        remoteAssetCount++;
                    }
                    if (hasInlinePayload(content)) {
                        inlinedAssetCount++;
                    }
                    if (!content.isRemote() && !hasInlinePayload(content)) {
                        unresolvedAssetCount++;
                    }
                    String sourcePath = sourcePath(content);
                    if (sourcePath != null) {
                        sourcePathCount++;
                        sourcePathCounts.merge(sourcePath, 1, Integer::sum);
                    }
                }
            }

            sampleModalities.forEach(modality -> sampleCounts.merge(modality, 1, Integer::sum));
            signatureCounts.merge(signature(sampleModalities), 1, Integer::sum);
            if (!hasText) {
                samplesWithoutText.add(sampleIndex);
            }
            if (hasBlankText) {
                samplesWithBlankText.add(sampleIndex);
            }
            if (!hasNonTextAsset) {
                samplesWithoutNonTextAsset.add(sampleIndex);
            }
        }

        return new MultimodalDatasetReport(
                samples.size(),
                partCount,
                partCounts,
                sampleCounts,
                signatureCounts,
                mimeTypeCounts,
                textPartCount,
                blankTextPartCount,
                remoteAssetCount,
                inlinedAssetCount,
                unresolvedAssetCount,
                sourcePathCount,
                duplicateSourcePaths(sourcePathCounts),
                samplesWithBlankText,
                samplesWithoutText,
                samplesWithoutNonTextAsset);
    }

    private static boolean hasInlinePayload(MultimodalContent content) {
        return (content.getBase64Data() != null && !content.getBase64Data().isBlank())
                || (content.getRawBytes() != null && content.getRawBytes().length > 0);
    }

    private static String normalizedMimeType(String mimeType) {
        return mimeType == null || mimeType.isBlank()
                ? "application/octet-stream"
                : mimeType.trim().toLowerCase(Locale.ROOT);
    }

    private static String sourcePath(MultimodalContent content) {
        Map<String, Object> metadata = content.getMetadata();
        Object value = metadata == null ? null : metadata.get("sourcePath");
        if (value != null) {
            String sourcePath = value.toString().trim();
            if (!sourcePath.isEmpty()) {
                return sourcePath;
            }
        }
        return content.isRemote() ? content.getUri().trim() : null;
    }

    private static String signature(Set<ModalityType> modalities) {
        List<String> names = modalities.stream()
                .map(Enum::name)
                .sorted()
                .toList();
        return String.join("+", names);
    }

    private static List<String> duplicateSourcePaths(Map<String, Integer> sourcePathCounts) {
        List<String> duplicates = new ArrayList<>();
        sourcePathCounts.forEach((path, count) -> {
            if (count > 1) {
                duplicates.add(path);
            }
        });
        Collections.sort(duplicates);
        return duplicates;
    }
}
