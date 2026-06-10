package tech.kayys.gollek.spi.model;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime requirement declared by a model-family plugin for unified multimodal execution.
 */
public record ModelFamilyUnifiedRuntimeRequirement(
        String modelType,
        List<String> requiredInputModalities,
        boolean productionReadyRequired,
        String reason,
        Map<String, String> metadata) {

    public ModelFamilyUnifiedRuntimeRequirement {
        modelType = normalize(modelType);
        if (modelType.isBlank()) {
            modelType = "unknown";
        }
        requiredInputModalities = normalizeDistinct(requiredInputModalities);
        reason = reason == null || reason.isBlank() ? "unified runtime required" : reason.trim();
        metadata = copyMetadata(metadata);
    }

    public static ModelFamilyUnifiedRuntimeRequirement productionReady(
            String modelType,
            List<String> requiredInputModalities,
            String reason) {
        return new ModelFamilyUnifiedRuntimeRequirement(
                modelType,
                requiredInputModalities,
                true,
                reason,
                Map.of());
    }

    public boolean requiresInputModality(String modality) {
        String normalized = normalize(modality);
        return !normalized.isBlank() && requiredInputModalities.contains(normalized);
    }

    private static List<String> normalizeDistinct(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<String> normalized = new LinkedHashSet<>();
        for (String value : values) {
            String item = normalize(value);
            if (!item.isBlank()) {
                normalized.add(item);
            }
        }
        return List.copyOf(normalized);
    }

    private static Map<String, String> copyMetadata(Map<String, String> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, String> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            String normalizedKey = normalize(key);
            if (!normalizedKey.isBlank() && value != null) {
                copy.put(normalizedKey, value.trim());
            }
        });
        return Map.copyOf(copy);
    }

    private static String normalize(String value) {
        return Objects.toString(value, "").trim().toLowerCase(Locale.ROOT);
    }
}
