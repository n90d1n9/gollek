package tech.kayys.gollek.spi.model;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Lightweight catalogue metadata for a model-family extension.
 */
public record ModelFamilyDescriptor(
        String id,
        String displayName,
        List<String> modelTypes,
        List<String> architectureClassNames,
        List<ModelFamilyCapability> capabilities,
        Map<String, String> metadata) {

    public ModelFamilyDescriptor {
        id = requiredNormalized(id, "model family id");
        displayName = displayName == null || displayName.isBlank() ? id : displayName.trim();
        modelTypes = normalizedDistinct(modelTypes);
        architectureClassNames = trimmedDistinct(architectureClassNames);
        capabilities = capabilities == null ? List.of() : List.copyOf(new LinkedHashSet<>(capabilities));
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public boolean supportsDirectSafetensorInference() {
        return capabilities.contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE);
    }

    private static String requiredNormalized(String value, String fieldName) {
        String normalized = normalize(value);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return normalized;
    }

    private static List<String> normalizedDistinct(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        Set<String> normalized = new LinkedHashSet<>();
        for (String value : values) {
            String item = normalize(value);
            if (!item.isBlank()) {
                normalized.add(item);
            }
        }
        return List.copyOf(normalized);
    }

    private static List<String> trimmedDistinct(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        Set<String> trimmed = new LinkedHashSet<>();
        for (String value : values) {
            String item = Objects.toString(value, "").trim();
            if (!item.isBlank()) {
                trimmed.add(item);
            }
        }
        return List.copyOf(trimmed);
    }

    private static String normalize(String value) {
        return Objects.toString(value, "").trim().toLowerCase(Locale.ROOT);
    }
}
