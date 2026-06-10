package tech.kayys.gollek.spi.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Derived runtime support summary for diagnostics and runner gating.
 */
public record ModelFamilySupportReport(
        String id,
        String displayName,
        List<String> modelTypes,
        List<String> architectureClassNames,
        List<String> architectureAdapterIds,
        List<String> tokenizerProfileIds,
        List<ModelTokenizerKind> tokenizerKinds,
        ModelFamilyBundleProfile bundleProfile,
        List<ModelFamilyCapability> capabilities,
        ModelFamilyDirectSupport directSafetensorStatus,
        String directSafetensorReason,
        Map<String, String> directSafetensorCaveats,
        Map<String, String> metadata) {

    public ModelFamilySupportReport {
        modelTypes = modelTypes == null ? List.of() : List.copyOf(modelTypes);
        architectureClassNames = architectureClassNames == null ? List.of() : List.copyOf(architectureClassNames);
        architectureAdapterIds = architectureAdapterIds == null ? List.of() : List.copyOf(architectureAdapterIds);
        tokenizerProfileIds = tokenizerProfileIds == null ? List.of() : List.copyOf(tokenizerProfileIds);
        tokenizerKinds = tokenizerKinds == null ? List.of() : List.copyOf(tokenizerKinds);
        bundleProfile = bundleProfile == null ? ModelFamilyBundleProfile.OPTIONAL : bundleProfile;
        capabilities = capabilities == null ? List.of() : List.copyOf(capabilities);
        directSafetensorStatus = directSafetensorStatus == null
                ? ModelFamilyDirectSupport.NOT_ADVERTISED
                : directSafetensorStatus;
        directSafetensorReason = directSafetensorReason == null || directSafetensorReason.isBlank()
                ? directSafetensorStatus.label()
                : directSafetensorReason;
        directSafetensorCaveats = sortedCopy(directSafetensorCaveats);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public static ModelFamilySupportReport from(ModelFamilyPlugin plugin) {
        return from((ModelFamilyRuntimeAdapter) plugin);
    }

    public static ModelFamilySupportReport from(ModelFamilyRuntimeAdapter adapter) {
        ModelFamilyDescriptor descriptor = adapter.descriptor();
        List<ModelArchitecture> adapters = safeAdapters(adapter);
        List<ModelTokenizerDescriptor> tokenizers = safeTokenizers(adapter);
        List<String> adapterIds = adapters.stream()
                .map(ModelFamilySupportReport::safeAdapterId)
                .filter(id -> !id.isBlank())
                .distinct()
                .toList();
        List<String> tokenizerProfileIds = tokenizers.stream()
                .map(ModelFamilySupportReport::safeTokenizerId)
                .filter(id -> !id.isBlank())
                .distinct()
                .toList();
        List<ModelTokenizerKind> tokenizerKinds = tokenizers.stream()
                .map(ModelFamilySupportReport::safeTokenizerKind)
                .filter(kind -> kind != null)
                .distinct()
                .toList();

        DirectStatus direct = directStatus(descriptor, adapterIds);
        return new ModelFamilySupportReport(
                descriptor.id(),
                descriptor.displayName(),
                descriptor.modelTypes(),
                descriptor.architectureClassNames(),
                adapterIds,
                tokenizerProfileIds,
                tokenizerKinds,
                ModelFamilyBundleProfile.fromMetadata(descriptor.metadata()),
                descriptor.capabilities(),
                direct.status(),
                direct.reason(),
                directCaveats(descriptor.metadata()),
                descriptor.metadata());
    }

    public boolean directSafetensorReady() {
        return directSafetensorStatus.ready();
    }

    public String directSafetensorLabel() {
        return directSafetensorStatus.label();
    }

    public String shortDirectSafetensorSummary() {
        String summary = directSafetensorLabel() + ":" + directSafetensorReason;
        if (directSafetensorCaveats.isEmpty()) {
            return summary;
        }
        return summary + ";caveats=" + shortDirectSafetensorCaveats();
    }

    public String shortDirectSafetensorCaveats() {
        if (directSafetensorCaveats.isEmpty()) {
            return "";
        }
        return directSafetensorCaveats.entrySet().stream()
                .map(entry -> entry.getKey() + ":" + entry.getValue())
                .reduce((a, b) -> a + "," + b)
                .orElse("");
    }

    public boolean defaultBundle() {
        return bundleProfile.defaultBundle();
    }

    private static DirectStatus directStatus(ModelFamilyDescriptor descriptor, List<String> adapterIds) {
        String reason = descriptor.metadata().get("direct_safetensor");
        ModelFamilyDirectSupport explicitStatus = statusFromReason(reason);
        if (explicitStatus == ModelFamilyDirectSupport.EXPERIMENTAL
                || explicitStatus == ModelFamilyDirectSupport.PENDING
                || explicitStatus == ModelFamilyDirectSupport.NOT_APPLICABLE) {
            return new DirectStatus(explicitStatus, reason);
        }
        if (descriptor.supportsDirectSafetensorInference() && !adapterIds.isEmpty()) {
            return new DirectStatus(ModelFamilyDirectSupport.READY,
                    reason == null || reason.isBlank() ? "ready" : reason);
        }
        if (descriptor.supportsDirectSafetensorInference()) {
            return new DirectStatus(ModelFamilyDirectSupport.DECLARED_NO_ADAPTER,
                    "direct_safetensor capability has no architecture adapter");
        }
        if (reason != null && !reason.isBlank() && explicitStatus != ModelFamilyDirectSupport.NOT_ADVERTISED) {
            return new DirectStatus(explicitStatus, reason);
        }
        if (reason != null && !reason.isBlank()) {
            return new DirectStatus(ModelFamilyDirectSupport.NOT_ADVERTISED, reason);
        }
        return new DirectStatus(ModelFamilyDirectSupport.NOT_ADVERTISED, "not_advertised");
    }

    private static ModelFamilyDirectSupport statusFromReason(String reason) {
        if (reason == null || reason.isBlank()) {
            return ModelFamilyDirectSupport.NOT_ADVERTISED;
        }
        String normalized = reason.toLowerCase(Locale.ROOT);
        if (normalized.startsWith("experimental")) {
            return ModelFamilyDirectSupport.EXPERIMENTAL;
        }
        if (normalized.startsWith("not")) {
            return ModelFamilyDirectSupport.NOT_APPLICABLE;
        }
        if (normalized.startsWith("pending") || normalized.endsWith("_pending")
                || normalized.contains("_pending_")) {
            return ModelFamilyDirectSupport.PENDING;
        }
        return ModelFamilyDirectSupport.NOT_ADVERTISED;
    }

    private static Map<String, String> directCaveats(Map<String, String> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return Map.of();
        }
        Map<String, String> caveats = new TreeMap<>();
        metadata.forEach((key, value) -> {
            if (key != null && value != null
                    && key.endsWith("_direct_safetensor")
                    && !"direct_safetensor".equals(key)
                    && !value.isBlank()) {
                caveats.putIfAbsent(key.substring(0, key.length() - "_direct_safetensor".length()), value);
            }
        });
        return sortedCopy(caveats);
    }

    private static Map<String, String> sortedCopy(Map<String, String> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(new TreeMap<>(values)));
    }

    private static List<ModelArchitecture> safeAdapters(ModelFamilyRuntimeAdapter adapter) {
        try {
            List<ModelArchitecture> adapters = adapter.architectureAdapters();
            return adapters == null ? List.of() : adapters.stream()
                    .filter(candidate -> candidate != null)
                    .toList();
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<ModelTokenizerDescriptor> safeTokenizers(ModelFamilyRuntimeAdapter adapter) {
        try {
            List<ModelTokenizerDescriptor> tokenizers = adapter.tokenizerDescriptors();
            return tokenizers == null ? List.of() : tokenizers.stream()
                    .filter(tokenizer -> tokenizer != null)
                    .toList();
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static String safeAdapterId(ModelArchitecture adapter) {
        try {
            String id = adapter.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
        }
    }

    private static String safeTokenizerId(ModelTokenizerDescriptor tokenizer) {
        try {
            String id = tokenizer.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
        }
    }

    private static ModelTokenizerKind safeTokenizerKind(ModelTokenizerDescriptor tokenizer) {
        try {
            return tokenizer.kind();
        } catch (RuntimeException error) {
            return null;
        }
    }

    private record DirectStatus(ModelFamilyDirectSupport status, String reason) {
    }
}
