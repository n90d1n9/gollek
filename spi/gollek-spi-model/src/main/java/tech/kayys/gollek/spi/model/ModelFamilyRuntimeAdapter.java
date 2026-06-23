package tech.kayys.gollek.spi.model;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime-facing surface for detachable model-family knowledge.
 *
 * <p>The adapter keeps reusable family metadata close to the plugin while still
 * letting production builds attach only the families they need.</p>
 */
public interface ModelFamilyRuntimeAdapter {

    ModelFamilyDescriptor descriptor();

    /**
     * Architecture adapters that can be used by direct runtime engines.
     */
    default List<ModelArchitecture> architectureAdapters() {
        return List.of();
    }

    /**
     * Tokenizer profiles that runtime engines can select when files are present.
     */
    default List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of();
    }

    /**
     * Chat-template identifiers this family can format.
     */
    default List<String> chatTemplateIds() {
        ModelFamilyDescriptor descriptor = descriptor();
        if (descriptor == null || !descriptor.capabilities().contains(ModelFamilyCapability.CHAT_TEMPLATE)) {
            return List.of();
        }

        List<String> configured = splitMetadata(
                descriptor.metadata().get("chat_template_ids"),
                descriptor.metadata().get("chat_template_id"),
                descriptor.metadata().get("chat_template"));
        if (!configured.isEmpty()) {
            return configured;
        }
        if (!descriptor.modelTypes().isEmpty()) {
            return descriptor.modelTypes();
        }
        return List.of(descriptor.id());
    }

    /**
     * Unified multimodal runtimes required by this family for full execution.
     */
    default List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
        ModelFamilyDescriptor descriptor = descriptor();
        if (descriptor == null) {
            return List.of();
        }
        List<String> modelTypes = splitMetadata(
                descriptor.metadata().get("unified_model_types"),
                descriptor.metadata().get("unified_model_type"));
        if (modelTypes.isEmpty()) {
            return List.of();
        }
        List<String> modalities = splitMetadata(
                descriptor.metadata().get("unified_runtime_required_modalities"),
                descriptor.metadata().get("unified_runtime_modalities"),
                descriptor.metadata().get("unified_input_modalities"));
        boolean productionReadyRequired = !"false".equals(normalize(
                descriptor.metadata().get("unified_runtime_production_required")));
        String reason = firstNonBlank(
                descriptor.metadata().get("unified_runtime_reason"),
                descriptor.metadata().get("unified_runtime"),
                "unified runtime required");
        return modelTypes.stream()
                .map(modelType -> new ModelFamilyUnifiedRuntimeRequirement(
                        modelType,
                        modalities,
                        productionReadyRequired,
                        reason,
                        Map.of("source", "descriptor_metadata")))
                .toList();
    }

    /**
     * Runtime traits are config-sensitive, so they are resolved lazily.
     */
    default ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        if (config != null) {
            String modelType = normalize(config.getModelType());
            String architectureClassName = normalize(config.getPrimaryArchitecture());
            for (ModelArchitecture adapter : safeArchitectureAdapters()) {
                if (!adapterMatches(adapter, modelType, architectureClassName)) {
                    continue;
                }
                try {
                    ModelRuntimeTraits traits = adapter.runtimeTraits(config);
                    if (traits != null) {
                        return traits;
                    }
                } catch (RuntimeException ignored) {
                    // Fall back to config-derived traits if a family adapter is not runtime-safe.
                }
            }
        }
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }

    /**
     * Derived support status for diagnostics and runtime policy gates.
     */
    default ModelFamilySupportReport supportReport() {
        return ModelFamilySupportReport.from(this);
    }

    /**
     * A compact runtime manifest that runners can inspect without knowing the
     * plugin implementation class.
     */
    default ModelFamilyRuntimeManifest runtimeManifest() {
        return ModelFamilyRuntimeManifest.from(this);
    }

    private List<ModelArchitecture> safeArchitectureAdapters() {
        try {
            List<ModelArchitecture> adapters = architectureAdapters();
            return adapters == null ? List.of() : adapters;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static boolean adapterMatches(ModelArchitecture adapter, String modelType, String architectureClassName) {
        if (adapter == null) {
            return false;
        }
        boolean modelTypeMatches = !modelType.isBlank() && safeSupportedModelTypes(adapter).stream()
                .map(ModelFamilyRuntimeAdapter::normalize)
                .anyMatch(modelType::equals);
        boolean architectureMatches = !architectureClassName.isBlank() && safeSupportedArchClassNames(adapter).stream()
                .map(ModelFamilyRuntimeAdapter::normalize)
                .anyMatch(architectureClassName::equals);
        return modelTypeMatches || architectureMatches;
    }

    private static List<String> safeSupportedModelTypes(ModelArchitecture adapter) {
        try {
            List<String> modelTypes = adapter.supportedModelTypes();
            return modelTypes == null ? List.of() : modelTypes;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> safeSupportedArchClassNames(ModelArchitecture adapter) {
        try {
            List<String> architectures = adapter.supportedArchClassNames();
            return architectures == null ? List.of() : architectures;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static String normalize(String value) {
        return Objects.toString(value, "").trim().toLowerCase(Locale.ROOT);
    }

    private static List<String> splitMetadata(String... values) {
        return Arrays.stream(values == null ? new String[0] : values)
                .filter(value -> value != null && !value.isBlank())
                .flatMap(value -> Arrays.stream(value.split(",")))
                .map(value -> Objects.toString(value, "").trim())
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
    }

    private static String firstNonBlank(String... values) {
        return Arrays.stream(values == null ? new String[0] : values)
                .filter(value -> value != null && !value.isBlank())
                .map(String::trim)
                .findFirst()
                .orElse("");
    }
}
