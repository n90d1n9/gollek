package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Validates a lightweight Hugging Face style model fixture against a detachable
 * model-family plugin.
 *
 * <p>Fixtures are intentionally tiny: usually a {@code config.json} plus one
 * tokenizer marker file such as {@code tokenizer.json}. The goal is to catch
 * drift between Transformers config claims, reusable tokenizer descriptors, and
 * direct-runtime architecture adapters before a family is bundled for
 * production.</p>
 */
public final class ModelFamilyFixtureValidator {

    private ModelFamilyFixtureValidator() {
    }

    public record Options(
            boolean requireTokenizerFiles,
            boolean requireArchitectureAdapterMatch,
            boolean requireArchitectureClaim) {

        public static Options strict() {
            return new Options(true, true, true);
        }

        public static Options configOnly() {
            return new Options(false, false, true);
        }
    }

    public static List<ModelFamilyContractViolation> validate(
            ModelFamilyPlugin plugin,
            Path modelDir) {
        return validate(plugin, modelDir, Options.strict());
    }

    public static List<ModelFamilyContractViolation> validate(
            ModelFamilyPlugin plugin,
            Path modelDir,
            Options options) {
        return validate(plugin, modelDir, new ObjectMapper(), options);
    }

    private static List<ModelFamilyContractViolation> validate(
            ModelFamilyPlugin plugin,
            Path modelDir,
            ObjectMapper mapper,
            Options options) {
        Options resolvedOptions = options == null ? Options.strict() : options;
        List<ModelFamilyContractViolation> violations = new ArrayList<>();

        if (plugin == null) {
            return List.of(new ModelFamilyContractViolation("unknown", "fixture_plugin_null",
                    "model-family fixture validation requires a plugin instance"));
        }

        ModelFamilyDescriptor descriptor = safeDescriptor(plugin, violations);
        String familyId = descriptor == null ? safeFamilyId(plugin) : descriptor.id();
        if (descriptor == null) {
            return List.copyOf(violations);
        }

        if (modelDir == null) {
            return List.of(new ModelFamilyContractViolation(familyId, "fixture_directory_null",
                    "model-family fixture directory must not be null"));
        }
        if (!Files.isDirectory(modelDir)) {
            return List.of(new ModelFamilyContractViolation(familyId, "fixture_directory_missing",
                    "model-family fixture directory does not exist: " + modelDir));
        }

        Path configPath = modelDir.resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            return List.of(new ModelFamilyContractViolation(familyId, "fixture_config_missing",
                    "model-family fixture must include config.json under " + modelDir));
        }

        JsonNode configRoot;
        ModelConfig config;
        try {
            configRoot = mapper.readTree(configPath.toFile());
            config = new ModelConfigLoader(mapper).load(configPath);
        } catch (IOException | RuntimeException error) {
            return List.of(new ModelFamilyContractViolation(familyId, "fixture_config_unreadable",
                    "could not parse fixture config.json: "
                            + error.getClass().getSimpleName() + ": " + error.getMessage()));
        }

        Set<String> modelTypes = collectModelTypes(configRoot, config);
        Set<String> architectures = collectArchitectures(configRoot, config);

        validateModelTypeClaims(descriptor, modelTypes, violations);
        validateArchitectureClaims(descriptor, architectures, resolvedOptions, violations);
        validateTokenizerFiles(plugin, descriptor, modelDir, resolvedOptions, violations);
        validateArchitectureAdapterMatch(plugin, descriptor, modelTypes, architectures, resolvedOptions, violations);

        return List.copyOf(violations);
    }

    private static void validateModelTypeClaims(
            ModelFamilyDescriptor descriptor,
            Set<String> modelTypes,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        if (modelTypes.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "fixture_model_type_missing",
                    "fixture config.json should declare at least one model_type"));
            return;
        }
        if (!intersects(modelTypes, descriptor.modelTypes())) {
            violations.add(new ModelFamilyContractViolation(familyId, "fixture_model_type_unclaimed",
                    "fixture model_type values " + sorted(modelTypes)
                            + " are not claimed by descriptor modelTypes " + sorted(descriptor.modelTypes())));
        }
    }

    private static void validateArchitectureClaims(
            ModelFamilyDescriptor descriptor,
            Set<String> architectures,
            Options options,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        if (architectures.isEmpty()) {
            if (options.requireArchitectureClaim()) {
                violations.add(new ModelFamilyContractViolation(familyId, "fixture_architecture_missing",
                        "fixture config.json should declare at least one architectures entry"));
            }
            return;
        }
        if (!intersects(architectures, descriptor.architectureClassNames())) {
            violations.add(new ModelFamilyContractViolation(familyId, "fixture_architecture_unclaimed",
                    "fixture architectures " + sorted(architectures)
                            + " are not claimed by descriptor architectureClassNames "
                            + sorted(descriptor.architectureClassNames())));
        }
    }

    private static void validateTokenizerFiles(
            ModelFamilyPlugin plugin,
            ModelFamilyDescriptor descriptor,
            Path modelDir,
            Options options,
            List<ModelFamilyContractViolation> violations) {
        if (!options.requireTokenizerFiles()
                || !descriptor.capabilities().contains(ModelFamilyCapability.TOKENIZER)) {
            return;
        }

        List<ModelTokenizerDescriptor> tokenizers = safeTokenizers(plugin, descriptor.id(), violations);
        if (tokenizers.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(descriptor.id(), "fixture_tokenizer_descriptors_missing",
                    "fixture requires tokenizer files but plugin has no tokenizer descriptors"));
            return;
        }

        boolean matched = tokenizers.stream()
                .filter(Objects::nonNull)
                .anyMatch(tokenizer -> tokenizer.firstExistingFileGroup(modelDir).isPresent());
        if (!matched) {
            violations.add(new ModelFamilyContractViolation(descriptor.id(), "fixture_tokenizer_files_unmatched",
                    "fixture does not satisfy any tokenizer descriptor file group: "
                            + tokenizers.stream()
                                    .filter(Objects::nonNull)
                                    .map(ModelTokenizerDescriptor::id)
                                    .sorted()
                                    .toList()));
        }
    }

    private static void validateArchitectureAdapterMatch(
            ModelFamilyPlugin plugin,
            ModelFamilyDescriptor descriptor,
            Set<String> modelTypes,
            Set<String> architectures,
            Options options,
            List<ModelFamilyContractViolation> violations) {
        if (!options.requireArchitectureAdapterMatch() || !descriptor.supportsDirectSafetensorInference()) {
            return;
        }

        List<ModelArchitecture> adapters = safeAdapters(plugin, descriptor.id(), violations);
        boolean matched = adapters.stream()
                .filter(Objects::nonNull)
                .anyMatch(adapter -> intersects(modelTypes, safeModelTypes(adapter))
                        || intersects(architectures, safeArchitectureClassNames(adapter)));
        if (!matched) {
            violations.add(new ModelFamilyContractViolation(descriptor.id(),
                    "fixture_architecture_adapter_unmatched",
                    "no direct architecture adapter matches fixture model_type " + sorted(modelTypes)
                            + " or architectures " + sorted(architectures)));
        }
    }

    private static ModelFamilyDescriptor safeDescriptor(
            ModelFamilyPlugin plugin,
            List<ModelFamilyContractViolation> violations) {
        try {
            ModelFamilyDescriptor descriptor = plugin.descriptor();
            if (descriptor == null) {
                violations.add(new ModelFamilyContractViolation(safeFamilyId(plugin), "fixture_descriptor_null",
                        "descriptor() must not return null"));
            }
            return descriptor;
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(safeFamilyId(plugin), "fixture_descriptor_unavailable",
                    "descriptor() threw " + error.getClass().getSimpleName() + ": " + error.getMessage()));
            return null;
        }
    }

    private static List<ModelTokenizerDescriptor> safeTokenizers(
            ModelFamilyPlugin plugin,
            String familyId,
            List<ModelFamilyContractViolation> violations) {
        try {
            List<ModelTokenizerDescriptor> tokenizers = plugin.tokenizerDescriptors();
            return tokenizers == null ? List.of() : List.copyOf(tokenizers);
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "fixture_tokenizers_unavailable",
                    "tokenizerDescriptors() threw " + error.getClass().getSimpleName() + ": "
                            + error.getMessage()));
            return List.of();
        }
    }

    private static List<ModelArchitecture> safeAdapters(
            ModelFamilyPlugin plugin,
            String familyId,
            List<ModelFamilyContractViolation> violations) {
        try {
            List<ModelArchitecture> adapters = plugin.architectureAdapters();
            return adapters == null ? List.of() : List.copyOf(adapters);
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "fixture_architecture_adapters_unavailable",
                    "architectureAdapters() threw " + error.getClass().getSimpleName() + ": "
                            + error.getMessage()));
            return List.of();
        }
    }

    private static List<String> safeModelTypes(ModelArchitecture adapter) {
        try {
            List<String> modelTypes = adapter.supportedModelTypes();
            return modelTypes == null ? List.of() : modelTypes;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> safeArchitectureClassNames(ModelArchitecture adapter) {
        try {
            List<String> architectures = adapter.supportedArchClassNames();
            return architectures == null ? List.of() : architectures;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static Set<String> collectModelTypes(JsonNode root, ModelConfig config) {
        Set<String> values = new LinkedHashSet<>();
        collectFieldValues(root, "model_type", values);
        addIfPresent(values, config.getModelType());
        return values;
    }

    private static Set<String> collectArchitectures(JsonNode root, ModelConfig config) {
        Set<String> values = new LinkedHashSet<>();
        collectFieldValues(root, "architectures", values);
        if (config.getArchitectures() != null) {
            config.getArchitectures().forEach(value -> addIfPresent(values, value));
        }
        return values;
    }

    private static void collectFieldValues(JsonNode node, String fieldName, Set<String> values) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return;
        }
        if (node.isObject()) {
            JsonNode field = node.get(fieldName);
            if (field != null) {
                addJsonValues(values, field);
            }
            node.fields().forEachRemaining(entry -> collectFieldValues(entry.getValue(), fieldName, values));
        } else if (node.isArray()) {
            node.forEach(child -> collectFieldValues(child, fieldName, values));
        }
    }

    private static void addJsonValues(Set<String> values, JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return;
        }
        if (node.isArray()) {
            node.forEach(child -> addJsonValues(values, child));
            return;
        }
        if (node.isTextual() || node.isNumber() || node.isBoolean()) {
            addIfPresent(values, node.asText());
        }
    }

    private static void addIfPresent(Set<String> values, String value) {
        String text = Objects.toString(value, "").trim();
        if (!text.isBlank()) {
            values.add(text);
        }
    }

    private static boolean intersects(Collection<String> first, Collection<String> second) {
        Set<String> normalizedSecond = new LinkedHashSet<>();
        for (String value : second) {
            String normalized = normalize(value);
            if (!normalized.isBlank()) {
                normalizedSecond.add(normalized);
            }
        }
        for (String value : first) {
            if (normalizedSecond.contains(normalize(value))) {
                return true;
            }
        }
        return false;
    }

    private static List<String> sorted(Collection<String> values) {
        return values.stream()
                .filter(Objects::nonNull)
                .map(String::trim)
                .filter(value -> !value.isBlank())
                .sorted()
                .toList();
    }

    private static String safeFamilyId(ModelFamilyPlugin plugin) {
        try {
            String pluginId = Objects.toString(plugin.id(), "").trim();
            if (pluginId.startsWith("model-family/")) {
                return pluginId.substring("model-family/".length());
            }
            return pluginId.isBlank() ? "unknown" : pluginId;
        } catch (RuntimeException error) {
            return "unknown";
        }
    }

    private static String normalize(String value) {
        return Objects.toString(value, "").trim().toLowerCase(Locale.ROOT);
    }
}
