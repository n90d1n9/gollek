package tech.kayys.gollek.spi.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Process-local registry for model-family plugins discovered through CDI,
 * ServiceLoader, or the plugin manager.
 */
public final class ModelFamilyPluginRegistry {

    private static final ModelFamilyPluginRegistry GLOBAL = new ModelFamilyPluginRegistry();

    private final Map<String, ModelFamilyPlugin> plugins = new ConcurrentHashMap<>();
    private volatile boolean serviceLoaderScanned;

    private ModelFamilyPluginRegistry() {
    }

    public static ModelFamilyPluginRegistry global() {
        return GLOBAL;
    }

    public void register(ModelFamilyPlugin plugin) {
        if (plugin != null) {
            String pluginId = safePluginId(plugin);
            if (pluginId.isBlank()) {
                pluginId = plugin.getClass().getName() + "@" + System.identityHashCode(plugin);
            }
            plugins.put(normalizePluginId(pluginId), plugin);
        }
    }

    public void unregister(String pluginId) {
        if (pluginId != null) {
            String normalized = normalizePluginId(pluginId);
            plugins.remove(normalized);
            plugins.entrySet().removeIf(entry -> {
                ModelFamilyPlugin plugin = entry.getValue();
                boolean pluginIdMatches = normalizePluginId(safePluginId(plugin)).equals(normalized);
                boolean descriptorIdMatches = safeDescriptor(plugin)
                        .map(descriptor -> normalizePluginId(descriptor.id()).equals(normalized))
                        .orElse(false);
                return pluginIdMatches || descriptorIdMatches;
            });
        }
    }

    public Optional<ModelFamilyPlugin> plugin(String pluginId) {
        if (pluginId == null || pluginId.isBlank()) {
            return Optional.empty();
        }
        String normalized = normalizePluginId(pluginId);
        ModelFamilyPlugin plugin = plugins.get(normalized);
        if (plugin != null) {
            return Optional.of(plugin);
        }
        return all().stream()
                .filter(candidate -> normalizePluginId(safePluginId(candidate)).equals(normalized)
                        || safeDescriptor(candidate)
                                .map(descriptor -> normalizePluginId(descriptor.id()).equals(normalized))
                                .orElse(false))
                .findFirst();
    }

    public List<ModelFamilyPlugin> all() {
        return plugins.values().stream()
                .sorted(Comparator.comparingInt(ModelFamilyPluginRegistry::safeOrder)
                        .thenComparing(ModelFamilyPluginRegistry::safePluginSortKey))
                .toList();
    }

    public List<ModelArchitecture> architectureAdapters() {
        Map<String, ModelArchitecture> adapters = new LinkedHashMap<>();
        for (ModelFamilyPlugin plugin : all()) {
            for (ModelArchitecture adapter : safeArchitectureAdapters(plugin)) {
                String adapterId = safeAdapterId(adapter);
                if (!adapterId.isBlank()) {
                    adapters.putIfAbsent(adapterId, adapter);
                }
            }
        }
        return List.copyOf(adapters.values());
    }

    public List<ModelArchitecture> architectureAdaptersFor(ModelConfig config) {
        if (config == null) {
            return List.of();
        }
        return architectureAdaptersFor(config.modelType(), config.primaryArchitecture());
    }

    public List<ModelArchitecture> architectureAdaptersFor(String modelType, String architectureClassName) {
        Map<String, ModelArchitecture> adapters = new LinkedHashMap<>();
        for (ModelFamilyPlugin plugin : pluginsFor(modelType, architectureClassName)) {
            for (ModelArchitecture adapter : safeArchitectureAdapters(plugin)) {
                String adapterId = safeAdapterId(adapter);
                if (!adapterId.isBlank()) {
                    adapters.putIfAbsent(adapterId, adapter);
                }
            }
        }
        return List.copyOf(adapters.values());
    }

    public List<ModelFamilyPlugin> pluginsFor(String modelType, String architectureClassName) {
        String normalizedModelType = normalize(modelType);
        String normalizedArch = normalize(architectureClassName);
        List<ModelFamilyPlugin> matches = new ArrayList<>();
        for (ModelFamilyPlugin plugin : all()) {
            Optional<ModelFamilyDescriptor> maybeDescriptor = safeDescriptor(plugin);
            if (maybeDescriptor.isEmpty()) {
                continue;
            }
            ModelFamilyDescriptor descriptor = maybeDescriptor.orElseThrow();
            boolean modelTypeMatches = !normalizedModelType.isBlank()
                    && descriptor.modelTypes().stream()
                            .map(ModelFamilyPluginRegistry::normalize)
                            .anyMatch(normalizedModelType::equals);
            boolean architectureMatches = !normalizedArch.isBlank()
                    && descriptor.architectureClassNames().stream()
                            .map(ModelFamilyPluginRegistry::normalize)
                            .anyMatch(normalizedArch::equals);
            if (modelTypeMatches || architectureMatches) {
                matches.add(plugin);
            }
        }
        return List.copyOf(matches);
    }

    public List<ModelFamilyPlugin> pluginsForModelType(String modelType) {
        return pluginsFor(modelType, null);
    }

    /**
     * Resolve the parsed {@code config.json} identity to attached families.
     */
    public ModelFamilyResolution resolve(ModelConfig config) {
        if (config == null) {
            return resolve(null, null);
        }
        return resolve(config.modelType(), config.primaryArchitecture());
    }

    /**
     * Resolve Hugging Face config identity claims to attached model-family plugins.
     */
    public ModelFamilyResolution resolve(String modelType, String architectureClassName) {
        List<ModelFamilyPlugin> matches = pluginsFor(modelType, architectureClassName);
        List<String> familyIds = matches.stream()
                .map(plugin -> safeDescriptor(plugin)
                        .map(ModelFamilyDescriptor::id)
                        .orElseGet(() -> normalizePluginId(safePluginId(plugin))))
                .distinct()
                .toList();
        ModelFamilyResolution.Status status = switch (familyIds.size()) {
            case 0 -> ModelFamilyResolution.Status.NOT_FOUND;
            case 1 -> ModelFamilyResolution.Status.RESOLVED;
            default -> ModelFamilyResolution.Status.AMBIGUOUS;
        };
        return new ModelFamilyResolution(
                modelType,
                architectureClassName,
                status,
                familyIds,
                matches.stream()
                        .map(ModelFamilyPluginRegistry::safeSupportReport)
                        .flatMap(Optional::stream)
                        .toList(),
                matches.stream()
                        .flatMap(plugin -> safeTokenizerDescriptors(plugin).stream())
                        .toList());
    }

    public ModelFamilyResolution resolveModelType(String modelType) {
        return resolve(modelType, null);
    }

    public List<ModelTokenizerDescriptor> tokenizerDescriptorsFor(String modelType, String architectureClassName) {
        List<ModelTokenizerDescriptor> descriptors = new ArrayList<>();
        for (ModelFamilyPlugin plugin : pluginsFor(modelType, architectureClassName)) {
            descriptors.addAll(safeTokenizerDescriptors(plugin));
        }
        return List.copyOf(descriptors);
    }

    public List<ModelFamilySupportReport> supportReports() {
        return all().stream()
                .map(ModelFamilyPluginRegistry::safeSupportReport)
                .flatMap(Optional::stream)
                .toList();
    }

    public List<ModelFamilySupportReport> supportReportsForProfile(ModelFamilyBundleProfile profile) {
        ModelFamilyBundleProfile target = profile == null ? ModelFamilyBundleProfile.OPTIONAL : profile;
        return supportReports().stream()
                .filter(report -> report.bundleProfile() == target)
                .toList();
    }

    public List<ModelFamilySupportReport> supportReportsFor(String modelType, String architectureClassName) {
        return pluginsFor(modelType, architectureClassName).stream()
                .map(ModelFamilyPluginRegistry::safeSupportReport)
                .flatMap(Optional::stream)
                .toList();
    }

    public List<ModelFamilyCapabilityMatrixEntry> capabilityMatrix() {
        return supportReports().stream()
                .map(ModelFamilyCapabilityMatrixEntry::from)
                .toList();
    }

    public synchronized List<ModelFamilyPlugin> discoverServiceLoaderPlugins() {
        if (serviceLoaderScanned) {
            return all();
        }
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader == null) {
            loader = ModelFamilyPlugin.class.getClassLoader();
        }
        var iterator = ServiceLoader.load(ModelFamilyPlugin.class, loader).iterator();
        while (true) {
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                register(iterator.next());
            } catch (RuntimeException | ServiceConfigurationError error) {
                // Keep discovery resilient: one bad extension should not hide
                // families that are packaged correctly.
            }
        }
        serviceLoaderScanned = true;
        return all();
    }

    public Map<String, List<String>> modelTypeClaims() {
        Map<String, List<String>> claims = new LinkedHashMap<>();
        for (ModelFamilyPlugin plugin : all()) {
            Optional<ModelFamilyDescriptor> maybeDescriptor = safeDescriptor(plugin);
            if (maybeDescriptor.isEmpty()) {
                continue;
            }
            ModelFamilyDescriptor descriptor = maybeDescriptor.orElseThrow();
            for (String modelType : descriptor.modelTypes()) {
                String normalized = normalize(modelType);
                if (!normalized.isBlank()) {
                    List<String> familyIds = new ArrayList<>(
                            claims.getOrDefault(normalized, List.of()));
                    familyIds.add(descriptor.id());
                    claims.put(normalized, List.copyOf(familyIds));
                }
            }
        }
        return Collections.unmodifiableMap(claims);
    }

    public List<ModelFamilyClaimConflict> modelTypeConflicts() {
        List<ModelFamilyClaimConflict> conflicts = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : modelTypeClaims().entrySet()) {
            List<String> familyIds = entry.getValue().stream().distinct().toList();
            if (familyIds.size() > 1) {
                conflicts.add(new ModelFamilyClaimConflict("model_type", entry.getKey(), familyIds));
            }
        }
        return List.copyOf(conflicts);
    }

    public List<ModelFamilyContractViolation> contractViolations() {
        return ModelFamilyContractValidator.validateAll(all());
    }

    public List<ModelFamilyDescriptor> descriptors() {
        List<ModelFamilyDescriptor> descriptors = new ArrayList<>();
        for (ModelFamilyPlugin plugin : all()) {
            safeDescriptor(plugin).ifPresent(descriptors::add);
        }
        return List.copyOf(descriptors);
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static Optional<ModelFamilySupportReport> safeSupportReport(ModelFamilyPlugin plugin) {
        try {
            return Optional.ofNullable(plugin.supportReport());
        } catch (RuntimeException error) {
            return Optional.empty();
        }
    }

    private static Optional<ModelFamilyDescriptor> safeDescriptor(ModelFamilyPlugin plugin) {
        try {
            return Optional.ofNullable(plugin.descriptor());
        } catch (RuntimeException error) {
            return Optional.empty();
        }
    }

    private static String safePluginId(ModelFamilyPlugin plugin) {
        try {
            String id = plugin.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
        }
    }

    private static String safePluginSortKey(ModelFamilyPlugin plugin) {
        String pluginId = safePluginId(plugin);
        if (!pluginId.isBlank()) {
            return pluginId;
        }
        return plugin.getClass().getName() + "@" + System.identityHashCode(plugin);
    }

    private static int safeOrder(ModelFamilyPlugin plugin) {
        try {
            return plugin.order();
        } catch (RuntimeException error) {
            return Integer.MAX_VALUE;
        }
    }

    private static List<ModelArchitecture> safeArchitectureAdapters(ModelFamilyPlugin plugin) {
        try {
            List<ModelArchitecture> adapters = plugin.architectureAdapters();
            return adapters == null ? List.of() : adapters.stream()
                    .filter(adapter -> adapter != null)
                    .toList();
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<ModelTokenizerDescriptor> safeTokenizerDescriptors(ModelFamilyPlugin plugin) {
        try {
            List<ModelTokenizerDescriptor> descriptors = plugin.tokenizerDescriptors();
            return descriptors == null ? List.of() : descriptors.stream()
                    .filter(descriptor -> descriptor != null)
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

    private static String normalizePluginId(String value) {
        String normalized = normalize(value);
        return normalized.startsWith("model-family/")
                ? normalized
                : "model-family/" + normalized;
    }
}
