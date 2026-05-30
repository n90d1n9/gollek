package tech.kayys.gollek.model.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Any;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.model.ModelFamilyResolution;
import tech.kayys.gollek.spi.model.ModelFamilySupportReport;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.jboss.logging.Logger;

/**
 * CDI registry that auto-discovers all {@link ModelArchitecture}
 * implementations
 * and resolves the correct one for a given {@link ModelConfig}.
 */
@ApplicationScoped
public class ModelArchitectureRegistry {

    private static final Logger log = Logger.getLogger(ModelArchitectureRegistry.class);

    @Inject
    @Any
    Instance<ModelArchitecture> allArchitectures;

    @Inject
    @Any
    Instance<ModelFamilyPlugin> allFamilyPlugins;

    public Instance<ModelArchitecture> getAllArchitectures() {
        return allArchitectures;
    }

    /**
     * Resolve the architecture for a loaded model config.
     *
     * @param config the parsed config.json
     * @return the matching architecture
     * @throws IllegalArgumentException if no registered architecture matches
     */
    public ModelArchitecture resolve(ModelConfig config) {
        String primaryArch = config.primaryArchitecture();
        String modelType = config.modelType();

        registerFamilyPlugins();
        List<ModelArchitecture> familyAdapters = ModelFamilyPluginRegistry.global().architectureAdaptersFor(config);
        ModelArchitecture resolved = resolveFrom(familyAdapters, modelType, primaryArch);
        if (resolved != null) {
            log.debugf("Architecture resolved by matched model-family plugin: %s → %s",
                    modelType != null ? modelType : primaryArch,
                    resolved.id());
            return resolved;
        }

        resolved = resolveFrom(discoverArchitectures(), modelType, primaryArch);
        if (resolved != null) {
            return resolved;
        }
        throw new IllegalArgumentException(
                "No registered ModelArchitecture for arch='" + primaryArch
                        + "', model_type='" + modelType + "'. "
                        + supportHint(modelType, primaryArch)
                        + "Register a custom ModelArchitecture CDI bean or install a direct-ready model-family plugin.");
    }

    private ModelArchitecture resolveFrom(List<ModelArchitecture> architectures, String modelType, String primaryArch) {
        for (ModelArchitecture arch : architectures) {
            if (modelType != null && arch.supportedModelTypes().contains(modelType)) {
                log.debugf("Architecture resolved by model_type: %s → %s", modelType, arch.id());
                return arch;
            }
        }
        for (ModelArchitecture arch : architectures) {
            if (arch.supportedArchClassNames().contains(primaryArch)) {
                log.debugf("Architecture resolved by class name: %s → %s", primaryArch, arch.id());
                return arch;
            }
        }
        return null;
    }

    /**
     * Resolve the architecture for a GGUF model architecture string.
     *
     * @param ggufArch the architecture string from GGUF metadata
     * @return the matching architecture
     * @throws IllegalArgumentException if no registered architecture matches
     */
    public ModelArchitecture resolveGguf(String ggufArch) {
        for (ModelArchitecture arch : discoverArchitectures()) {
            if (arch.matchesGgufArch(ggufArch)) {
                log.debugf("Architecture resolved by GGUF arch: %s → %s", ggufArch, arch.id());
                return arch;
            }
        }
        throw new IllegalArgumentException(
                "No registered ModelArchitecture for GGUF arch='" + ggufArch + "'. "
                        + supportHint(ggufArch, ggufArch)
                        + "Register a custom ModelArchitecture CDI bean or install a direct-ready model-family plugin.");
    }

    /** All registered architecture IDs — useful for health/info endpoints. */
    public List<String> registeredIds() {
        List<String> ids = new ArrayList<>();
        discoverArchitectures().forEach(a -> ids.add(a.id()));
        return Collections.unmodifiableList(ids);
    }

    /** All registered model-family descriptors for diagnostics and CLI output. */
    public List<ModelFamilyDescriptor> registeredFamilies() {
        registerFamilyPlugins();
        return ModelFamilyPluginRegistry.global().descriptors();
    }

    private List<ModelArchitecture> discoverArchitectures() {
        registerFamilyPlugins();

        Map<String, ModelArchitecture> architectures = new LinkedHashMap<>();
        if (allArchitectures != null && !allArchitectures.isUnsatisfied()) {
            for (ModelArchitecture arch : allArchitectures) {
                architectures.putIfAbsent(arch.id(), arch);
            }
        }
        for (ModelArchitecture arch : ModelFamilyPluginRegistry.global().architectureAdapters()) {
            architectures.putIfAbsent(arch.id(), arch);
        }
        return List.copyOf(architectures.values());
    }

    private void registerFamilyPlugins() {
        if (allFamilyPlugins != null && !allFamilyPlugins.isUnsatisfied()) {
            for (ModelFamilyPlugin plugin : allFamilyPlugins) {
                ModelFamilyPluginRegistry.global().register(plugin);
            }
        }
        ModelFamilyPluginRegistry.global().discoverServiceLoaderPlugins();
    }

    private String supportHint(String modelType, String primaryArch) {
        ModelFamilyResolution resolution = ModelFamilyPluginRegistry.global().resolve(modelType, primaryArch);
        List<ModelFamilySupportReport> reports = resolution.supportReports();
        StringBuilder hint = new StringBuilder("Model-family resolution: ")
                .append(resolution.summary())
                .append(". ");
        if (!resolution.problemCodes().isEmpty()) {
            hint.append("Problems: ")
                    .append(String.join(", ", resolution.problemCodes()))
                    .append(". ");
        }
        if (!resolution.remediationHints().isEmpty()) {
            hint.append("Remediation: ")
                    .append(String.join(" ", resolution.remediationHints()))
                    .append(" ");
        }
        String summary = reports.stream()
                .map(report -> report.id() + "[" + report.shortDirectSafetensorSummary() + "]")
                .reduce((a, b) -> a + ", " + b)
                .orElse("");
        if (!summary.isBlank()) {
            hint.append("Matching model-family plugin(s): ")
                    .append(summary)
                    .append(". ");
        }
        if (resolution.resolved()) {
            hint.append("No compatible architecture adapter was selected from the matched family. ");
        }
        return hint.toString();
    }
}
