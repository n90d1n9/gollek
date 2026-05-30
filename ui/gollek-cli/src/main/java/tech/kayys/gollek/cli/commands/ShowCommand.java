package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Any;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelResolver;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDirectSupport;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.model.ModelFamilyResolution;
import tech.kayys.gollek.spi.model.ModelFamilySupportReport;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.context.RequestContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Show model details using GollekSdk.
 * Usage: gollek show <model-id>
 */
@Dependent
@Unremovable
@Command(name = "show", description = "Show details for a specific model")
public class ShowCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    @Inject
    GollekSdk sdk;

    @Inject
    @Any
    Instance<ModelFamilyPlugin> modelFamilyPluginInstances;

    @Parameters(index = "0", description = "Model ID or path")
    public String modelId;

    @Option(names = { "--json" }, description = "Print model details as JSON")
    boolean json;

    @Override
    public void run() {
        try {
            Optional<ModelResolver.ResolvedModel> resolvedOpt = ModelResolver.resolve(sdk, modelId);
            if (resolvedOpt.isEmpty()) {
                LocalModelIndex.refreshFromDisk();
                Optional<LocalModelIndex.Entry> idx = LocalModelIndex.find(modelId);
                if (idx.isEmpty()) {
                    System.err.println("Model not found: " + modelId);
                    return;
                }
                LocalModelIndex.Entry entry = idx.get();
                ModelInfo model = ModelInfo.builder()
                        .modelId(entry.id != null ? entry.id : modelId)
                        .name(entry.name)
                        .format(entry.format)
                        .sizeBytes(entry.sizeBytes)
                        .updatedAt(LocalModelIndex.parseInstant(entry.updatedAt))
                        .requestContext(RequestContext.of("community", "community"))
                        .metadata(java.util.Map.of(
                                "path", entry.path != null ? entry.path : "",
                                "source", entry.source != null ? entry.source : "local"))
                        .build();
                resolvedOpt = Optional.of(new ModelResolver.ResolvedModel(
                        model.getModelId(), model, entry.path != null ? Path.of(entry.path) : null, false));
            }

            ModelResolver.ResolvedModel resolved = resolvedOpt.get();
            Optional<ModelFamilyResolution> modelFamily = resolveModelFamily(resolved);

            if (json) {
                printModelJson(resolved, modelFamily);
            } else {
                printModelDetails(resolved, modelFamily);
            }

        } catch (Exception e) {
            System.err.println("Failed to show model: " + e.getMessage());
        }
    }

    private void printModelDetails(
            ModelResolver.ResolvedModel resolved,
            Optional<ModelFamilyResolution> modelFamily) {
        ModelInfo model = resolved.info();
        System.out.println("Model Details");
        System.out.println("=".repeat(50));
        System.out.printf("ID:       %s%n", model.getModelId());
        System.out.printf("Name:     %s%n", model.getName() != null ? model.getName() : "N/A");
        System.out.printf("Version:  %s%n", model.getVersion() != null ? model.getVersion() : "N/A");
        System.out.printf("Format:   %s%n", model.getFormat() != null ? model.getFormat() : "N/A");
        System.out.printf("Runtime:  %s%n", isRunnableLocally(resolved) ? "runnable" : "checkpoint-only");
        System.out.printf("Size:     %s%n", model.getSizeFormatted());
        if (model.getQuantization() != null) {
            System.out.printf("Quant:    %s%n", model.getQuantization());
        }
        System.out.printf("Created:  %s%n", model.getCreatedAt() != null ? model.getCreatedAt() : "N/A");
        System.out.printf("Modified: %s%n", model.getUpdatedAt() != null ? model.getUpdatedAt() : "N/A");

        if (model.getMetadata() != null && !model.getMetadata().isEmpty()) {
            System.out.println("\nMetadata:");
            model.getMetadata().forEach((key, value) -> System.out.printf("  %s: %s%n", key, value));
        }
        modelFamily.ifPresent(resolution -> printModelFamilyDetails(resolution, modelDirectory(resolved)));
        if (!isRunnableLocally(resolved)) {
            System.out.println("\nNote:");
            System.out.println(
                    "  Stored as origin checkpoint artifacts; convert to GGUF/TorchScript for local inference.");
        }
    }

    private void printModelFamilyDetails(ModelFamilyResolution resolution, Optional<Path> modelDir) {
        System.out.println("\nModel Family:");
        System.out.printf("  Status:   %s%n", resolution.status());
        System.out.printf("  Summary:  %s%n", resolution.summary());
        System.out.printf("  Families: %s%n", resolution.familyIds().isEmpty()
                ? "N/A"
                : String.join(", ", resolution.familyIds()));
        List<String> problemCodes = modelFamilyProblemCodes(resolution, modelDir);
        if (!problemCodes.isEmpty()) {
            System.out.printf("  Problems: %s%n", String.join(", ", problemCodes));
            System.out.println("  Remediation:");
            for (String hint : modelFamilyRemediationHints(resolution, modelDir)) {
                System.out.printf("    %s%n", hint);
            }
        }
        if (!resolution.supportReports().isEmpty()) {
            System.out.println("  Support:");
            for (ModelFamilySupportReport report : resolution.supportReports()) {
                System.out.printf("    %s: bundle=%s direct=%s tokenizers=%s%n",
                        report.id(),
                        report.bundleProfile(),
                        report.shortDirectSafetensorSummary(),
                        report.tokenizerProfileIds().isEmpty()
                                ? "none"
                                : String.join(", ", report.tokenizerProfileIds()));
            }
        }
        Map<String, Object> directArchitecture = directArchitectureReport(resolution);
        @SuppressWarnings("unchecked")
        List<String> adapterIds = (List<String>) directArchitecture.get("adapterIds");
        if (!adapterIds.isEmpty()) {
            System.out.printf("  Direct Architecture: selected=%s adapters=%s%n",
                    directArchitecture.get("selectedAdapterId"),
                    String.join(", ", adapterIds));
        }
        if (!resolution.tokenizerDescriptors().isEmpty()) {
            System.out.println("  Tokenizers:");
            for (ModelTokenizerDescriptor descriptor : resolution.tokenizerDescriptors()) {
                String usable = modelDir
                        .map(dir -> descriptor.firstExistingFileGroup(dir).isPresent() ? "usable" : "missing files")
                        .orElse("not inspected");
                System.out.printf("    %s (%s): %s%n", descriptor.id(), descriptor.kind(), usable);
            }
        }
    }

    private void printModelJson(
            ModelResolver.ResolvedModel resolved,
            Optional<ModelFamilyResolution> modelFamily) throws Exception {
        ModelInfo model = resolved.info();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", model.getModelId());
        out.put("name", model.getName());
        out.put("version", model.getVersion());
        out.put("format", model.getFormat());
        out.put("runtime", isRunnableLocally(resolved) ? "runnable" : "checkpoint-only");
        out.put("sizeBytes", model.getSizeBytes());
        out.put("size", model.getSizeFormatted());
        out.put("createdAt", model.getCreatedAt() != null ? model.getCreatedAt().toString() : null);
        out.put("updatedAt", model.getUpdatedAt() != null ? model.getUpdatedAt().toString() : null);
        out.put("metadata", model.getMetadata());
        Optional<Path> modelDir = modelDirectory(resolved);
        out.put("modelFamily", modelFamily.map(resolution -> modelFamilyReport(resolution, modelDir)).orElse(null));
        System.out.println(JSON.writeValueAsString(out));
    }

    private Map<String, Object> modelFamilyReport(ModelFamilyResolution resolution, Optional<Path> modelDir) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("status", resolution.status().name());
        report.put("resolved", resolution.resolved());
        report.put("ambiguous", resolution.ambiguous());
        report.put("modelType", resolution.modelType());
        report.put("architectureClassName", resolution.architectureClassName());
        report.put("familyIds", resolution.familyIds());
        report.put("summary", resolution.summary());
        report.put("requiresAttention", !modelFamilyProblemCodes(resolution, modelDir).isEmpty());
        report.put("problemCodes", modelFamilyProblemCodes(resolution, modelDir));
        report.put("remediationHints", modelFamilyRemediationHints(resolution, modelDir));
        report.put("supportReports", resolution.supportReports().stream()
                .map(this::supportReport)
                .toList());
        report.put("directArchitecture", directArchitectureReport(resolution));
        report.put("tokenizers", resolution.tokenizerDescriptors().stream()
                .map(descriptor -> tokenizerReport(descriptor, modelDir))
                .toList());
        return report;
    }

    private Map<String, Object> supportReport(ModelFamilySupportReport report) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", report.id());
        out.put("displayName", report.displayName());
        out.put("bundleProfile", report.bundleProfile().name());
        out.put("capabilities", report.capabilities().stream().map(Enum::name).toList());
        out.put("architectureAdapterIds", report.architectureAdapterIds());
        out.put("tokenizerProfileIds", report.tokenizerProfileIds());
        out.put("tokenizerKinds", report.tokenizerKinds().stream().map(Enum::name).toList());
        out.put("directSafetensorStatus", report.directSafetensorStatus().name());
        out.put("directSafetensorReason", report.directSafetensorReason());
        out.put("directSafetensorCaveats", report.directSafetensorCaveats());
        return out;
    }

    private Map<String, Object> tokenizerReport(ModelTokenizerDescriptor descriptor, Optional<Path> modelDir) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", descriptor.id());
        out.put("kind", descriptor.kind().name());
        out.put("requiredFileGroups", descriptor.requiredFileGroups());
        out.put("options", descriptor.options());
        out.put("fileStatusAvailable", modelDir.isPresent());
        out.put("usable", modelDir
                .map(dir -> descriptor.firstExistingFileGroup(dir).isPresent())
                .orElse(false));
        out.put("existingFileGroup", modelDir
                .flatMap(descriptor::firstExistingFileGroup)
                .map(paths -> paths.stream()
                        .map(path -> modelDir.get().relativize(path).toString())
                        .toList())
                .orElse(List.of()));
        out.put("missingFileGroups", modelDir
                .map(dir -> missingFileGroups(dir, descriptor))
                .orElse(List.of()));
        return out;
    }

    private List<String> modelFamilyProblemCodes(ModelFamilyResolution resolution, Optional<Path> modelDir) {
        LinkedHashSet<String> problemCodes = new LinkedHashSet<>(resolution.problemCodes());
        problemCodes.addAll(directArchitectureProblemCodes(resolution));
        if (modelDir.isPresent()
                && resolution.resolved()
                && !resolution.tokenizerDescriptors().isEmpty()
                && resolution.tokenizerDescriptors().stream()
                        .noneMatch(descriptor -> descriptor.firstExistingFileGroup(modelDir.get()).isPresent())) {
            problemCodes.add("model_family_tokenizer_files_missing");
        }
        return List.copyOf(problemCodes);
    }

    private List<String> modelFamilyRemediationHints(ModelFamilyResolution resolution, Optional<Path> modelDir) {
        List<String> hints = new ArrayList<>(resolution.remediationHints());
        List<String> architectureProblems = directArchitectureProblemCodes(resolution);
        if (architectureProblems.contains("model_family_architecture_adapters_missing")) {
            hints.add("Publish an architecture adapter from the matched model-family plugin or remove the direct SafeTensor capability until the adapter is ready.");
        }
        if (architectureProblems.contains("model_family_architecture_adapter_unmatched")) {
            hints.add("Update the matched architecture adapter claims for model_type="
                    + resolution.modelType() + ", architecture=" + resolution.architectureClassName() + ".");
        }
        if (modelFamilyProblemCodes(resolution, modelDir).contains("model_family_tokenizer_files_missing")) {
            String requirements = resolution.tokenizerDescriptors().stream()
                    .map(descriptor -> descriptor.id() + " requires " + descriptor.requiredFileGroups())
                    .reduce((left, right) -> left + "; " + right)
                    .orElse("no tokenizer descriptor requirements were available");
            hints.add("Add one required tokenizer file group for the matched family: " + requirements + ".");
        }
        return List.copyOf(hints);
    }

    private List<List<String>> missingFileGroups(Path modelDir, ModelTokenizerDescriptor descriptor) {
        return descriptor.requiredFileGroups().stream()
                .filter(group -> group.stream().anyMatch(relative -> !Files.exists(modelDir.resolve(relative))))
                .map(List::copyOf)
                .toList();
    }

    private Map<String, Object> directArchitectureReport(ModelFamilyResolution resolution) {
        List<ModelArchitecture> adapters = ModelFamilyPluginRegistry.global()
                .architectureAdaptersFor(resolution.modelType(), resolution.architectureClassName());
        Optional<ArchitectureSelection> selected = selectArchitectureAdapter(
                adapters,
                resolution.modelType(),
                resolution.architectureClassName());
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("directSupportExpected", directAdapterExpected(resolution));
        report.put("directSupportStatuses", resolution.supportReports().stream()
                .map(support -> support.directSafetensorStatus().name())
                .distinct()
                .toList());
        report.put("adapterIds", adapters.stream()
                .map(this::safeAdapterId)
                .filter(id -> !id.isBlank())
                .toList());
        report.put("selectedAdapterId", selected.map(ArchitectureSelection::adapterId).orElse(null));
        report.put("selectedBy", selected.map(ArchitectureSelection::selectedBy).orElse(null));
        report.put("problemCodes", directArchitectureProblemCodes(resolution));
        return report;
    }

    private List<String> directArchitectureProblemCodes(ModelFamilyResolution resolution) {
        if (!directAdapterExpected(resolution)) {
            return List.of();
        }
        List<ModelArchitecture> adapters = ModelFamilyPluginRegistry.global()
                .architectureAdaptersFor(resolution.modelType(), resolution.architectureClassName());
        if (adapters.isEmpty()) {
            return List.of("model_family_architecture_adapters_missing");
        }
        if (selectArchitectureAdapter(adapters, resolution.modelType(), resolution.architectureClassName()).isEmpty()) {
            return List.of("model_family_architecture_adapter_unmatched");
        }
        return List.of();
    }

    private boolean directAdapterExpected(ModelFamilyResolution resolution) {
        return resolution.supportReports().stream().anyMatch(report ->
                report.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.READY
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.EXPERIMENTAL
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.DECLARED_NO_ADAPTER);
    }

    private Optional<ArchitectureSelection> selectArchitectureAdapter(
            List<ModelArchitecture> adapters,
            String modelType,
            String architectureClassName) {
        for (ModelArchitecture adapter : adapters) {
            if (modelType != null && supportedModelTypes(adapter).contains(modelType)) {
                return Optional.of(new ArchitectureSelection(safeAdapterId(adapter), "model_type"));
            }
        }
        for (ModelArchitecture adapter : adapters) {
            if (architectureClassName != null && supportedArchitectureClassNames(adapter).contains(architectureClassName)) {
                return Optional.of(new ArchitectureSelection(safeAdapterId(adapter), "architecture"));
            }
        }
        return Optional.empty();
    }

    private String safeAdapterId(ModelArchitecture adapter) {
        try {
            String id = adapter.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
        }
    }

    private List<String> supportedModelTypes(ModelArchitecture adapter) {
        try {
            List<String> modelTypes = adapter.supportedModelTypes();
            return modelTypes == null ? List.of() : modelTypes;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private List<String> supportedArchitectureClassNames(ModelArchitecture adapter) {
        try {
            List<String> architectureClassNames = adapter.supportedArchClassNames();
            return architectureClassNames == null ? List.of() : architectureClassNames;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private Optional<ModelFamilyResolution> resolveModelFamily(ModelResolver.ResolvedModel resolved) {
        try {
            Optional<Path> modelDir = modelDirectory(resolved);
            if (modelDir.isEmpty() || !Files.isRegularFile(modelDir.get().resolve("config.json"))) {
                return Optional.empty();
            }
            registerModelFamilyPlugins();
            ModelConfig config = ModelConfig.load(modelDir.get().resolve("config.json"), JSON);
            return Optional.of(ModelFamilyPluginRegistry.global().resolve(config));
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Path> modelDirectory(ModelResolver.ResolvedModel resolved) {
        Path path = resolved.localPath();
        if (path == null) {
            path = ModelResolver.extractPath(resolved.info()).orElse(null);
        }
        if (path == null) {
            return Optional.empty();
        }
        if (Files.isRegularFile(path)) {
            return Optional.ofNullable(path.getParent());
        }
        return Optional.of(path);
    }

    private void registerModelFamilyPlugins() {
        if (modelFamilyPluginInstances != null && !modelFamilyPluginInstances.isUnsatisfied()) {
            for (ModelFamilyPlugin plugin : modelFamilyPluginInstances) {
                ModelFamilyPluginRegistry.global().register(plugin);
            }
        }
        ModelFamilyPluginRegistry.global().discoverServiceLoaderPlugins();
    }

    private boolean isRunnableLocally(ModelResolver.ResolvedModel resolved) {
        ModelInfo model = resolved.info();
        String format = model.getFormat() != null ? model.getFormat().trim().toUpperCase(Locale.ROOT) : "";
        if (format.equals("GGUF") || format.equals("TORCHSCRIPT") || format.equals("ONNX")) {
            return true;
        }
        if (format.equals("SAFETENSORS") || format.equals("PYTORCH") || format.equals("BIN")) {
            return true;
        }
        Path path = resolved.localPath();
        if (path == null) {
            path = ModelResolver.extractPath(model).orElse(null);
        }
        if (path != null) {
            String normalized = path.toString().replace('\\', '/').toLowerCase(Locale.ROOT);
            if (normalized.endsWith(".gguf")) {
                return true;
            }
            if (normalized.endsWith(".safetensors")
                    || normalized.endsWith(".safetensor")
                    || normalized.endsWith(".bin")
                    || normalized.endsWith(".pt")
                    || normalized.endsWith(".pth")) {
                return true;
            }
        }
        return true;
    }

    private record ArchitectureSelection(String adapterId, String selectedBy) {
    }
}
