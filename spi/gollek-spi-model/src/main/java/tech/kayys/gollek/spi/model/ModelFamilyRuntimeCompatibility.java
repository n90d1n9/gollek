package tech.kayys.gollek.spi.model;

import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Runtime-facing compatibility decision for a resolved model-family plugin.
 */
public record ModelFamilyRuntimeCompatibility(
        String runtimeId,
        ModelFamilyResolution modelFamily,
        String selectedArchitectureAdapterId,
        String selectedArchitectureAdapterBy,
        List<String> architectureAdapterIds,
        boolean tokenizerFileInspectionAvailable,
        List<String> usableTokenizerIds,
        List<String> problemCodes,
        List<String> remediationHints) {

    public static final String DIRECT_SAFETENSOR_RUNTIME = "direct_safetensor";

    public ModelFamilyRuntimeCompatibility {
        runtimeId = runtimeId == null || runtimeId.isBlank() ? "unknown" : runtimeId.trim();
        modelFamily = modelFamily == null
                ? new ModelFamilyResolution(null, null, ModelFamilyResolution.Status.NOT_FOUND,
                        List.of(), List.of(), List.of(), List.of())
                : modelFamily;
        selectedArchitectureAdapterId = selectedArchitectureAdapterId == null
                ? ""
                : selectedArchitectureAdapterId.trim();
        selectedArchitectureAdapterBy = selectedArchitectureAdapterBy == null
                ? ""
                : selectedArchitectureAdapterBy.trim();
        architectureAdapterIds = List.copyOf(architectureAdapterIds == null ? List.of() : architectureAdapterIds);
        usableTokenizerIds = List.copyOf(usableTokenizerIds == null ? List.of() : usableTokenizerIds);
        problemCodes = List.copyOf(problemCodes == null ? List.of() : problemCodes);
        remediationHints = List.copyOf(remediationHints == null ? List.of() : remediationHints);
    }

    public static ModelFamilyRuntimeCompatibility directSafetensor(
            ModelFamilyResolution resolution,
            List<ModelArchitecture> adapters) {
        return directSafetensor(resolution, adapters, null);
    }

    public static ModelFamilyRuntimeCompatibility directSafetensor(
            ModelFamilyResolution resolution,
            List<ModelArchitecture> adapters,
            Path modelDir) {
        ModelFamilyResolution resolved = resolution == null
                ? new ModelFamilyResolution(null, null, ModelFamilyResolution.Status.NOT_FOUND,
                        List.of(), List.of(), List.of(), List.of())
                : resolution;
        List<ModelArchitecture> safeAdapters = adapters == null ? List.of() : adapters.stream()
                .filter(Objects::nonNull)
                .toList();
        List<String> adapterIds = safeAdapters.stream()
                .map(ModelFamilyRuntimeCompatibility::safeAdapterId)
                .filter(id -> !id.isBlank())
                .distinct()
                .toList();
        Optional<AdapterSelection> selected = selectArchitectureAdapter(
                safeAdapters,
                resolved.modelType(),
                resolved.architectureClassName());

        LinkedHashSet<String> problems = new LinkedHashSet<>(resolved.problemCodes());
        LinkedHashSet<String> remediation = new LinkedHashSet<>(resolved.remediationHints());
        List<String> usableTokenizerIds = usableTokenizerIds(resolved, modelDir);

        if (resolved.resolved()) {
            if (!directSafetensorAdvertised(resolved)) {
                problems.add("model_family_direct_safetensor_not_advertised");
                remediation.add("Route this artifact through a non-direct runner or attach a model-family plugin that advertises direct SafeTensor support.");
            } else if (!directSafetensorReady(resolved)) {
                problems.add("model_family_direct_safetensor_not_ready");
                remediation.add("Keep this family out of direct production bundles until its direct SafeTensor status is ready.");
            }

            if (adapterIds.isEmpty()) {
                problems.add("model_family_architecture_adapters_missing");
                remediation.add("Publish an architecture adapter from the matched model-family plugin before using the direct SafeTensor runtime.");
            } else if (selected.isEmpty()) {
                problems.add("model_family_architecture_adapter_unmatched");
                remediation.add("Update the architecture adapter claims for model_type="
                        + resolved.modelType() + ", architecture=" + resolved.architectureClassName() + ".");
            }

            if (resolved.tokenizerDescriptors().isEmpty()) {
                problems.add("model_family_tokenizer_descriptors_missing");
                remediation.add("Publish at least one tokenizer descriptor for the matched model-family plugin.");
            } else if (modelDir != null && usableTokenizerIds.isEmpty()) {
                problems.add("model_family_tokenizer_files_missing");
                remediation.add("Add one required tokenizer file group for the matched family before starting text generation.");
            }

            addPendingQuantizedLoaderProblems(modelDir, problems, remediation);
        }

        return new ModelFamilyRuntimeCompatibility(
                DIRECT_SAFETENSOR_RUNTIME,
                resolved,
                selected.map(AdapterSelection::adapterId).orElse(""),
                selected.map(AdapterSelection::selectedBy).orElse(""),
                adapterIds,
                modelDir != null,
                usableTokenizerIds,
                List.copyOf(problems),
                List.copyOf(remediation));
    }

    public boolean compatible() {
        return problemCodes.isEmpty();
    }

    public boolean requiresAttention() {
        return !compatible();
    }

    public boolean architectureAdapterReady() {
        return !selectedArchitectureAdapterId.isBlank();
    }

    public boolean tokenizerReady() {
        if (modelFamily.tokenizerDescriptors().isEmpty()) {
            return false;
        }
        return !tokenizerFileInspectionAvailable || !usableTokenizerIds.isEmpty();
    }

    public String summary() {
        String status = compatible() ? "compatible" : "blocked";
        return status + " " + runtimeId + " for " + modelFamily.summary();
    }

    private static boolean directSafetensorAdvertised(ModelFamilyResolution resolution) {
        return resolution.supportReports().stream().anyMatch(report ->
                report.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                        || report.directSafetensorStatus() != ModelFamilyDirectSupport.NOT_ADVERTISED);
    }

    private static boolean directSafetensorReady(ModelFamilyResolution resolution) {
        return resolution.supportReports().stream()
                .anyMatch(report -> report.directSafetensorStatus() == ModelFamilyDirectSupport.READY);
    }

    private static List<String> usableTokenizerIds(ModelFamilyResolution resolution, Path modelDir) {
        if (modelDir == null) {
            return List.of();
        }
        return resolution.tokenizerDescriptors().stream()
                .filter(descriptor -> descriptor.firstExistingFileGroup(modelDir).isPresent())
                .map(ModelTokenizerDescriptor::id)
                .distinct()
                .toList();
    }

    private static void addPendingQuantizedLoaderProblems(
            Path modelDir,
            LinkedHashSet<String> problems,
            LinkedHashSet<String> remediation) {
        ModelFamilyQuantizedLoaderProfile pending = ModelFamilyQuantizedLoaderProfile.fromModelDir(modelDir);
        if (pending == null) {
            return;
        }

        problems.addAll(pending.problemCodes());
        remediation.add(pending.remediationHint());
    }

    private static Optional<AdapterSelection> selectArchitectureAdapter(
            List<ModelArchitecture> adapters,
            String modelType,
            String architectureClassName) {
        Set<String> modelTypes = normalizedSet(List.of(modelType));
        Set<String> architectureClassNames = trimmedSet(List.of(architectureClassName));
        for (ModelArchitecture adapter : adapters) {
            if (!modelTypes.isEmpty()
                    && safeModelTypes(adapter).stream()
                            .map(ModelFamilyRuntimeCompatibility::normalize)
                            .anyMatch(modelTypes::contains)) {
                return Optional.of(new AdapterSelection(safeAdapterId(adapter), "model_type"));
            }
        }
        for (ModelArchitecture adapter : adapters) {
            if (!architectureClassNames.isEmpty()
                    && safeArchitectureClassNames(adapter).stream()
                            .map(value -> Objects.toString(value, "").trim())
                            .anyMatch(architectureClassNames::contains)) {
                return Optional.of(new AdapterSelection(safeAdapterId(adapter), "architecture"));
            }
        }
        return Optional.empty();
    }

    private static Set<String> normalizedSet(List<String> values) {
        Set<String> normalized = new LinkedHashSet<>();
        for (String value : values) {
            String item = normalize(value);
            if (!item.isBlank()) {
                normalized.add(item);
            }
        }
        return normalized;
    }

    private static Set<String> trimmedSet(List<String> values) {
        Set<String> trimmed = new LinkedHashSet<>();
        for (String value : values) {
            String item = Objects.toString(value, "").trim();
            if (!item.isBlank()) {
                trimmed.add(item);
            }
        }
        return trimmed;
    }

    private static String safeAdapterId(ModelArchitecture adapter) {
        try {
            String id = adapter.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
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

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private record AdapterSelection(String adapterId, String selectedBy) {
    }

}
