package tech.kayys.gollek.spi.model;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Compact runtime manifest for one attached model-family adapter.
 */
public record ModelFamilyRuntimeManifest(
        String familyId,
        String displayName,
        List<String> modelTypes,
        List<String> architectureClassNames,
        List<String> architectureAdapterIds,
        List<ModelTokenizerDescriptor> tokenizerDescriptors,
        List<String> tokenizerProfileIds,
        List<ModelTokenizerKind> tokenizerKinds,
        List<String> chatTemplateIds,
        ModelFamilyBundleProfile bundleProfile,
        List<ModelFamilyCapability> capabilities,
        ModelFamilyDirectSupport directSafetensorStatus,
        String directSafetensorReason,
        Map<String, String> directSafetensorCaveats,
        List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements,
        Map<String, String> metadata) {

    public ModelFamilyRuntimeManifest {
        familyId = familyId == null || familyId.isBlank() ? "unknown" : familyId.trim();
        displayName = displayName == null || displayName.isBlank() ? familyId : displayName.trim();
        modelTypes = copyStrings(modelTypes);
        architectureClassNames = copyStrings(architectureClassNames);
        architectureAdapterIds = copyStrings(architectureAdapterIds);
        tokenizerDescriptors = tokenizerDescriptors == null
                ? List.of()
                : tokenizerDescriptors.stream()
                        .filter(Objects::nonNull)
                        .toList();
        tokenizerProfileIds = copyStrings(tokenizerProfileIds);
        tokenizerKinds = tokenizerKinds == null ? List.of() : List.copyOf(tokenizerKinds);
        chatTemplateIds = copyStrings(chatTemplateIds);
        bundleProfile = bundleProfile == null ? ModelFamilyBundleProfile.OPTIONAL : bundleProfile;
        capabilities = capabilities == null ? List.of() : List.copyOf(capabilities);
        directSafetensorStatus = directSafetensorStatus == null
                ? ModelFamilyDirectSupport.NOT_ADVERTISED
                : directSafetensorStatus;
        directSafetensorReason = directSafetensorReason == null || directSafetensorReason.isBlank()
                ? directSafetensorStatus.label()
                : directSafetensorReason.trim();
        directSafetensorCaveats = directSafetensorCaveats == null ? Map.of() : Map.copyOf(directSafetensorCaveats);
        unifiedRuntimeRequirements = unifiedRuntimeRequirements == null
                ? List.of()
                : unifiedRuntimeRequirements.stream()
                        .filter(Objects::nonNull)
                        .toList();
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public static ModelFamilyRuntimeManifest from(ModelFamilyRuntimeAdapter adapter) {
        ModelFamilySupportReport support = ModelFamilySupportReport.from(adapter);
        List<ModelTokenizerDescriptor> tokenizers = safeTokenizers(adapter);
        return new ModelFamilyRuntimeManifest(
                support.id(),
                support.displayName(),
                support.modelTypes(),
                support.architectureClassNames(),
                support.architectureAdapterIds(),
                tokenizers,
                support.tokenizerProfileIds(),
                support.tokenizerKinds(),
                safeChatTemplateIds(adapter),
                support.bundleProfile(),
                support.capabilities(),
                support.directSafetensorStatus(),
                support.directSafetensorReason(),
                support.directSafetensorCaveats(),
                safeUnifiedRuntimeRequirements(adapter),
                support.metadata());
    }

    public boolean tokenizerReady() {
        return !tokenizerDescriptors.isEmpty();
    }

    public boolean chatTemplateReady() {
        return !chatTemplateIds.isEmpty();
    }

    public boolean directSafetensorReady() {
        return directSafetensorStatus.ready();
    }

    public boolean requiresUnifiedRuntime() {
        return !unifiedRuntimeRequirements.isEmpty();
    }

    private static List<ModelFamilyUnifiedRuntimeRequirement> safeUnifiedRuntimeRequirements(
            ModelFamilyRuntimeAdapter adapter) {
        try {
            List<ModelFamilyUnifiedRuntimeRequirement> requirements = adapter.unifiedRuntimeRequirements();
            return requirements == null ? List.of() : requirements.stream()
                    .filter(Objects::nonNull)
                    .toList();
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<ModelTokenizerDescriptor> safeTokenizers(ModelFamilyRuntimeAdapter adapter) {
        try {
            List<ModelTokenizerDescriptor> tokenizers = adapter.tokenizerDescriptors();
            return tokenizers == null ? List.of() : tokenizers.stream()
                    .filter(Objects::nonNull)
                    .toList();
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> safeChatTemplateIds(ModelFamilyRuntimeAdapter adapter) {
        try {
            List<String> templates = adapter.chatTemplateIds();
            return copyStrings(templates);
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> copyStrings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> Objects.toString(value, "").trim())
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
    }
}
