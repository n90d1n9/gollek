package tech.kayys.gollek.spi.model;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Shared contract checks for detachable model-family plugins.
 */
public final class ModelFamilyContractValidator {

    private static final Pattern FAMILY_ID_PATTERN = Pattern.compile("[a-z][a-z0-9_]*");
    private static final Pattern MODEL_TYPE_PATTERN = Pattern.compile("[a-z][a-z0-9_.-]*");
    private static final Pattern SCOPED_DIRECT_SAFETENSOR_KEY_PATTERN =
            Pattern.compile("[a-z][a-z0-9_]*_direct_safetensor");
    private static final Pattern TOKENIZER_ID_PATTERN = Pattern.compile("[a-zA-Z0-9][a-zA-Z0-9_.:-]*");
    private static final Pattern ORIGIN_WHITESPACE_PATTERN = Pattern.compile("\\s");
    private static final String DIRECT_SAFETENSOR_KEY = "direct_safetensor";
    private static final String DIRECT_SAFETENSOR_SUFFIX = "_direct_safetensor";
    private static final String TRANSFORMERS_MODELS_ORIGIN_PREFIX =
            "3rdparty/transformers/src/transformers/models/";
    private static final String EXTERNAL_ORIGIN_PREFIX = "external/";
    private static final String LEGACY_ORIGIN_PREFIX = "legacy/";
    private static final String TOKENIZER_METADATA_STATUS_KEY = "tokenizer_metadata_status";
    private static final String TOKENIZER_METADATA_PENDING_REASON_KEY = "tokenizer_metadata_pending_reason";
    private static final String TOKENIZER_METADATA_STATUS_READY = "ready";
    private static final String TOKENIZER_METADATA_STATUS_PENDING = "pending";
    private static final Set<String> PROFILE_KEYS = Set.of(
            ModelFamilyBundleProfile.CORE.key(),
            ModelFamilyBundleProfile.OPTIONAL.key(),
            ModelFamilyBundleProfile.METADATA_ONLY.key(),
            ModelFamilyBundleProfile.EXPERIMENTAL.key());

    private ModelFamilyContractValidator() {
    }

    public static List<ModelFamilyContractViolation> validate(ModelFamilyPlugin plugin) {
        if (plugin == null) {
            return List.of(new ModelFamilyContractViolation("unknown", "plugin_null",
                    "model-family plugin instance must not be null"));
        }

        List<ModelFamilyContractViolation> violations = new ArrayList<>();
        ModelFamilyDescriptor descriptor;
        try {
            descriptor = plugin.descriptor();
        } catch (RuntimeException error) {
            return List.of(new ModelFamilyContractViolation(pluginId(plugin), "descriptor_unavailable",
                    "descriptor() threw " + error.getClass().getSimpleName() + ": " + error.getMessage()));
        }

        String familyId = descriptor == null ? pluginId(plugin) : descriptor.id();
        if (descriptor == null) {
            violations.add(new ModelFamilyContractViolation(familyId, "descriptor_null",
                    "descriptor() must not return null"));
            return List.copyOf(violations);
        }

        List<ModelTokenizerDescriptor> tokenizers = safeTokenizers(plugin, descriptor.id(), violations);
        List<ModelArchitecture> adapters = safeAdapters(plugin, descriptor.id(), violations);
        List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements =
                safeUnifiedRuntimeRequirements(plugin, descriptor.id(), violations);
        validateDescriptor(plugin, descriptor, violations);
        validateTokenizerDescriptors(descriptor, tokenizers, violations);
        validateArchitectureAdapters(descriptor, adapters, violations);
        validateUnifiedRuntimeRequirements(descriptor, unifiedRuntimeRequirements, violations);
        validateSupportReport(plugin, descriptor, adapters, tokenizers, violations);
        return List.copyOf(violations);
    }

    public static List<ModelFamilyContractViolation> validateAll(Collection<ModelFamilyPlugin> plugins) {
        if (plugins == null || plugins.isEmpty()) {
            return List.of();
        }

        List<ModelFamilyContractViolation> violations = new ArrayList<>();
        Map<String, List<String>> modelTypeClaims = new LinkedHashMap<>();
        for (ModelFamilyPlugin plugin : plugins) {
            violations.addAll(validate(plugin));
            ModelFamilyDescriptor descriptor = safeDescriptor(plugin);
            if (descriptor == null) {
                continue;
            }
            for (String modelType : descriptor.modelTypes()) {
                String normalized = normalize(modelType);
                if (normalized.isBlank()) {
                    continue;
                }
                List<String> familyIds = new ArrayList<>(modelTypeClaims.getOrDefault(normalized, List.of()));
                familyIds.add(descriptor.id());
                modelTypeClaims.put(normalized, List.copyOf(familyIds));
            }
        }

        for (Map.Entry<String, List<String>> entry : modelTypeClaims.entrySet()) {
            List<String> familyIds = entry.getValue().stream().distinct().toList();
            if (familyIds.size() > 1) {
                violations.add(new ModelFamilyContractViolation(String.join(",", familyIds),
                        "duplicate_model_type_claim",
                        "model_type '" + entry.getKey() + "' is claimed by " + String.join(", ", familyIds)));
            }
        }

        return List.copyOf(violations);
    }

    private static void validateDescriptor(
            ModelFamilyPlugin plugin,
            ModelFamilyDescriptor descriptor,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        if (!FAMILY_ID_PATTERN.matcher(familyId).matches()) {
            violations.add(new ModelFamilyContractViolation(familyId, "invalid_family_id",
                    "family id must match " + FAMILY_ID_PATTERN.pattern()));
        }

        String expectedPluginId = "model-family/" + familyId;
        String actualPluginId = pluginId(plugin);
        if (!expectedPluginId.equals(actualPluginId)) {
            violations.add(new ModelFamilyContractViolation(familyId, "plugin_id_mismatch",
                    "plugin id is '" + actualPluginId + "', expected '" + expectedPluginId + "'"));
        }

        if (descriptor.modelTypes().isEmpty() && descriptor.architectureClassNames().isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "missing_claims",
                    "descriptor must claim at least one model_type or architecture class"));
        }
        for (String modelType : descriptor.modelTypes()) {
            if (!MODEL_TYPE_PATTERN.matcher(modelType).matches()) {
                violations.add(new ModelFamilyContractViolation(familyId, "invalid_model_type",
                        "model_type '" + modelType + "' must match " + MODEL_TYPE_PATTERN.pattern()));
            }
        }

        String profile = normalizedMetadata(descriptor, ModelFamilyBundleProfile.METADATA_KEY);
        if (profile.isBlank()) {
            violations.add(new ModelFamilyContractViolation(familyId, "missing_bundle_profile",
                    "metadata.bundle_profile must be one of " + String.join(", ", PROFILE_KEYS)));
        } else if (!PROFILE_KEYS.contains(profile)) {
            violations.add(new ModelFamilyContractViolation(familyId, "unknown_bundle_profile",
                    "metadata.bundle_profile '" + profile + "' must be one of " + String.join(", ", PROFILE_KEYS)));
        }

        String origin = Objects.toString(descriptor.metadata().get("origin"), "").trim();
        if (origin.isBlank()) {
            violations.add(new ModelFamilyContractViolation(familyId, "missing_origin",
                    "metadata.origin should point to the Transformers source family"));
        } else {
            validateOriginMetadata(descriptor, origin, violations);
        }

        validateScopedDirectSafetensorMetadata(descriptor, violations);
        validateCapabilityShape(descriptor, violations);
    }

    private static void validateOriginMetadata(
            ModelFamilyDescriptor descriptor,
            String origin,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        for (String item : splitMetadata(origin)) {
            if (item.isBlank()) {
                violations.add(new ModelFamilyContractViolation(familyId, "origin_empty_segment",
                        "metadata.origin should not contain empty comma-separated segments"));
                continue;
            }
            if (ORIGIN_WHITESPACE_PATTERN.matcher(item).find()) {
                violations.add(new ModelFamilyContractViolation(familyId, "origin_contains_whitespace",
                        "metadata.origin segment '" + item + "' should not contain whitespace"));
            }
            if (item.contains("/")
                    && !item.startsWith(TRANSFORMERS_MODELS_ORIGIN_PREFIX)
                    && !item.startsWith(EXTERNAL_ORIGIN_PREFIX)
                    && !item.startsWith(LEGACY_ORIGIN_PREFIX)) {
                violations.add(new ModelFamilyContractViolation(familyId, "unexpected_origin_path",
                        "metadata.origin segment '" + item + "' should use "
                                + TRANSFORMERS_MODELS_ORIGIN_PREFIX + ", " + EXTERNAL_ORIGIN_PREFIX
                                + ", or " + LEGACY_ORIGIN_PREFIX));
            }
        }
    }

    private static void validateScopedDirectSafetensorMetadata(
            ModelFamilyDescriptor descriptor,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        descriptor.metadata().forEach((key, value) -> {
            if (key == null || !key.endsWith(DIRECT_SAFETENSOR_SUFFIX) || DIRECT_SAFETENSOR_KEY.equals(key)) {
                return;
            }
            if (!SCOPED_DIRECT_SAFETENSOR_KEY_PATTERN.matcher(key).matches()) {
                violations.add(new ModelFamilyContractViolation(familyId, "invalid_scoped_direct_safetensor_key",
                        "metadata key '" + key + "' must match "
                                + SCOPED_DIRECT_SAFETENSOR_KEY_PATTERN.pattern()));
            }
            if (!isScopedDirectSafetensorReason(value)) {
                violations.add(new ModelFamilyContractViolation(familyId, "invalid_scoped_direct_safetensor_reason",
                        "metadata." + key + " should start with pending/experimental/not or include _pending"));
            }
        });
    }

    private static void validateCapabilityShape(
            ModelFamilyDescriptor descriptor,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        EnumSet<ModelFamilyCapability> capabilities = descriptor.capabilities().isEmpty()
                ? EnumSet.noneOf(ModelFamilyCapability.class)
                : EnumSet.copyOf(descriptor.capabilities());

        if (capabilities.contains(ModelFamilyCapability.MULTIMODAL)
                && !capabilities.contains(ModelFamilyCapability.VISION)
                && !capabilities.contains(ModelFamilyCapability.AUDIO)) {
            violations.add(new ModelFamilyContractViolation(familyId, "multimodal_without_modality",
                    "MULTIMODAL should also advertise at least one concrete modality such as VISION or AUDIO"));
        }
    }

    private static void validateTokenizerDescriptors(
            ModelFamilyDescriptor descriptor,
            List<ModelTokenizerDescriptor> tokenizers,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        boolean tokenCap = descriptor.capabilities().contains(ModelFamilyCapability.TOKENIZER);
        validateTokenizerMetadataChecklist(descriptor, tokenizers, tokenCap, violations);

        if (tokenCap && tokenizers.isEmpty() && !tokenizerMetadataPending(descriptor)) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_capability_without_descriptor",
                    "TOKENIZER capability requires at least one tokenizer descriptor"));
        }
        if (!tokenCap && !tokenizers.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_descriptor_without_capability",
                    "tokenizer descriptors should be paired with TOKENIZER capability"));
        }

        Set<String> tokenizerIds = new LinkedHashSet<>();
        for (ModelTokenizerDescriptor tokenizer : tokenizers) {
            if (tokenizer == null) {
                violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_descriptor_null",
                        "tokenizerDescriptors() must not contain null entries"));
                continue;
            }
            if (!TOKENIZER_ID_PATTERN.matcher(tokenizer.id()).matches()) {
                violations.add(new ModelFamilyContractViolation(familyId, "invalid_tokenizer_id",
                        "tokenizer id '" + tokenizer.id() + "' must match " + TOKENIZER_ID_PATTERN.pattern()));
            }
            if (!tokenizerIds.add(tokenizer.id())) {
                violations.add(new ModelFamilyContractViolation(familyId, "duplicate_tokenizer_id",
                        "tokenizer id '" + tokenizer.id() + "' is declared more than once"));
            }
            validateTokenizerFileGroups(familyId, tokenizer, violations);
        }
    }

    private static void validateTokenizerMetadataChecklist(
            ModelFamilyDescriptor descriptor,
            List<ModelTokenizerDescriptor> tokenizers,
            boolean tokenCap,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        String status = normalize(descriptor.metadata().get(TOKENIZER_METADATA_STATUS_KEY));
        String pendingReason = Objects.toString(
                descriptor.metadata().get(TOKENIZER_METADATA_PENDING_REASON_KEY), "").trim();
        if (status.isBlank()) {
            if (!pendingReason.isBlank()) {
                violations.add(new ModelFamilyContractViolation(familyId,
                        "tokenizer_metadata_pending_reason_without_pending_status",
                        "metadata." + TOKENIZER_METADATA_PENDING_REASON_KEY
                                + " should only be declared with tokenizer_metadata_status=pending"));
            }
            return;
        }

        if (!TOKENIZER_METADATA_STATUS_READY.equals(status)
                && !TOKENIZER_METADATA_STATUS_PENDING.equals(status)) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_metadata_status_invalid",
                    "metadata." + TOKENIZER_METADATA_STATUS_KEY + " must be ready or pending when declared"));
            return;
        }
        if (!tokenCap) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_metadata_without_capability",
                    "metadata." + TOKENIZER_METADATA_STATUS_KEY + " should be paired with TOKENIZER capability"));
        }
        if (TOKENIZER_METADATA_STATUS_READY.equals(status) && tokenizers.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_metadata_ready_without_descriptor",
                    "tokenizer metadata status ready requires at least one tokenizer descriptor"));
        }
        if (TOKENIZER_METADATA_STATUS_PENDING.equals(status) && !tokenizers.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_metadata_pending_with_descriptor",
                    "tokenizer metadata status pending should omit tokenizer descriptors until reusable metadata is ready"));
        }
        if (TOKENIZER_METADATA_STATUS_PENDING.equals(status) && pendingReason.isBlank()) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_metadata_pending_reason_missing",
                    "metadata." + TOKENIZER_METADATA_PENDING_REASON_KEY
                            + " is required when tokenizer_metadata_status=pending"));
        }
        if (!TOKENIZER_METADATA_STATUS_PENDING.equals(status) && !pendingReason.isBlank()) {
            violations.add(new ModelFamilyContractViolation(familyId,
                    "tokenizer_metadata_pending_reason_without_pending_status",
                    "metadata." + TOKENIZER_METADATA_PENDING_REASON_KEY
                            + " should only be declared with tokenizer_metadata_status=pending"));
        }
    }

    private static void validateTokenizerFileGroups(
            String familyId,
            ModelTokenizerDescriptor tokenizer,
            List<ModelFamilyContractViolation> violations) {
        if (tokenizer.kind() != ModelTokenizerKind.CUSTOM && tokenizer.requiredFileGroups().isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_missing_file_groups",
                    "tokenizer '" + tokenizer.id() + "' should declare required file groups"));
        }

        for (List<String> group : tokenizer.requiredFileGroups()) {
            if (group == null || group.isEmpty()) {
                violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_empty_file_group",
                        "tokenizer '" + tokenizer.id() + "' has an empty required file group"));
                continue;
            }
            for (String file : group) {
                String item = Objects.toString(file, "").trim();
                if (item.isBlank()) {
                    violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_blank_file",
                            "tokenizer '" + tokenizer.id() + "' has a blank required file"));
                    continue;
                }
                Path path = Path.of(item);
                if (path.isAbsolute() || item.startsWith("../") || item.contains("/../")) {
                    violations.add(new ModelFamilyContractViolation(familyId, "tokenizer_unsafe_file",
                            "tokenizer '" + tokenizer.id() + "' has unsafe required file '" + item + "'"));
                }
            }
        }
    }

    private static void validateArchitectureAdapters(
            ModelFamilyDescriptor descriptor,
            List<ModelArchitecture> adapters,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        boolean directCap = descriptor.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE);
        if (directCap && adapters.isEmpty()) {
            violations.add(new ModelFamilyContractViolation(familyId, "direct_safetensor_without_adapter",
                    "DIRECT_SAFETENSOR_INFERENCE requires at least one architecture adapter"));
        }

        Set<String> adapterIds = new LinkedHashSet<>();
        for (ModelArchitecture adapter : adapters) {
            if (adapter == null) {
                violations.add(new ModelFamilyContractViolation(familyId, "architecture_adapter_null",
                        "architectureAdapters() must not contain null entries"));
                continue;
            }
            String adapterId = Objects.toString(adapter.id(), "").trim();
            if (adapterId.isBlank()) {
                violations.add(new ModelFamilyContractViolation(familyId, "architecture_adapter_blank_id",
                        "architecture adapter id must not be blank"));
            } else if (!adapterIds.add(adapterId)) {
                violations.add(new ModelFamilyContractViolation(familyId, "duplicate_architecture_adapter_id",
                        "architecture adapter id '" + adapterId + "' is declared more than once"));
            }

            if (directCap && !adapterMatchesDescriptorClaims(adapter, descriptor)) {
                violations.add(new ModelFamilyContractViolation(familyId, "architecture_adapter_unclaimed",
                        "architecture adapter '" + adapterId + "' should support at least one descriptor model_type "
                                + "or architecture class"));
            }
        }
    }

    private static boolean adapterMatchesDescriptorClaims(ModelArchitecture adapter, ModelFamilyDescriptor descriptor) {
        Set<String> descriptorModelTypes = descriptor.modelTypes().stream()
                .map(ModelFamilyContractValidator::normalize)
                .filter(type -> !type.isBlank())
                .collect(java.util.stream.Collectors.toSet());
        Set<String> descriptorArchitectures = descriptor.architectureClassNames().stream()
                .map(ModelFamilyContractValidator::normalize)
                .filter(architecture -> !architecture.isBlank())
                .collect(java.util.stream.Collectors.toSet());

        return safeAdapterModelTypes(adapter).stream()
                .map(ModelFamilyContractValidator::normalize)
                .anyMatch(descriptorModelTypes::contains)
                || safeAdapterArchitectureClassNames(adapter).stream()
                        .map(ModelFamilyContractValidator::normalize)
                        .anyMatch(descriptorArchitectures::contains);
    }

    private static List<String> safeAdapterModelTypes(ModelArchitecture adapter) {
        try {
            List<String> modelTypes = adapter.supportedModelTypes();
            return modelTypes == null ? List.of() : modelTypes;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> safeAdapterArchitectureClassNames(ModelArchitecture adapter) {
        try {
            List<String> architectures = adapter.supportedArchClassNames();
            return architectures == null ? List.of() : architectures;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<ModelTokenizerDescriptor> safeTokenizers(
            ModelFamilyPlugin plugin,
            String familyId,
            List<ModelFamilyContractViolation> violations) {
        try {
            List<ModelTokenizerDescriptor> tokenizers = plugin.tokenizerDescriptors();
            return tokenizers == null ? List.of() : new ArrayList<>(tokenizers);
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "tokenizers_unavailable",
                    "tokenizerDescriptors() threw " + error.getClass().getSimpleName() + ": " + error.getMessage()));
            return List.of();
        }
    }

    private static List<ModelArchitecture> safeAdapters(
            ModelFamilyPlugin plugin,
            String familyId,
            List<ModelFamilyContractViolation> violations) {
        try {
            List<ModelArchitecture> adapters = plugin.architectureAdapters();
            return adapters == null ? List.of() : new ArrayList<>(adapters);
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "architecture_adapters_unavailable",
                    "architectureAdapters() threw " + error.getClass().getSimpleName() + ": " + error.getMessage()));
            return List.of();
        }
    }

    private static List<ModelFamilyUnifiedRuntimeRequirement> safeUnifiedRuntimeRequirements(
            ModelFamilyPlugin plugin,
            String familyId,
            List<ModelFamilyContractViolation> violations) {
        try {
            List<ModelFamilyUnifiedRuntimeRequirement> requirements = plugin.unifiedRuntimeRequirements();
            return requirements == null ? List.of() : new ArrayList<>(requirements);
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "unified_runtime_requirements_unavailable",
                    "unifiedRuntimeRequirements() threw " + error.getClass().getSimpleName() + ": "
                            + error.getMessage()));
            return List.of();
        }
    }

    private static void validateUnifiedRuntimeRequirements(
            ModelFamilyDescriptor descriptor,
            List<ModelFamilyUnifiedRuntimeRequirement> requirements,
            List<ModelFamilyContractViolation> violations) {
        if (requirements.isEmpty()) {
            return;
        }

        String familyId = descriptor.id();
        Set<String> claimedModelTypes = normalizedStringSet(descriptor.modelTypes());
        Set<String> metadataModelTypes = new LinkedHashSet<>();
        metadataModelTypes.addAll(splitMetadata(descriptor.metadata().get("unified_model_type")));
        metadataModelTypes.addAll(splitMetadata(descriptor.metadata().get("unified_model_types")));

        if (metadataModelTypes.isEmpty()
                && normalize(descriptor.metadata().get("unified_runtime")).isBlank()
                && normalize(descriptor.metadata().get("unified_runtime_reason")).isBlank()) {
            violations.add(new ModelFamilyContractViolation(familyId, "unified_runtime_requirement_missing_metadata",
                    "unified runtime requirements should be mirrored in descriptor metadata for discovery"));
        }

        for (ModelFamilyUnifiedRuntimeRequirement requirement : requirements) {
            if (requirement == null) {
                violations.add(new ModelFamilyContractViolation(familyId, "unified_runtime_requirement_null",
                        "unifiedRuntimeRequirements() must not contain null entries"));
                continue;
            }

            String modelType = normalize(requirement.modelType());
            if ("unknown".equals(modelType)) {
                violations.add(new ModelFamilyContractViolation(familyId,
                        "unified_runtime_requirement_unknown_model_type",
                        "unified runtime requirement modelType must not be blank or unknown"));
            } else if (!MODEL_TYPE_PATTERN.matcher(modelType).matches()) {
                violations.add(new ModelFamilyContractViolation(familyId,
                        "invalid_unified_runtime_requirement_model_type",
                        "unified runtime requirement modelType '" + modelType + "' must match "
                                + MODEL_TYPE_PATTERN.pattern()));
            }

            if (!claimedModelTypes.contains(modelType) && !metadataModelTypes.contains(modelType)) {
                violations.add(new ModelFamilyContractViolation(familyId,
                        "unified_runtime_requirement_unclaimed_model_type",
                        "unified runtime requirement modelType '" + modelType
                                + "' should be claimed by descriptor.modelTypes or unified_model_type metadata"));
            }

            if (requirement.requiredInputModalities().isEmpty()) {
                violations.add(new ModelFamilyContractViolation(familyId,
                        "unified_runtime_requirement_missing_modalities",
                        "unified runtime requirement should declare required input modalities"));
            }
        }
    }

    private static void validateSupportReport(
            ModelFamilyPlugin plugin,
            ModelFamilyDescriptor descriptor,
            List<ModelArchitecture> adapters,
            List<ModelTokenizerDescriptor> tokenizers,
            List<ModelFamilyContractViolation> violations) {
        String familyId = descriptor.id();
        ModelFamilySupportReport report;
        try {
            report = plugin.supportReport();
            if (report == null) {
                violations.add(new ModelFamilyContractViolation(familyId, "support_report_null",
                        "supportReport() must not return null"));
                return;
            }
        } catch (RuntimeException error) {
            violations.add(new ModelFamilyContractViolation(familyId, "support_report_unavailable",
                    "supportReport() threw " + error.getClass().getSimpleName() + ": " + error.getMessage()));
            return;
        }

        ModelFamilySupportReport expected = expectedSupportReport(descriptor, adapters, tokenizers);
        if (!Objects.equals(report.id(), expected.id())) {
            addSupportReportMismatch(familyId, violations, "support_report_id_mismatch",
                    "id", expected.id(), report.id());
        }
        if (!Objects.equals(report.displayName(), expected.displayName())) {
            addSupportReportMismatch(familyId, violations, "support_report_display_name_mismatch",
                    "displayName", expected.displayName(), report.displayName());
        }
        if (!normalizedStringSet(report.modelTypes()).equals(normalizedStringSet(expected.modelTypes()))) {
            addSupportReportMismatch(familyId, violations, "support_report_model_types_mismatch",
                    "modelTypes", expected.modelTypes(), report.modelTypes());
        }
        if (!trimmedStringSet(report.architectureClassNames())
                .equals(trimmedStringSet(expected.architectureClassNames()))) {
            addSupportReportMismatch(familyId, violations, "support_report_architectures_mismatch",
                    "architectureClassNames", expected.architectureClassNames(), report.architectureClassNames());
        }
        if (!new LinkedHashSet<>(report.architectureAdapterIds())
                .equals(new LinkedHashSet<>(expected.architectureAdapterIds()))) {
            addSupportReportMismatch(familyId, violations, "support_report_architecture_adapters_mismatch",
                    "architectureAdapterIds", expected.architectureAdapterIds(), report.architectureAdapterIds());
        }
        if (!new LinkedHashSet<>(report.tokenizerProfileIds())
                .equals(new LinkedHashSet<>(expected.tokenizerProfileIds()))) {
            addSupportReportMismatch(familyId, violations, "support_report_tokenizers_mismatch",
                    "tokenizerProfileIds", expected.tokenizerProfileIds(), report.tokenizerProfileIds());
        }
        if (!new LinkedHashSet<>(report.tokenizerKinds())
                .equals(new LinkedHashSet<>(expected.tokenizerKinds()))) {
            addSupportReportMismatch(familyId, violations, "support_report_tokenizer_kinds_mismatch",
                    "tokenizerKinds", expected.tokenizerKinds(), report.tokenizerKinds());
        }
        if (!new LinkedHashSet<>(report.capabilities()).equals(new LinkedHashSet<>(expected.capabilities()))) {
            addSupportReportMismatch(familyId, violations, "support_report_capabilities_mismatch",
                    "capabilities", expected.capabilities(), report.capabilities());
        }
        if (report.bundleProfile() != expected.bundleProfile()) {
            addSupportReportMismatch(familyId, violations, "support_report_bundle_profile_mismatch",
                    "bundleProfile", expected.bundleProfile(), report.bundleProfile());
        }
        if (report.directSafetensorStatus() != expected.directSafetensorStatus()
                || !Objects.equals(report.directSafetensorReason(), expected.directSafetensorReason())
                || !Objects.equals(report.directSafetensorCaveats(), expected.directSafetensorCaveats())) {
            addSupportReportMismatch(familyId, violations, "support_report_direct_safetensor_mismatch",
                    "directSafetensor", expected.shortDirectSafetensorSummary(),
                    report.shortDirectSafetensorSummary());
        }
    }

    private static ModelFamilySupportReport expectedSupportReport(
            ModelFamilyDescriptor descriptor,
            List<ModelArchitecture> adapters,
            List<ModelTokenizerDescriptor> tokenizers) {
        return ModelFamilySupportReport.from(new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return descriptor;
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return adapters;
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return tokenizers;
            }
        });
    }

    private static void addSupportReportMismatch(
            String familyId,
            List<ModelFamilyContractViolation> violations,
            String code,
            String fieldName,
            Object expected,
            Object actual) {
        violations.add(new ModelFamilyContractViolation(familyId, code,
                "supportReport()." + fieldName + " is '" + actual + "', expected '" + expected + "'"));
    }

    private static Set<String> normalizedStringSet(List<String> values) {
        if (values == null || values.isEmpty()) {
            return Set.of();
        }
        return values.stream()
                .map(ModelFamilyContractValidator::normalize)
                .filter(value -> !value.isBlank())
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    }

    private static Set<String> trimmedStringSet(List<String> values) {
        if (values == null || values.isEmpty()) {
            return Set.of();
        }
        return values.stream()
                .map(value -> Objects.toString(value, "").trim())
                .filter(value -> !value.isBlank())
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    }

    private static ModelFamilyDescriptor safeDescriptor(ModelFamilyPlugin plugin) {
        try {
            return plugin == null ? null : plugin.descriptor();
        } catch (RuntimeException error) {
            return null;
        }
    }

    private static String pluginId(ModelFamilyPlugin plugin) {
        try {
            return normalize(plugin == null ? "" : plugin.id());
        } catch (RuntimeException error) {
            return "unknown";
        }
    }

    private static String normalizedMetadata(ModelFamilyDescriptor descriptor, String key) {
        return normalize(descriptor.metadata().get(key)).replace('-', '_');
    }

    private static boolean tokenizerMetadataPending(ModelFamilyDescriptor descriptor) {
        return TOKENIZER_METADATA_STATUS_PENDING.equals(
                normalize(descriptor.metadata().get(TOKENIZER_METADATA_STATUS_KEY)));
    }

    private static List<String> splitMetadata(String value) {
        if (value == null || value.isBlank()) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (String item : value.split(",", -1)) {
            values.add(Objects.toString(item, "").trim().toLowerCase(Locale.ROOT));
        }
        return List.copyOf(values);
    }

    private static boolean isScopedDirectSafetensorReason(String value) {
        String normalized = normalize(value);
        return normalized.startsWith("pending")
                || normalized.startsWith("experimental")
                || normalized.startsWith("not")
                || normalized.endsWith("_pending")
                || normalized.contains("_pending_");
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}
