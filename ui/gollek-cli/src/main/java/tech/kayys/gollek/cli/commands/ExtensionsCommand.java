package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.plugin.kernel.KernelPlatformDetector;
import tech.kayys.gollek.cli.util.ModelFamilyBundleManifest;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.spi.model.ModelFamilyCapabilityMatrixEntry;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Show built-in modules, runner plugins, and kernel platform info.
 * Provides a complete view of the multilevel plugin system.
 */
@Dependent
@Unremovable
@Command(
        name = "modules",
        aliases = { "components", "runtime-extensions" },
        description = "Show packaged runtime modules, built-in plugins, and kernel info")
public class ExtensionsCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private static final int JSON_SCHEMA_VERSION = 1;

    private record ExtensionDef(String type, String name, String profile, String markerClass, String providerId) {
    }

    private static final List<ExtensionDef> EXTENSIONS = List.of(
            // Local runtimes
            new ExtensionDef("runtime", "GGUF", "base", "tech.kayys.gollek.inference.llamacpp.LlamaCppProvider", "gguf"),
            new ExtensionDef("runtime", "ONNX", "base", "tech.kayys.gollek.onnx.runner.OnnxRuntimeRunner", "onnx"),
            new ExtensionDef("runtime", "SafeTensor", "base",
                    "tech.kayys.gollek.safetensor.engine.warmup.SafetensorProvider", "safetensor"),
            new ExtensionDef("runtime", "LibTorch", "experimental",
                    "tech.kayys.gollek.inference.libtorch.LibTorchProvider", "libtorch"),
            new ExtensionDef("runtime", "LiteRT", "optional",
                    "tech.kayys.gollek.provider.litert.LiteRTProvider", "litert"),
            new ExtensionDef("runtime", "TensorRT", "optional",
                    "tech.kayys.gollek.inference.tensorrt.TensorRTProvider", "tensorrt"),

            // Cloud providers
            new ExtensionDef("cloud", "Gemini", "ext-cloud-gemini",
                    "tech.kayys.gollek.provider.gemini.GeminiProvider", "gemini"),
            new ExtensionDef("cloud", "Cerebras", "ext-cloud-cerebras",
                    "tech.kayys.gollek.provider.cerebras.CerebrasProvider", "cerebras"),
            new ExtensionDef("cloud", "Mistral", "ext-cloud-mistral",
                    "tech.kayys.gollek.provider.mistral.MistralProvider", "mistral"),
            new ExtensionDef("cloud", "OpenAI", "optional",
                    "tech.kayys.gollek.provider.openai.OpenAiProvider", "openai"),
            new ExtensionDef("cloud", "Anthropic", "optional",
                    "tech.kayys.gollek.provider.anthropic.AnthropicProvider", "anthropic"),

            // Tool/integration providers
            new ExtensionDef("tool", "MCP", "base",
                    "tech.kayys.gollek.provider.mcp.McpProvider", "mcp"));

    @Inject
    GollekSdk sdk;

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    @Inject
    Instance<ModelFamilyPlugin> modelFamilyPluginInstances;

    @Option(names = { "-a", "--all" }, description = "Show missing runtime modules too")
    boolean showAll;

    @Option(names = { "--json" }, description = "Print a machine-readable module report")
    boolean jsonOutput;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    @Override
    public void run() {
        if (jsonOutput) {
            printJsonReport();
            return;
        }

        // === Section 1: Kernel Platform ===
        printKernelInfo();
        System.out.println();

        // === Section 2: Runtime Modules ===
        printExtensions();
        System.out.println();

        // === Section 3: Runner Plugins ===
        printRunnerPlugins();
        System.out.println();

        // === Section 4: Model Families ===
        printModelFamilies();
        System.out.println();

        // === Section 5: Dynamic Plugins ===
        printDynamicPlugins();
    }

    private void printJsonReport() {
        try {
            System.out.println(JSON.writeValueAsString(buildJsonReport()));
        } catch (Exception e) {
            throw new IllegalStateException("Failed to render modules JSON report", e);
        }
    }

    private Map<String, Object> buildJsonReport() {
        Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins();
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schemaVersion", JSON_SCHEMA_VERSION);
        report.put("showAll", showAll);
        report.put("kernel", kernelReport());
        report.put("runtimeModules", runtimeModuleReports());
        report.put("runnerPlugins", runnerPluginSectionReport());
        report.put("modelFamilyBundle", modelFamilyBundleReport(manifest, families));
        report.put("modelFamilyPlugins", modelFamilyPluginSectionReport(families));
        report.put("dynamicPlugins", dynamicPluginSectionReport());
        return report;
    }

    private Map<String, Object> kernelReport() {
        KernelPlatform platform = KernelPlatformDetector.detect();

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("platform", platform.name().toLowerCase());
        report.put("displayName", platform.getDisplayName());
        report.put("description", platform.getDescription());
        report.put("gpuAccelerated", platform.isGpu());
        report.put("architecture", System.getProperty("os.arch", "unknown"));
        report.put("osName", System.getProperty("os.name", "unknown"));
        report.put("osVersion", System.getProperty("os.version", ""));
        report.put("javaVersion", System.getProperty("java.version", "unknown"));
        report.put("javaVendor", System.getProperty("java.vendor", "unknown"));
        return report;
    }

    private List<Map<String, Object>> runtimeModuleReports() {
        Set<String> runtimeProviders = getRuntimeProviderIds();
        return EXTENSIONS.stream()
                .filter(extension -> showAll || isClassPresent(extension.markerClass()))
                .map(extension -> runtimeModuleReport(extension, runtimeProviders))
                .toList();
    }

    private Map<String, Object> runtimeModuleReport(ExtensionDef extension, Set<String> runtimeProviders) {
        boolean packaged = isClassPresent(extension.markerClass());

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("type", extension.type());
        report.put("name", extension.name());
        report.put("profile", extension.profile());
        report.put("packaged", packaged);
        report.put("providerAvailable", runtimeProviders.contains(extension.providerId()));
        report.put("providerId", extension.providerId());
        report.put("markerClass", extension.markerClass());
        return report;
    }

    private Map<String, Object> runnerPluginSectionReport() {
        Map<String, Object> section = new LinkedHashMap<>();
        try {
            section.put("plugins", collectRunnerPlugins().values().stream()
                    .map(this::runnerPluginReport)
                    .toList());
            section.put("error", null);
        } catch (Exception e) {
            section.put("plugins", List.of());
            section.put("error", e.getMessage());
        }
        return section;
    }

    private Map<String, Object> runnerPluginReport(RunnerPlugin plugin) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", plugin.id());
        report.put("name", plugin.name());
        report.put("version", plugin.version());
        report.put("format", plugin.format());
        report.put("status", "active");
        report.put("priority", plugin.priority());
        report.put("supportedFormats", plugin.supportedFormats());
        report.put("supportedArchitectures", plugin.supportedArchitectures());
        return report;
    }

    private Map<String, Object> modelFamilyBundleReport(ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("present", manifest.present());
        report.put("schemaVersion", manifest.schemaVersion());
        report.put("fingerprint", manifest.bundleFingerprint());
        report.put("detached", manifest.detached());
        report.put("bundlePreset", manifest.hasBundlePreset() ? manifest.bundlePreset() : null);
        report.put("availableBundlePresets", manifest.availableBundlePresets());
        report.put("activeBundlePreset", manifest.activeBundlePreset()
                .map(this::modelFamilyBundlePresetReport)
                .orElse(null));
        report.put("activeBundlePresetConformance",
                modelFamilyBundlePresetConformanceReport(manifest.activeBundlePresetConformance()));
        report.put("selectorSource", manifest.selectorSource());
        report.put("explicitSelectors", manifest.explicitSelectors());
        report.put("presetSelectors", manifest.presetSelectors());
        report.put("defaultSelectors", manifest.defaultSelectors());
        report.put("policySource", manifest.policySource());
        report.put("presetRequiredFamilies", manifest.presetRequiredFamilies());
        report.put("presetForbiddenFamilies", manifest.presetForbiddenFamilies());
        report.put("presetRequiredAliases", manifest.presetRequiredAliases());
        report.put("presetForbiddenAliases", manifest.presetForbiddenAliases());
        report.put("explicitRequiredFamilies", manifest.explicitRequiredFamilies());
        report.put("explicitForbiddenFamilies", manifest.explicitForbiddenFamilies());
        report.put("explicitRequiredAliases", manifest.explicitRequiredAliases());
        report.put("explicitForbiddenAliases", manifest.explicitForbiddenAliases());
        report.put("requiredFamilies", manifest.bundlePolicy().requiredFamilies());
        report.put("forbiddenFamilies", manifest.bundlePolicy().forbiddenFamilies());
        report.put("requiredAliases", manifest.bundlePolicy().requiredAliases());
        report.put("forbiddenAliases", manifest.bundlePolicy().forbiddenAliases());
        report.put("policyStatus", modelFamilyBundlePolicyStatusReport(manifest.bundlePolicy()));
        report.put("policyViolations", modelFamilyBundlePolicyViolationsReport(manifest.bundlePolicy().violations()));
        report.put("fixtureStatus", modelFamilyBundleFixtureStatusReport(manifest.fixtureStatus()));
        report.put("availabilityStatus", modelFamilyBundleAvailabilityReport(
                PluginAvailabilityChecker.modelFamilyBundleAvailability(manifest, discoveredFamilies.keySet())));
        report.put("selectors", manifest.selectors());
        report.put("requestedFamilies", manifest.requestedFamilies());
        report.put("requestedProfiles", manifest.requestedProfiles());
        report.put("requestedAliases", manifest.requestedAliases());
        report.put("reservedSelectors", manifest.reservedSelectors());
        report.put("unknownSelectors", manifest.unknownSelectors());
        report.put("families", manifest.families());
        report.put("profiles", manifest.profiles());
        report.put("availableFamilies", manifest.availableFamilies());
        report.put("availableProfiles", manifest.availableProfiles());
        report.put("availableSelectors", manifest.availableSelectors());
        report.put("bundlePresets", manifest.bundlePresets().stream()
                .map(this::modelFamilyBundlePresetReport)
                .toList());
        report.put("bundleAliases", manifest.bundleAliasCoverage().stream()
                .map(this::modelFamilyBundleAliasReport)
                .toList());
        report.put("completeAliases", manifest.completeBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::id)
                .toList());
        report.put("partialAliases", manifest.partialBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::id)
                .toList());
        report.put("omittedFamilies", manifest.omittedFamilies());
        report.put("omittedFamiliesWithProfiles", manifest.omittedFamiliesWithProfiles());
        report.put("missingDiscovered", manifest.missingDiscovered(discoveredFamilies.keySet()));
        report.put("selection", modelFamilySelectionReport(manifest, discoveredFamilies));
        return report;
    }

    private Map<String, Object> modelFamilyBundleFixtureStatusReport(
            ModelFamilyBundleManifest.FixtureStatus fixtureStatus) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("known", fixtureStatus.statusKnown());
        report.put("passed", fixtureStatus.passed());
        report.put("status", fixtureStatus.statusLabel());
        report.put("summary", fixtureStatus.compactStatus());
        report.put("requiredSelectors", fixtureStatus.requiredSelectors());
        report.put("requiredFamilies", fixtureStatus.requiredFamilies());
        report.put("requiredFingerprint", fixtureStatus.requiredFingerprint());
        report.put("inventoryFingerprint", fixtureStatus.inventoryFingerprint());
        report.put("availableFamilyCount", fixtureStatus.availableFamilyCount());
        report.put("fixtureFamilyCount", fixtureStatus.fixtureFamilyCount());
        report.put("requiredFamilyCount", fixtureStatus.requiredFamilyCount());
        report.put("requiredPassedCount", fixtureStatus.requiredPassedCount());
        report.put("missingRequiredCount", fixtureStatus.missingRequiredCount());
        report.put("problemFamilyCount", fixtureStatus.problemFamilyCount());
        report.put("missingRequiredFamilies", fixtureStatus.missingRequiredFamilies());
        report.put("problemFamilies", fixtureStatus.problemFamilies());
        return report;
    }

    private Map<String, Object> modelFamilyBundleAvailabilityReport(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("present", availability.present());
        report.put("detached", availability.detached());
        report.put("healthy", availability.healthy());
        report.put("status", availability.status());
        report.put("summary", availability.compactSummary());
        report.put("selectedFamilyCount", availability.selectedFamilyCount());
        report.put("discoveredSelectedFamilyCount", availability.discoveredSelectedFamilyCount());
        report.put("missingSelectedFamilyCount", availability.missingSelectedFamilyCount());
        report.put("omittedFamilyCount", availability.omittedFamilyCount());
        report.put("policyStatus", availability.policyStatus());
        report.put("policyViolationCount", availability.policyViolationCount());
        report.put("fixtureStatus", availability.fixtureStatus());
        report.put("fixturePassed", availability.fixturePassed());
        report.put("fixtureMissingRequiredCount", availability.fixtureMissingRequiredCount());
        report.put("fixtureProblemFamilyCount", availability.fixtureProblemFamilyCount());
        report.put("presetConformanceStatus", availability.presetConformanceStatus());
        report.put("problems", availability.problems());
        report.put("remediationHints", availability.remediationHints());
        report.put("missingSelectedFamilies", availability.missingSelectedFamilies());
        report.put("omittedFamilies", availability.omittedFamilies());
        report.put("fixtureMissingRequiredFamilies", availability.fixtureMissingRequiredFamilies());
        report.put("fixtureProblemFamilies", availability.fixtureProblemFamilies());
        return report;
    }

    private Map<String, Object> modelFamilyBundlePresetReport(ModelFamilyBundleManifest.BundlePreset preset) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", preset.id());
        report.put("description", preset.description());
        report.put("selectors", preset.selectors());
        report.put("requiredFamilies", preset.requiredFamilies());
        report.put("forbiddenFamilies", preset.forbiddenFamilies());
        report.put("requiredAliases", preset.requiredAliases());
        report.put("forbiddenAliases", preset.forbiddenAliases());
        report.put("selectedFamilies", preset.selectedFamilies());
        report.put("selectedCount", preset.selectedCount());
        Map<String, Object> policyStatus = new LinkedHashMap<>();
        policyStatus.put("known", preset.policyStatusKnown());
        policyStatus.put("passed", preset.policyPassed());
        policyStatus.put("status", preset.policyStatusLabel());
        policyStatus.put("violationCount", preset.policyViolationCount());
        report.put("policyStatus", policyStatus);
        report.put("policyViolations", modelFamilyBundlePolicyViolationsReport(preset.policyViolations()));
        return report;
    }

    private Map<String, Object> modelFamilyBundlePresetConformanceReport(
            ModelFamilyBundleManifest.BundlePresetConformance conformance) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("presetId", conformance.hasPreset() ? conformance.presetId() : null);
        report.put("presetMetadataPresent", conformance.presetMetadataPresent());
        report.put("status", conformance.statusLabel());
        report.put("summary", conformance.compactSummary());
        report.put("matchesPreset", conformance.matchesPreset());
        report.put("cleanPresetBuild", conformance.cleanPresetBuild());
        report.put("selectorsMatch", conformance.selectorsMatch());
        report.put("policyInputsMatch", conformance.policyInputsMatch());
        report.put("explicitSelectorOverride", conformance.explicitSelectorOverride());
        report.put("explicitPolicyOverride", conformance.explicitPolicyOverride());
        report.put("selectorAdditions", conformance.selectorAdditions());
        report.put("selectorOmissions", conformance.selectorOmissions());
        report.put("requiredFamilyAdditions", conformance.requiredFamilyAdditions());
        report.put("requiredFamilyOmissions", conformance.requiredFamilyOmissions());
        report.put("forbiddenFamilyAdditions", conformance.forbiddenFamilyAdditions());
        report.put("forbiddenFamilyOmissions", conformance.forbiddenFamilyOmissions());
        report.put("requiredAliasAdditions", conformance.requiredAliasAdditions());
        report.put("requiredAliasOmissions", conformance.requiredAliasOmissions());
        report.put("forbiddenAliasAdditions", conformance.forbiddenAliasAdditions());
        report.put("forbiddenAliasOmissions", conformance.forbiddenAliasOmissions());
        return report;
    }

    private Map<String, Object> modelFamilyBundlePolicyStatusReport(ModelFamilyBundleManifest.BundlePolicy policy) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("known", policy.statusKnown());
        report.put("passed", policy.passed());
        report.put("status", policy.statusLabel());
        report.put("violationCount", policy.violationCount());
        return report;
    }

    private Map<String, Object> modelFamilyBundlePolicyViolationsReport(
            ModelFamilyBundleManifest.BundlePolicyViolations violations) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("missingRequired", violations.missingRequiredFamilies());
        report.put("selectedForbidden", violations.selectedForbiddenFamilies());
        report.put("missingRequiredAliases", aliasFamilyMapReport(violations.missingRequiredAliases(), "missingFamilies"));
        report.put("selectedForbiddenAliases",
                aliasFamilyMapReport(violations.selectedForbiddenAliases(), "selectedFamilies"));
        return report;
    }

    private List<Map<String, Object>> aliasFamilyMapReport(Map<String, List<String>> values, String familyField) {
        return values.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(entry -> {
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("alias", entry.getKey());
                    report.put(familyField, entry.getValue());
                    return report;
                })
                .toList();
    }

    private Map<String, Object> modelFamilyBundleAliasReport(ModelFamilyBundleManifest.BundleAliasCoverage alias) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", alias.id());
        report.put("description", alias.description());
        report.put("families", alias.families());
        report.put("familyCount", alias.familyCount());
        report.put("selectedFamilies", alias.selectedFamilies());
        report.put("selectedCount", alias.selectedCount());
        report.put("missingFamilies", alias.missingFamilies());
        report.put("missingCount", alias.missingCount());
        report.put("complete", alias.complete());
        report.put("partial", alias.partial());
        return report;
    }

    private List<Map<String, Object>> modelFamilySelectionReport(ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        LinkedHashSet<String> familyIds = new LinkedHashSet<>(manifest.availableFamilies());
        familyIds.addAll(manifest.families());
        familyIds.addAll(discoveredFamilies.keySet());

        return familyIds.stream()
                .map(familyId -> modelFamilySelectionEntry(familyId, manifest, discoveredFamilies))
                .toList();
    }

    private Map<String, Object> modelFamilySelectionEntry(String familyId,
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        String normalizedId = familyId.startsWith("model-family/")
                ? familyId.substring("model-family/".length())
                : familyId;
        ModelFamilyPlugin discovered = discoveredFamilies.get(normalizedId);
        String profile = manifest.familyProfiles().get(normalizedId);
        if ((profile == null || profile.isBlank()) && discovered != null) {
            profile = discovered.supportReport().bundleProfile().key();
        }

        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", normalizedId);
        entry.put("selected", manifest.isFamilySelected(normalizedId));
        entry.put("discovered", discovered != null);
        entry.put("profile", profile);
        entry.put("path", manifest.familyPaths().get(normalizedId));
        return entry;
    }

    private Map<String, Object> modelFamilyPluginSectionReport(Map<String, ModelFamilyPlugin> families) {
        Map<String, Object> section = new LinkedHashMap<>();
        List<ModelFamilyCapabilityMatrixEntry> matrix = ModelFamilyPluginRegistry.global().capabilityMatrix();
        section.put("plugins", families.values().stream()
                .map(this::modelFamilyPluginReport)
                .toList());
        section.put("capabilityMatrix", matrix.stream()
                .map(this::modelFamilyCapabilityMatrixReport)
                .toList());
        section.put("capabilityTotals", modelFamilyCapabilityTotals(matrix));
        section.put("conflicts", ModelFamilyPluginRegistry.global().modelTypeConflicts().stream()
                .map(conflict -> {
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("claimType", conflict.claimType());
                    report.put("claim", conflict.claim());
                    report.put("familyIds", conflict.familyIds());
                    report.put("summary", conflict.summary());
                    return report;
                })
                .toList());
        section.put("contractViolations", ModelFamilyPluginRegistry.global().contractViolations().stream()
                .map(violation -> {
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("familyId", violation.familyId());
                    report.put("code", violation.code());
                    report.put("message", violation.message());
                    report.put("summary", violation.summary());
                    return report;
                })
                .toList());
        return section;
    }

    private Map<String, Object> modelFamilyPluginReport(ModelFamilyPlugin plugin) {
        var report = plugin.supportReport();

        Map<String, Object> value = new LinkedHashMap<>();
        value.put("id", report.id());
        value.put("displayName", report.displayName());
        value.put("modelTypes", report.modelTypes());
        value.put("architectureClassNames", report.architectureClassNames());
        value.put("architectureAdapterIds", report.architectureAdapterIds());
        value.put("tokenizerProfileIds", report.tokenizerProfileIds());
        value.put("tokenizerKinds", report.tokenizerKinds().stream()
                .map(kind -> kind.name().toLowerCase())
                .toList());
        value.put("bundleProfile", report.bundleProfile().key());
        value.put("defaultBundle", report.defaultBundle());
        value.put("capabilities", report.capabilities().stream()
                .map(capability -> capability.name().toLowerCase())
                .toList());
        value.put("directSafetensorStatus", report.directSafetensorStatus().name().toLowerCase());
        value.put("directSafetensorReady", report.directSafetensorReady());
        value.put("directSafetensorLabel", report.directSafetensorLabel());
        value.put("directSafetensorReason", report.directSafetensorReason());
        value.put("directSafetensorCaveats", report.directSafetensorCaveats());
        value.put("metadata", report.metadata());
        return value;
    }

    private Map<String, Object> modelFamilyCapabilityMatrixReport(ModelFamilyCapabilityMatrixEntry entry) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("id", entry.id());
        value.put("displayName", entry.displayName());
        value.put("bundleProfile", entry.bundleProfile());
        value.put("defaultBundle", entry.defaultBundle());
        value.put("causalLm", entry.causalLm());
        value.put("encoder", entry.encoder());
        value.put("decoder", entry.decoder());
        value.put("embedding", entry.embedding());
        value.put("tokenizer", entry.tokenizer());
        value.put("chatTemplate", entry.chatTemplate());
        value.put("vision", entry.vision());
        value.put("audio", entry.audio());
        value.put("multimodal", entry.multimodal());
        value.put("training", entry.training());
        value.put("gguf", entry.gguf());
        value.put("onnx", entry.onnx());
        value.put("architectureAdapterIds", entry.architectureAdapterIds());
        value.put("architectureAdapterCount", entry.architectureAdapterCount());
        value.put("architectureAdapterPresent", entry.architectureAdapterPresent());
        value.put("directSafetensorStatus", entry.directSafetensorStatus().label());
        value.put("directSafetensorReady", entry.directSafetensorReady());
        value.put("directSafetensorReason", entry.directSafetensorReason());
        value.put("directSafetensorCaveats", entry.directSafetensorCaveats());
        value.put("summary", entry.compactSummary());
        return value;
    }

    private Map<String, Object> modelFamilyCapabilityTotals(List<ModelFamilyCapabilityMatrixEntry> matrix) {
        Map<String, Object> totals = new LinkedHashMap<>();
        totals.put("families", matrix.size());
        totals.put("tokenizer", count(matrix, ModelFamilyCapabilityMatrixEntry::tokenizer));
        totals.put("gguf", count(matrix, ModelFamilyCapabilityMatrixEntry::gguf));
        totals.put("onnx", count(matrix, ModelFamilyCapabilityMatrixEntry::onnx));
        totals.put("training", count(matrix, ModelFamilyCapabilityMatrixEntry::training));
        totals.put("vision", count(matrix, ModelFamilyCapabilityMatrixEntry::vision));
        totals.put("audio", count(matrix, ModelFamilyCapabilityMatrixEntry::audio));
        totals.put("multimodal", count(matrix, ModelFamilyCapabilityMatrixEntry::multimodal));
        totals.put("architectureAdapterFamilies",
                count(matrix, ModelFamilyCapabilityMatrixEntry::architectureAdapterPresent));
        totals.put("architectureAdapterCount", matrix.stream()
                .mapToLong(ModelFamilyCapabilityMatrixEntry::architectureAdapterCount)
                .sum());
        totals.put("directSafetensorReady", count(matrix, ModelFamilyCapabilityMatrixEntry::directSafetensorReady));
        totals.put("directSafetensorExperimental", count(matrix,
                entry -> "experimental".equals(entry.directSafetensorStatus().label())));
        totals.put("directSafetensorPending", count(matrix,
                entry -> "pending".equals(entry.directSafetensorStatus().label())));
        return totals;
    }

    private long count(List<ModelFamilyCapabilityMatrixEntry> matrix, Predicate<ModelFamilyCapabilityMatrixEntry> test) {
        return matrix.stream().filter(test).count();
    }

    private Map<String, Object> dynamicPluginSectionReport() {
        Map<String, Object> section = new LinkedHashMap<>();
        try {
            List<tech.kayys.gollek.spi.plugin.GollekPlugin.PluginMetadata> plugins = sdk.listPlugins();
            section.put("plugins", plugins == null ? List.of() : plugins.stream()
                    .map(plugin -> {
                        Map<String, Object> report = new LinkedHashMap<>();
                        report.put("id", plugin.id());
                        report.put("version", plugin.version());
                        report.put("implementationClass", plugin.implementationClass());
                        report.put("order", plugin.order());
                        return report;
                    })
                    .toList());
            section.put("error", null);
        } catch (Exception e) {
            section.put("plugins", List.of());
            section.put("error", e.getMessage());
        }
        return section;
    }

    private void printKernelInfo() {
        System.out.println(BOLD + "=== Kernel Platform ===" + RESET);
        KernelPlatform platform = KernelPlatformDetector.detect();
        System.out.printf("  Platform:     %s%s%s%n", CYAN, platform.getDisplayName(), RESET);
        System.out.printf("  GPU Accel:    %s%n", platform.isCpu()
                ? YELLOW + "No (CPU only)" + RESET
                : GREEN + "Yes" + RESET);
        System.out.printf("  Architecture: %s%n", System.getProperty("os.arch", "unknown"));
        System.out.printf("  OS:           %s %s%n",
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.version", ""));
        System.out.printf("  Java:         %s (%s)%n",
                System.getProperty("java.version", "unknown"),
                System.getProperty("java.vendor", "unknown"));
    }

    private void printExtensions() {
        Set<String> runtimeProviders = getRuntimeProviderIds();

        System.out.println(BOLD + "=== Runtime Modules ===" + RESET);
        System.out.printf("  %-8s %-12s %-18s %-10s %-10s%n",
                "TYPE", "NAME", "PROFILE", "PACKAGED", "PROVIDER");
        System.out.println("  " + "-".repeat(62));

        int shown = 0;
        for (ExtensionDef ext : EXTENSIONS) {
            boolean packaged = isClassPresent(ext.markerClass());
            boolean providerAvailable = runtimeProviders.contains(ext.providerId());
            if (!showAll && !packaged) {
                continue;
            }
            String packagedStr = packaged ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            String providerStr = providerAvailable ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            System.out.printf("  %-8s %-12s %-18s %-19s %-19s%n",
                    ext.type(),
                    ext.name(),
                    ext.profile(),
                    packagedStr,
                    providerStr);
            shown++;
        }

        if (shown == 0) {
            System.out.println("  No packaged runtime modules found. Use --all to see all possible modules.");
        }

        if (!runtimeProviders.isEmpty()) {
            System.out.printf("%n  " + DIM + "Active providers: %s" + RESET + "%n",
                    String.join(", ", runtimeProviders));
        }
    }

    private void printRunnerPlugins() {
        System.out.println(BOLD + "=== Runner Plugins ===" + RESET);
        try {
            Map<String, RunnerPlugin> plugins = collectRunnerPlugins();
            if (plugins.isEmpty()) {
                System.out.println("  No runner plugins discovered.");
                return;
            }

            System.out.printf("  %-20s %-12s %-10s%n", "ID", "FORMAT", "STATUS");
            System.out.println("  " + "-".repeat(44));

            int count = 0;
            for (RunnerPlugin plugin : plugins.values()) {
                String id = plugin.id();
                String format = plugin.format() != null ? plugin.format() : "unknown";
                String status = GREEN + "active" + RESET;
                System.out.printf("  %-20s %-12s %-19s%n", id, format, status);
                count++;
            }

            if (count == 0) {
                System.out.println("  No runner plugins active.");
            } else {
                System.out.printf("%n  " + DIM + "Total runners: %d" + RESET + "%n", count);
            }
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate runner plugins: " + e.getMessage() + RESET);
        }
    }

    private Map<String, RunnerPlugin> collectRunnerPlugins() {
        Map<String, RunnerPlugin> plugins = new LinkedHashMap<>();
        if (runnerPluginInstances != null && !runnerPluginInstances.isUnsatisfied()) {
            for (RunnerPlugin plugin : runnerPluginInstances) {
                plugins.putIfAbsent(plugin.id(), plugin);
            }
        }
        for (RunnerPlugin plugin : RunnerPluginManager.getInstance().getAvailablePlugins()) {
            plugins.putIfAbsent(plugin.id(), plugin);
        }
        return plugins;
    }

    private void printModelFamilies() {
        System.out.println(BOLD + "=== Model Family Plugins ===" + RESET);
        try {
            Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins();
            ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
            printModelFamilyBundleManifest(manifest, families);
            if (families.isEmpty()) {
                if (manifest.detached()) {
                    System.out.println("  Model-family plugins intentionally detached by this CLI build.");
                    System.out.println("  Rebuild with -Pgollek.modelFamilies=core,direct,vlm,embedding,research,all or another selector alias to attach them.");
                } else {
                    System.out.println("  No model-family plugins discovered.");
                }
                return;
            }

            System.out.printf("  %-14s %-18s %-13s %-16s %-18s %-14s %-24s%n",
                    "ID", "NAME", "PROFILE", "CAPABILITIES", "TOKENIZER", "DIRECT", "REASON");
            System.out.println("  " + "-".repeat(126));

            for (ModelFamilyPlugin plugin : families.values()) {
                var report = plugin.supportReport();
                String direct = report.directSafetensorReady()
                        ? GREEN + report.directSafetensorLabel() + RESET
                        : DIM + report.directSafetensorLabel() + RESET;
                String capabilities = report.capabilities().stream()
                        .limit(3)
                        .map(Enum::name)
                        .map(String::toLowerCase)
                        .reduce((a, b) -> a + "," + b)
                        .orElse("none");
                String tokenizers = report.tokenizerKinds().stream()
                        .limit(2)
                        .map(kind -> kind.name().toLowerCase())
                        .reduce((a, b) -> a + "," + b)
                        .orElse(DIM + "none" + RESET);
                System.out.printf("  %-14s %-18s %-13s %-16s %-27s %-23s %-24s%n",
                        report.id(),
                        report.displayName(),
                        report.bundleProfile().key(),
                        capabilities,
                        tokenizers,
                        direct,
                        shortText(report.directSafetensorReason(), 24));
            }

            List<ModelFamilyCapabilityMatrixEntry> matrix = ModelFamilyPluginRegistry.global().capabilityMatrix();
            printModelFamilyCapabilitySummary(matrix);

            List<String> directCaveats = families.values().stream()
                    .map(ModelFamilyPlugin::supportReport)
                    .filter(report -> !report.directSafetensorCaveats().isEmpty())
                    .map(report -> "  - " + report.id() + ": " + report.shortDirectSafetensorCaveats())
                    .toList();
            if (!directCaveats.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Partial direct SafeTensor caveats:" + RESET);
                directCaveats.forEach(System.out::println);
            }

            var conflicts = ModelFamilyPluginRegistry.global().modelTypeConflicts();
            if (!conflicts.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Model family claim conflicts:" + RESET);
                for (var conflict : conflicts) {
                    System.out.println("  - " + conflict.summary());
                }
            }

            var contractViolations = ModelFamilyPluginRegistry.global().contractViolations();
            if (!contractViolations.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Model family contract violations:" + RESET);
                for (var violation : contractViolations) {
                    System.out.println("  - " + violation.summary());
                }
            }

            System.out.printf("%n  " + DIM + "Total model families: %d" + RESET + "%n", families.size());
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate model families: " + e.getMessage() + RESET);
        }
    }

    private void printModelFamilyCapabilitySummary(List<ModelFamilyCapabilityMatrixEntry> matrix) {
        if (matrix.isEmpty()) {
            return;
        }

        System.out.println();
        System.out.println("  " + CYAN + "Capability matrix summary:" + RESET);
        System.out.printf(
                "  families=%d tokenizer=%d gguf=%d adapters=%d direct-ready=%d direct-experimental=%d "
                        + "direct-pending=%d onnx=%d training=%d multimodal=%d vision=%d audio=%d%n",
                matrix.size(),
                count(matrix, ModelFamilyCapabilityMatrixEntry::tokenizer),
                count(matrix, ModelFamilyCapabilityMatrixEntry::gguf),
                matrix.stream()
                        .mapToLong(ModelFamilyCapabilityMatrixEntry::architectureAdapterCount)
                        .sum(),
                count(matrix, ModelFamilyCapabilityMatrixEntry::directSafetensorReady),
                count(matrix, entry -> "experimental".equals(entry.directSafetensorStatus().label())),
                count(matrix, entry -> "pending".equals(entry.directSafetensorStatus().label())),
                count(matrix, ModelFamilyCapabilityMatrixEntry::onnx),
                count(matrix, ModelFamilyCapabilityMatrixEntry::training),
                count(matrix, ModelFamilyCapabilityMatrixEntry::multimodal),
                count(matrix, ModelFamilyCapabilityMatrixEntry::vision),
                count(matrix, ModelFamilyCapabilityMatrixEntry::audio));
    }

    private void printModelFamilyBundleManifest(ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        if (!manifest.present()) {
            return;
        }

        System.out.printf("  Build selectors: %s%n", manifest.joinedSelectors());
        System.out.printf("  Build selector source: %s%n", manifest.displaySelectorSource());
        System.out.printf("  Build bundle policy: %s%n", manifest.displayBundlePolicyStatus());
        System.out.printf("  Build policy source: %s%n", manifest.displayPolicySource());
        System.out.printf("  Build fixture status: %s%n", manifest.displayFixtureStatus());
        System.out.printf("  Build availability: %s%n",
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        manifest,
                        discoveredFamilies.keySet()).compactSummary());
        System.out.printf("  Build preset:   %s%n", manifest.displayBundlePreset());
        System.out.printf("  Build preset policy: %s%n", manifest.displayBundlePresetPolicyStatus());
        System.out.printf("  Build preset conformance: %s%n", manifest.displayActiveBundlePresetConformance());
        System.out.printf("  Build fingerprint: %s%n", manifest.displayFingerprint());
        System.out.printf("  Build profiles:  %s%n", manifest.displayProfiles());
        System.out.printf("  Build families:  %s%n", manifest.displayFamilies());

        List<String> missing = manifest.missingDiscovered(discoveredFamilies.keySet());
        if (!missing.isEmpty()) {
            System.out.printf("  %sManifest mismatch:%s expected families not discovered: %s%n",
                    YELLOW, RESET, String.join(", ", missing));
        }

        if (showAll) {
            System.out.printf("  Requested selector families: %s%n", manifest.joinedRequestedFamilies());
            System.out.printf("  Requested selector profiles: %s%n", manifest.joinedRequestedProfiles());
            System.out.printf("  Requested selector aliases: %s%n", manifest.joinedRequestedAliases());
            System.out.printf("  Available build selectors: %s%n", manifest.joinedAvailableSelectors());
            System.out.printf("  Available bundle presets: %s%n", manifest.joinedAvailableBundlePresets());
            System.out.printf("  Build bundle presets: %s%n", manifest.joinedBundlePresets());
            System.out.printf("  Build selector aliases: %s%n", manifest.joinedBundleAliases());
            System.out.printf("  Complete selector aliases: %s%n", manifest.joinedCompleteBundleAliases());
            System.out.printf("  Partial selector aliases: %s%n", manifest.joinedPartialBundleAliases());
            System.out.printf("  Unbundled families: %s%n", manifest.joinedOmittedFamiliesWithProfiles());
        }
        System.out.println();
    }

    private Map<String, ModelFamilyPlugin> collectModelFamilyPlugins() {
        Map<String, ModelFamilyPlugin> families = new LinkedHashMap<>();
        if (modelFamilyPluginInstances != null && !modelFamilyPluginInstances.isUnsatisfied()) {
            for (ModelFamilyPlugin plugin : modelFamilyPluginInstances) {
                ModelFamilyPluginRegistry.global().register(plugin);
                families.putIfAbsent(plugin.descriptor().id(), plugin);
            }
        }
        for (ModelFamilyPlugin plugin : ModelFamilyPluginRegistry.global().discoverServiceLoaderPlugins()) {
            families.putIfAbsent(plugin.descriptor().id(), plugin);
        }
        return families;
    }

    private static String shortText(String value, int maxLength) {
        if (value == null || value.isBlank() || value.length() <= maxLength) {
            return value == null || value.isBlank() ? "-" : value;
        }
        return value.substring(0, Math.max(0, maxLength - 3)) + "...";
    }

    private void printDynamicPlugins() {
        System.out.println(BOLD + "=== Dynamic Plugins ===" + RESET);
        try {
            List<tech.kayys.gollek.spi.plugin.GollekPlugin.PluginMetadata> plugins = sdk.listPlugins();
            if (plugins != null && !plugins.isEmpty()) {
                for (var plugin : plugins) {
                    System.out.printf("  - %s (%s) v%s%n",
                            plugin.implementationClass(), plugin.id(), plugin.version());
                }
            } else {
                System.out.println("  No dynamic plugins discovered.");
            }
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to fetch plugins: " + e.getMessage() + RESET);
        }

        System.out.println();
        System.out.println(DIM + "Tip: enable cloud modules at build time with "
                + "-Pext-cloud-gemini,ext-cloud-cerebras" + RESET);
        System.out.println(DIM + "Tip: select model-family bundles with "
                + "-Pgollek.modelFamilies=direct,vlm,embedding,moe,research, none or all" + RESET);
    }

    private Set<String> getRuntimeProviderIds() {
        Set<String> ids = new LinkedHashSet<>();
        try {
            List<ProviderInfo> providers = sdk.listAvailableProviders();
            for (ProviderInfo provider : providers) {
                if (provider.id() != null && !provider.id().isBlank()) {
                    ids.add(provider.id());
                }
            }
        } catch (Exception e) {
            // Keep output useful even if provider registry is unavailable.
        }
        return ids;
    }

    private boolean isClassPresent(String className) {
        try {
            Class.forName(className, false, Thread.currentThread().getContextClassLoader());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
