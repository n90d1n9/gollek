/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Plugin availability checker for CLI commands.
 * Checks both high-level providers (LLM providers like GGUF, Gemini)
 * and low-level runner plugins (ONNX Runtime, SafeTensor, etc.).
 */
@ApplicationScoped
public class PluginAvailabilityChecker {

    private static final Logger LOG = Logger.getLogger(PluginAvailabilityChecker.class);

    public record ModelFamilyBundleAvailability(
            boolean present,
            boolean detached,
            boolean healthy,
            String status,
            int selectedFamilyCount,
            int discoveredSelectedFamilyCount,
            int missingSelectedFamilyCount,
            int omittedFamilyCount,
            String policyStatus,
            int policyViolationCount,
            String fixtureStatus,
            Boolean fixturePassed,
            int fixtureMissingRequiredCount,
            int fixtureProblemFamilyCount,
            String presetConformanceStatus,
            List<String> problems,
            List<String> remediationHints,
            List<String> missingSelectedFamilies,
            List<String> omittedFamilies,
            List<String> fixtureMissingRequiredFamilies,
            List<String> fixtureProblemFamilies) {

        public ModelFamilyBundleAvailability {
            status = status == null || status.isBlank() ? "unknown" : status;
            policyStatus = policyStatus == null || policyStatus.isBlank() ? "unknown" : policyStatus;
            fixtureStatus = fixtureStatus == null || fixtureStatus.isBlank() ? "unknown" : fixtureStatus;
            fixtureMissingRequiredCount = Math.max(0, fixtureMissingRequiredCount);
            fixtureProblemFamilyCount = Math.max(0, fixtureProblemFamilyCount);
            presetConformanceStatus = presetConformanceStatus == null || presetConformanceStatus.isBlank()
                    ? "none"
                    : presetConformanceStatus;
            problems = List.copyOf(problems == null ? List.of() : problems);
            remediationHints = List.copyOf(remediationHints == null ? List.of() : remediationHints);
            missingSelectedFamilies = List.copyOf(missingSelectedFamilies == null
                    ? List.of()
                    : missingSelectedFamilies);
            omittedFamilies = List.copyOf(omittedFamilies == null ? List.of() : omittedFamilies);
            fixtureMissingRequiredFamilies = List.copyOf(fixtureMissingRequiredFamilies == null
                    ? List.of()
                    : fixtureMissingRequiredFamilies);
            fixtureProblemFamilies = List.copyOf(fixtureProblemFamilies == null ? List.of() : fixtureProblemFamilies);
        }

        public String compactSummary() {
            return "%s(selected=%d, discovered=%d, missing=%d, omitted=%d, policy=%s, fixture=%s, "
                    + "presetConformance=%s)"
                    .formatted(
                            status,
                            selectedFamilyCount,
                            discoveredSelectedFamilyCount,
                            missingSelectedFamilyCount,
                            omittedFamilyCount,
                            policyStatus,
                            fixtureStatus,
                            presetConformanceStatus);
        }
    }

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    Instance<RunnerPlugin> runnerPlugins;

    @Inject
    Instance<ModelFamilyPlugin> modelFamilyPlugins;

    /**
     * Check if any LLM providers are registered.
     */
    public boolean hasProviders() {
        try {
            var providers = providerRegistry.getAllProviders();
            return providers != null && !providers.isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking providers: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Check if a specific provider is available by ID.
     */
    public boolean hasProvider(String providerId) {
        try {
            return providerRegistry.hasProvider(providerId);
        } catch (Exception e) {
            LOG.debugf("Error checking provider %s: %s", providerId, e.getMessage());
            return false;
        }
    }

    /**
     * Check if any runner plugins are discovered.
     * Runner plugins are low-level execution backends (ONNX, GGUF native, etc.)
     */
    public boolean hasRunnerPlugins() {
        try {
            return !getRunnerPluginIds().isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking runner plugins: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get a list of all discovered runner plugin IDs.
     */
    public List<String> getRunnerPluginIds() {
        Set<String> ids = new LinkedHashSet<>();
        try {
            if (runnerPlugins != null && !runnerPlugins.isUnsatisfied()) {
                ids.addAll(runnerPlugins.stream()
                        .map(RunnerPlugin::id)
                        .collect(Collectors.toList()));
            }
        } catch (Exception e) {
            LOG.debugf("Error listing CDI runner plugins: %s", e.getMessage());
        }

        try {
            ids.addAll(RunnerPluginManager.getInstance().getAvailablePlugins().stream()
                    .map(RunnerPlugin::id)
                    .collect(Collectors.toList()));
        } catch (Exception e) {
            LOG.debugf("Error listing ServiceLoader runner plugins: %s", e.getMessage());
        }

        return List.copyOf(ids);
    }

    /**
     * Check if any model-family plugins are discovered.
     * Model-family plugins describe reusable architecture/tokenizer metadata.
     */
    public boolean hasModelFamilyPlugins() {
        try {
            return !getModelFamilyPluginIds().isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking model-family plugins: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get a list of all discovered model-family plugin IDs.
     */
    public List<String> getModelFamilyPluginIds() {
        Set<String> ids = new LinkedHashSet<>();
        try {
            if (modelFamilyPlugins != null && !modelFamilyPlugins.isUnsatisfied()) {
                modelFamilyPlugins.forEach(plugin -> {
                    ModelFamilyPluginRegistry.global().register(plugin);
                    ids.add(plugin.descriptor().id());
                });
            }
        } catch (Exception e) {
            LOG.debugf("Error listing CDI model-family plugins: %s", e.getMessage());
        }

        try {
            ids.addAll(ModelFamilyPluginRegistry.global().discoverServiceLoaderPlugins().stream()
                    .map(plugin -> plugin.descriptor().id())
                    .collect(Collectors.toList()));
        } catch (Exception e) {
            LOG.debugf("Error listing ServiceLoader model-family plugins: %s", e.getMessage());
        }

        return List.copyOf(ids);
    }

    /**
     * Get model-family IDs with direct SafeTensor readiness for diagnostics.
     */
    public List<String> getModelFamilySupportSummaries() {
        getModelFamilyPluginIds();
        try {
            return ModelFamilyPluginRegistry.global().supportReports().stream()
                    .map(report -> report.id() + "[" + report.bundleProfile().key() + "]("
                            + report.shortDirectSafetensorSummary() + ")")
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family support summaries: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get compact capability-matrix rows for packaged model-family plugins.
     */
    public List<String> getModelFamilyCapabilityMatrixSummaries() {
        getModelFamilyPluginIds();
        try {
            return ModelFamilyPluginRegistry.global().capabilityMatrix().stream()
                    .map(entry -> entry.compactSummary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family capability matrix summaries: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get duplicate model-type claims across packaged model-family plugins.
     */
    public List<String> getModelFamilyClaimConflicts() {
        getModelFamilyPluginIds();
        try {
            return ModelFamilyPluginRegistry.global().modelTypeConflicts().stream()
                    .map(conflict -> conflict.summary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family claim conflicts: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get model-family plugin contract violations across packaged families.
     */
    public List<String> getModelFamilyContractViolations() {
        getModelFamilyPluginIds();
        try {
            return ModelFamilyPluginRegistry.global().contractViolations().stream()
                    .map(violation -> violation.summary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family contract violations: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get the model-family bundle manifest packaged into this CLI build.
     */
    public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
        return ModelFamilyBundleManifest.load();
    }

    /**
     * Get build selector aliases packaged into this CLI build.
     */
    public List<String> getModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.bundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAlias::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get build bundle presets packaged into this CLI build.
     */
    public List<String> getModelFamilyBundlePresetSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.bundlePresets().stream()
                .map(ModelFamilyBundleManifest.BundlePreset::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get the active build preset packaged into this CLI build, including policy status.
     */
    public String getActiveModelFamilyBundlePresetSummary() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present() || !manifest.hasBundlePreset()) {
            return "";
        }
        return manifest.activeBundlePreset()
                .map(ModelFamilyBundleManifest.BundlePreset::compactSummary)
                .orElse(manifest.bundlePreset() + "(metadata=missing)");
    }

    /**
     * Get selector aliases fully covered by this CLI build.
     */
    public List<String> getCompleteModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.completeBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get selector aliases partially covered by this CLI build.
     */
    public List<String> getPartialModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.partialBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get build-selected model families that were not discovered at runtime.
     */
    public List<String> getMissingBundledModelFamilies() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.missingDiscovered(new LinkedHashSet<>(getModelFamilyPluginIds()));
    }

    /**
     * Get the packaged model-family bundle availability relative to runtime discovery.
     */
    public ModelFamilyBundleAvailability getModelFamilyBundleAvailability() {
        return modelFamilyBundleAvailability(
                getModelFamilyBundleManifest(),
                new LinkedHashSet<>(getModelFamilyPluginIds()));
    }

    /**
     * Build a reusable availability summary for model-family bundle diagnostics.
     */
    public static ModelFamilyBundleAvailability modelFamilyBundleAvailability(
            ModelFamilyBundleManifest manifest,
            Set<String> discoveredFamilyIds) {
        if (manifest == null || !manifest.present()) {
            return new ModelFamilyBundleAvailability(
                    false,
                    false,
                    false,
                    "manifest_missing",
                    0,
                    0,
                    0,
                    0,
                    "unknown",
                    0,
                    "unknown",
                    null,
                    0,
                    0,
                    "none",
                    List.of("model-family bundle manifest is missing"),
                    List.of("Run 'gollek modules --all' in a packaged CLI, or rebuild the CLI so "
                            + ModelFamilyBundleManifest.RESOURCE_PATH + " is generated."),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of());
        }

        List<String> missing = manifest.missingDiscovered(discoveredFamilyIds);
        List<String> omitted = manifest.omittedFamilies();
        List<String> problems = new ArrayList<>();
        List<String> hints = new ArrayList<>();
        int selectedCount = manifest.families().size();
        int discoveredSelectedCount = Math.max(0, selectedCount - missing.size());
        boolean policyFailed = manifest.bundlePolicy().statusKnown()
                && !Boolean.TRUE.equals(manifest.bundlePolicy().passed());
        ModelFamilyBundleManifest.FixtureStatus fixtureStatus = manifest.fixtureStatus();
        boolean fixtureFailed = fixtureStatus.statusKnown() && !Boolean.TRUE.equals(fixtureStatus.passed());
        ModelFamilyBundleManifest.BundlePresetConformance conformance = manifest.activeBundlePresetConformance();
        boolean presetDrifted = conformance.hasPreset()
                && (!conformance.presetMetadataPresent() || "drifted".equals(conformance.statusLabel()));

        if (manifest.detached()) {
            hints.add("Attach model-family plugins by rebuilding with -Pgollek.modelFamilies=<selector> "
                    + "when this CLI needs local tokenizer or architecture metadata.");
        } else {
            if (!missing.isEmpty()) {
                problems.add("selected model-family plugins were not discovered: " + String.join(", ", missing));
                hints.add("Ensure the selected model-family plugin artifacts are packaged and visible to CDI or "
                        + "ServiceLoader: " + String.join(", ", missing));
            }
            if (policyFailed) {
                problems.add("model-family bundle policy failed with "
                        + manifest.bundlePolicy().violationCount() + " violation(s)");
                hints.add("Adjust -Pgollek.requiredModelFamilies, -Pgollek.forbiddenModelFamilies, "
                        + "alias policy flags, or choose a bundle preset whose policy matches this build.");
            }
            if (fixtureFailed) {
                problems.add("model-family fixture gate failed: " + fixtureStatus.compactStatus());
                hints.add("Run :ui:gollek-cli:validateModelFamilyFixtures, repair missing/problem fixture families, "
                        + "then regenerate the bundle manifest and fixture fingerprint lock if intentional.");
            }
            if (presetDrifted) {
                problems.add("active model-family bundle preset conformance is " + conformance.statusLabel());
                hints.add("Rebuild with the reviewed -Pgollek.modelFamilyBundlePreset="
                        + conformance.presetId()
                        + " without selector/policy overrides, or intentionally update the preset and bundle lock.");
            }
        }

        String status;
        boolean healthy;
        if (manifest.detached()) {
            status = "detached";
            healthy = true;
        } else if (!missing.isEmpty()) {
            status = "missing_plugins";
            healthy = false;
        } else if (policyFailed) {
            status = "policy_failed";
            healthy = false;
        } else if (fixtureFailed) {
            status = "fixture_failed";
            healthy = false;
        } else if (presetDrifted) {
            status = "preset_drifted";
            healthy = false;
        } else {
            status = "ready";
            healthy = true;
        }

        return new ModelFamilyBundleAvailability(
                true,
                manifest.detached(),
                healthy,
                status,
                selectedCount,
                discoveredSelectedCount,
                missing.size(),
                omitted.size(),
                manifest.bundlePolicy().statusLabel(),
                manifest.bundlePolicy().violationCount(),
                fixtureStatus.statusLabel(),
                fixtureStatus.passed(),
                fixtureStatus.missingRequiredCount(),
                fixtureStatus.problemFamilyCount(),
                conformance.statusLabel(),
                problems,
                hints,
                missing,
                omitted,
                fixtureStatus.missingRequiredFamilies(),
                fixtureStatus.problemFamilies());
    }

    /**
     * Get an error message when no plugins are available at all.
     */
    public String getNoPluginsError() {
        StringBuilder sb = new StringBuilder("""
            Error: No inference providers or runner plugins are available.
            
            This typically means no runtime extensions are packaged in this CLI build.
            
            To fix this, ensure at least one of these dependencies is on the classpath:
              - gollek-ext-runner-gguf       (GGUF/llama.cpp runtime)
              - gollek-runner-onnx           (ONNX Runtime)
              - gollek-ext-runner-safetensor  (SafeTensor runtime)
              - gollek-ext-cloud-gemini      (Gemini cloud provider)
              - gollek-ext-cloud-cerebras    (Cerebras cloud provider)
            
            Run 'gollek modules --all' for packaged runtime details.
            """);
        appendModelFamilyBundleDiagnostics(sb);
        return sb.toString();
    }

    /**
     * Get an error message when a specific provider is not found.
     */
    public String getProviderNotFoundError(String providerId) {
        StringBuilder sb = new StringBuilder();
        sb.append("Error: Provider '").append(providerId).append("' is not available.\n\n");

        try {
            var available = providerRegistry.getAllProviders();
            if (available != null && !available.isEmpty()) {
                sb.append("Available providers:\n");
                for (var p : available) {
                    sb.append("  - ").append(p.id()).append("\n");
                }
            } else {
                sb.append("No providers are currently available.\n");
            }
        } catch (Exception e) {
            sb.append("Could not list available providers.\n");
        }

        List<String> runners = getRunnerPluginIds();
        if (!runners.isEmpty()) {
            sb.append("\nAvailable runner plugins: ").append(String.join(", ", runners)).append("\n");
        }

        List<String> families = getModelFamilySupportSummaries();
        if (!families.isEmpty()) {
            sb.append("Available model-family plugins: ").append(String.join(", ", families)).append("\n");
        }

        appendModelFamilyBundleDiagnostics(sb);

        List<String> conflicts = getModelFamilyClaimConflicts();
        if (!conflicts.isEmpty()) {
            sb.append("Model-family claim conflicts: ").append(String.join("; ", conflicts)).append("\n");
        }

        sb.append("\nRun 'gollek modules --all' for packaged runtime details.\n");
        return sb.toString();
    }

    private void appendModelFamilyBundleDiagnostics(StringBuilder sb) {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return;
        }
        Set<String> discoveredFamilies = new LinkedHashSet<>(getModelFamilyPluginIds());
        ModelFamilyBundleAvailability availability = modelFamilyBundleAvailability(manifest, discoveredFamilies);

        sb.append("Packaged model-family bundle: selectors=")
                .append(manifest.joinedSelectors())
                .append("; selectorSource=")
                .append(manifest.displaySelectorSource())
                .append("; preset=")
                .append(manifest.displayBundlePreset())
                .append("; policySource=")
                .append(manifest.displayPolicySource())
                .append("; fingerprint=")
                .append(manifest.displayFingerprint())
                .append("; profiles=")
                .append(manifest.displayProfiles())
                .append("; families=")
                .append(manifest.displayFamilies())
                .append("\n");

        sb.append("Packaged model-family bundle policy: ")
                .append(manifest.displayBundlePolicyStatus())
                .append("\n");
        appendFixtureDiagnostics(sb, manifest.fixtureStatus());
        sb.append("Packaged model-family bundle availability: ")
                .append(availability.compactSummary())
                .append("\n");
        appendAvailabilityDetails(sb, availability);
        appendBundlePolicyViolations(sb, "Packaged bundle policy", manifest.bundlePolicy().violations());

        if (manifest.hasBundlePreset()) {
            sb.append("Active model-family bundle preset: ")
                    .append(getActiveModelFamilyBundlePresetSummary())
                    .append("\n");
            sb.append("Active model-family bundle preset policy: ")
                    .append(manifest.displayBundlePresetPolicyStatus())
                    .append("\n");
            sb.append("Active model-family bundle preset conformance: ")
                    .append(manifest.displayActiveBundlePresetConformance())
                    .append("\n");
            manifest.activeBundlePreset().ifPresent(preset ->
                    appendActiveModelFamilyBundlePresetViolations(sb, preset));
        }

        if (!manifest.bundlePresets().isEmpty()) {
            sb.append("Available model-family bundle presets: ")
                    .append(String.join(", ", getModelFamilyBundlePresetSummaries()))
                    .append("\n");
        }

        if (!manifest.requestedAliases().isEmpty()) {
            sb.append("Requested model-family selector aliases: ")
                    .append(manifest.joinedRequestedAliases())
                    .append("\n");
        }

        if (!manifest.unknownSelectors().isEmpty()) {
            sb.append("Unknown packaged selector metadata: ")
                    .append(String.join(", ", manifest.unknownSelectors()))
                    .append("\n");
        }

        List<String> completeAliases = getCompleteModelFamilyBundleAliasSummaries();
        if (!completeAliases.isEmpty()) {
            sb.append("Complete model-family selector aliases: ")
                    .append(String.join(", ", completeAliases))
                    .append("\n");
        }

        List<String> partialAliases = getPartialModelFamilyBundleAliasSummaries();
        if (!partialAliases.isEmpty()) {
            sb.append("Partial model-family selector aliases: ")
                    .append(String.join(", ", partialAliases))
                    .append("\n");
        }

        if (manifest.detached()) {
            sb.append("Model-family plugins are intentionally detached in this CLI build.\n");
        } else if (!manifest.omittedFamilies().isEmpty()) {
            sb.append("Unbundled model-family plugins: ")
                    .append(manifest.joinedOmittedFamiliesWithProfiles())
                    .append("\n");
        }

        List<String> missing = manifest.missingDiscovered(discoveredFamilies);
        if (!missing.isEmpty()) {
            sb.append("Bundled model-family plugins not discovered: ")
                    .append(String.join(", ", missing))
                    .append("\n");
        }
    }

    private void appendFixtureDiagnostics(StringBuilder sb, ModelFamilyBundleManifest.FixtureStatus fixtureStatus) {
        sb.append("Packaged model-family fixture status: ")
                .append(fixtureStatus.compactStatus())
                .append("\n");
        if (!fixtureStatus.missingRequiredFamilies().isEmpty()) {
            sb.append("Packaged model-family missing required fixtures: ")
                    .append(String.join(", ", fixtureStatus.missingRequiredFamilies()))
                    .append("\n");
        }
        if (!fixtureStatus.problemFamilies().isEmpty()) {
            sb.append("Packaged model-family problem fixtures: ")
                    .append(String.join(", ", fixtureStatus.problemFamilies()))
                    .append("\n");
        }
    }

    private void appendAvailabilityDetails(StringBuilder sb, ModelFamilyBundleAvailability availability) {
        if (!availability.problems().isEmpty()) {
            sb.append("Packaged model-family bundle problems: ")
                    .append(String.join("; ", availability.problems()))
                    .append("\n");
        }
        if (!availability.remediationHints().isEmpty()) {
            sb.append("Packaged model-family bundle hints: ")
                    .append(String.join("; ", availability.remediationHints()))
                    .append("\n");
        }
    }

    private void appendActiveModelFamilyBundlePresetViolations(
            StringBuilder sb,
            ModelFamilyBundleManifest.BundlePreset preset) {
        if (Boolean.TRUE.equals(preset.policyPassed()) || preset.policyViolationCount() == 0) {
            return;
        }

        appendBundlePolicyViolations(sb, "Active preset", preset.policyViolations());
    }

    private void appendBundlePolicyViolations(
            StringBuilder sb,
            String label,
            ModelFamilyBundleManifest.BundlePolicyViolations violations) {
        if (!violations.missingRequiredFamilies().isEmpty()) {
            sb.append(label).append(" missing required families: ")
                    .append(String.join(", ", violations.missingRequiredFamilies()))
                    .append("\n");
        }
        if (!violations.selectedForbiddenFamilies().isEmpty()) {
            sb.append(label).append(" selected forbidden families: ")
                    .append(String.join(", ", violations.selectedForbiddenFamilies()))
                    .append("\n");
        }
        appendAliasFamilyViolations(
                sb,
                label + " missing required alias ",
                violations.missingRequiredAliases());
        appendAliasFamilyViolations(
                sb,
                label + " selected forbidden alias ",
                violations.selectedForbiddenAliases());
    }

    private void appendAliasFamilyViolations(
            StringBuilder sb,
            String prefix,
            java.util.Map<String, List<String>> violations) {
        violations.entrySet().stream()
                .sorted(java.util.Map.Entry.comparingByKey())
                .forEach(entry -> sb.append(prefix)
                        .append(entry.getKey())
                        .append(": ")
                        .append(String.join(", ", entry.getValue()))
                        .append("\n"));
    }
}
