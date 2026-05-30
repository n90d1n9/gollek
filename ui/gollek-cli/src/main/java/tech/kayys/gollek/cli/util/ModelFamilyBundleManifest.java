/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;

/**
 * Build-time model-family bundle metadata packaged into the CLI.
 */
public record ModelFamilyBundleManifest(
        boolean present,
        int schemaVersion,
        String bundleFingerprint,
        boolean detached,
        List<String> selectors,
        List<String> families,
        List<String> profiles,
        List<String> availableFamilies,
        List<String> availableProfiles,
        List<String> availableSelectors,
        String selectorSource,
        List<String> explicitSelectors,
        List<String> presetSelectors,
        List<String> defaultSelectors,
        String policySource,
        List<String> presetRequiredFamilies,
        List<String> presetForbiddenFamilies,
        List<String> presetRequiredAliases,
        List<String> presetForbiddenAliases,
        List<String> explicitRequiredFamilies,
        List<String> explicitForbiddenFamilies,
        List<String> explicitRequiredAliases,
        List<String> explicitForbiddenAliases,
        String bundlePreset,
        BundlePolicy bundlePolicy,
        FixtureStatus fixtureStatus,
        List<BundlePreset> bundlePresets,
        List<BundleAlias> bundleAliases,
        Map<String, Boolean> familySelected,
        Map<String, String> familyProfiles,
        Map<String, String> familyPaths) {

    public static final String RESOURCE_PATH = "META-INF/gollek-model-family-bundle.properties";
    public static final int CURRENT_SCHEMA_VERSION = 1;

    public ModelFamilyBundleManifest {
        schemaVersion = Math.max(0, schemaVersion);
        bundleFingerprint = bundleFingerprint == null || bundleFingerprint.isBlank()
                ? "-"
                : bundleFingerprint.trim();
        selectors = List.copyOf(selectors == null ? List.of() : selectors);
        families = List.copyOf(families == null ? List.of() : families);
        profiles = List.copyOf(profiles == null ? List.of() : profiles);
        availableFamilies = List.copyOf(availableFamilies == null ? List.of() : availableFamilies);
        availableProfiles = List.copyOf(availableProfiles == null ? List.of() : availableProfiles);
        availableSelectors = List.copyOf(availableSelectors == null ? List.of() : availableSelectors);
        selectorSource = normalizeSelectorSource(selectorSource);
        explicitSelectors = normalizeFamilyIds(explicitSelectors);
        presetSelectors = normalizeFamilyIds(presetSelectors);
        defaultSelectors = normalizeFamilyIds(defaultSelectors);
        policySource = normalizePolicySource(policySource);
        presetRequiredFamilies = normalizeFamilyIds(presetRequiredFamilies);
        presetForbiddenFamilies = normalizeFamilyIds(presetForbiddenFamilies);
        presetRequiredAliases = normalizeFamilyIds(presetRequiredAliases);
        presetForbiddenAliases = normalizeFamilyIds(presetForbiddenAliases);
        explicitRequiredFamilies = normalizeFamilyIds(explicitRequiredFamilies);
        explicitForbiddenFamilies = normalizeFamilyIds(explicitForbiddenFamilies);
        explicitRequiredAliases = normalizeFamilyIds(explicitRequiredAliases);
        explicitForbiddenAliases = normalizeFamilyIds(explicitForbiddenAliases);
        bundlePreset = normalizeFamilyId(bundlePreset);
        bundlePolicy = bundlePolicy == null ? BundlePolicy.empty() : bundlePolicy;
        fixtureStatus = fixtureStatus == null ? FixtureStatus.empty() : fixtureStatus;
        bundlePresets = List.copyOf(bundlePresets == null ? List.of() : bundlePresets);
        bundleAliases = List.copyOf(bundleAliases == null ? List.of() : bundleAliases);
        familySelected = Map.copyOf(familySelected == null ? Map.of() : familySelected);
        familyProfiles = Map.copyOf(familyProfiles == null ? Map.of() : familyProfiles);
        familyPaths = Map.copyOf(familyPaths == null ? Map.of() : familyPaths);
    }

    public static record BundlePreset(
            String id,
            String description,
            List<String> selectors,
            List<String> requiredFamilies,
            List<String> forbiddenFamilies,
            List<String> requiredAliases,
            List<String> forbiddenAliases,
            List<String> selectedFamilies,
            Boolean policyPassed,
            int policyViolationCount,
            BundlePolicyViolations policyViolations) {

        public BundlePreset {
            id = normalizeFamilyId(id);
            description = description == null ? "" : description.trim();
            selectors = normalizeFamilyIds(selectors);
            requiredFamilies = normalizeFamilyIds(requiredFamilies);
            forbiddenFamilies = normalizeFamilyIds(forbiddenFamilies);
            requiredAliases = normalizeFamilyIds(requiredAliases);
            forbiddenAliases = normalizeFamilyIds(forbiddenAliases);
            selectedFamilies = normalizeFamilyIds(selectedFamilies);
            policyViolationCount = Math.max(0, policyViolationCount);
            policyViolations = policyViolations == null ? BundlePolicyViolations.empty() : policyViolations;
        }

        public int selectedCount() {
            return selectedFamilies.size();
        }

        public boolean policyStatusKnown() {
            return policyPassed != null;
        }

        public String policyStatusLabel() {
            if (!policyStatusKnown()) {
                return "unknown";
            }
            return Boolean.TRUE.equals(policyPassed) ? "passed" : "failed";
        }

        public String compactSummary() {
            return "%s(selectors=%s, selected=%d, policy=%s, requiredAliases=%s, forbiddenAliases=%s)".formatted(
                    id,
                    joinOrDash(selectors),
                    selectedCount(),
                    policyStatusLabel(),
                    joinOrDash(requiredAliases),
                    joinOrDash(forbiddenAliases));
        }
    }

    public static record BundlePolicy(
            List<String> requiredFamilies,
            List<String> forbiddenFamilies,
            List<String> requiredAliases,
            List<String> forbiddenAliases,
            Boolean passed,
            int violationCount,
            BundlePolicyViolations violations) {

        public BundlePolicy {
            requiredFamilies = normalizeFamilyIds(requiredFamilies);
            forbiddenFamilies = normalizeFamilyIds(forbiddenFamilies);
            requiredAliases = normalizeFamilyIds(requiredAliases);
            forbiddenAliases = normalizeFamilyIds(forbiddenAliases);
            violationCount = Math.max(0, violationCount);
            violations = violations == null ? BundlePolicyViolations.empty() : violations;
        }

        public static BundlePolicy empty() {
            return new BundlePolicy(List.of(), List.of(), List.of(), List.of(), null, 0, BundlePolicyViolations.empty());
        }

        public boolean statusKnown() {
            return passed != null;
        }

        public String statusLabel() {
            if (!statusKnown()) {
                return "unknown";
            }
            return Boolean.TRUE.equals(passed) ? "passed" : "failed";
        }

        public String compactStatus() {
            return "%s (%d violation(s))".formatted(statusLabel(), violationCount);
        }
    }

    public static record BundlePolicyViolations(
            List<String> missingRequiredFamilies,
            List<String> selectedForbiddenFamilies,
            Map<String, List<String>> missingRequiredAliases,
            Map<String, List<String>> selectedForbiddenAliases) {

        public BundlePolicyViolations {
            missingRequiredFamilies = normalizeFamilyIds(missingRequiredFamilies);
            selectedForbiddenFamilies = normalizeFamilyIds(selectedForbiddenFamilies);
            missingRequiredAliases = normalizeAliasFamilyMap(missingRequiredAliases);
            selectedForbiddenAliases = normalizeAliasFamilyMap(selectedForbiddenAliases);
        }

        public static BundlePolicyViolations empty() {
            return new BundlePolicyViolations(List.of(), List.of(), Map.of(), Map.of());
        }
    }

    public static record FixtureStatus(
            List<String> requiredSelectors,
            List<String> requiredFamilies,
            Boolean passed,
            String requiredFingerprint,
            String inventoryFingerprint,
            int availableFamilyCount,
            int fixtureFamilyCount,
            int requiredFamilyCount,
            int requiredPassedCount,
            int missingRequiredCount,
            int problemFamilyCount,
            List<String> missingRequiredFamilies,
            List<String> problemFamilies) {

        public FixtureStatus {
            requiredSelectors = normalizeFamilyIds(requiredSelectors);
            requiredFamilies = normalizeFamilyIds(requiredFamilies);
            requiredFingerprint = requiredFingerprint == null || requiredFingerprint.isBlank()
                    ? "-"
                    : requiredFingerprint.trim();
            inventoryFingerprint = inventoryFingerprint == null || inventoryFingerprint.isBlank()
                    ? "-"
                    : inventoryFingerprint.trim();
            availableFamilyCount = Math.max(0, availableFamilyCount);
            fixtureFamilyCount = Math.max(0, fixtureFamilyCount);
            requiredFamilyCount = Math.max(0, requiredFamilyCount);
            requiredPassedCount = Math.max(0, requiredPassedCount);
            missingRequiredCount = Math.max(0, missingRequiredCount);
            problemFamilyCount = Math.max(0, problemFamilyCount);
            missingRequiredFamilies = normalizeFamilyIds(missingRequiredFamilies);
            problemFamilies = normalizeFamilyIds(problemFamilies);
        }

        public static FixtureStatus empty() {
            return new FixtureStatus(List.of(), List.of(), null, "-", "-", 0, 0, 0, 0, 0, 0, List.of(), List.of());
        }

        public boolean statusKnown() {
            return passed != null;
        }

        public String statusLabel() {
            if (!statusKnown()) {
                return "unknown";
            }
            return Boolean.TRUE.equals(passed) ? "passed" : "failed";
        }

        public String compactStatus() {
            return "%s (%d/%d required, %d missing, %d problem)".formatted(
                    statusLabel(),
                    requiredPassedCount,
                    requiredFamilyCount,
                    missingRequiredCount,
                    problemFamilyCount);
        }
    }

    public static record BundlePresetConformance(
            String presetId,
            boolean presetMetadataPresent,
            boolean selectorsMatch,
            boolean policyInputsMatch,
            boolean explicitSelectorOverride,
            boolean explicitPolicyOverride,
            List<String> selectorAdditions,
            List<String> selectorOmissions,
            List<String> requiredFamilyAdditions,
            List<String> requiredFamilyOmissions,
            List<String> forbiddenFamilyAdditions,
            List<String> forbiddenFamilyOmissions,
            List<String> requiredAliasAdditions,
            List<String> requiredAliasOmissions,
            List<String> forbiddenAliasAdditions,
            List<String> forbiddenAliasOmissions) {

        public BundlePresetConformance {
            presetId = normalizeFamilyId(presetId);
            selectorAdditions = normalizeFamilyIds(selectorAdditions);
            selectorOmissions = normalizeFamilyIds(selectorOmissions);
            requiredFamilyAdditions = normalizeFamilyIds(requiredFamilyAdditions);
            requiredFamilyOmissions = normalizeFamilyIds(requiredFamilyOmissions);
            forbiddenFamilyAdditions = normalizeFamilyIds(forbiddenFamilyAdditions);
            forbiddenFamilyOmissions = normalizeFamilyIds(forbiddenFamilyOmissions);
            requiredAliasAdditions = normalizeFamilyIds(requiredAliasAdditions);
            requiredAliasOmissions = normalizeFamilyIds(requiredAliasOmissions);
            forbiddenAliasAdditions = normalizeFamilyIds(forbiddenAliasAdditions);
            forbiddenAliasOmissions = normalizeFamilyIds(forbiddenAliasOmissions);
        }

        public static BundlePresetConformance none() {
            return new BundlePresetConformance(
                    "",
                    false,
                    false,
                    false,
                    false,
                    false,
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of());
        }

        public boolean hasPreset() {
            return !presetId.isBlank();
        }

        public boolean matchesPreset() {
            return presetMetadataPresent && selectorsMatch && policyInputsMatch;
        }

        public boolean cleanPresetBuild() {
            return matchesPreset() && !explicitSelectorOverride && !explicitPolicyOverride;
        }

        public String statusLabel() {
            if (!hasPreset()) {
                return "none";
            }
            if (!presetMetadataPresent) {
                return "unknown";
            }
            if (cleanPresetBuild()) {
                return "clean";
            }
            if (matchesPreset()) {
                return "matched";
            }
            return "drifted";
        }

        public String compactSummary() {
            if (!hasPreset()) {
                return "none";
            }
            if (!presetMetadataPresent) {
                return "unknown (preset metadata missing)";
            }
            if (cleanPresetBuild()) {
                return "clean";
            }
            if (matchesPreset()) {
                return "matched with explicit overrides";
            }

            List<String> reasons = new ArrayList<>();
            if (!selectorsMatch) {
                reasons.add("selectors differ");
            }
            if (!policyInputsMatch) {
                reasons.add("policy inputs differ");
            }
            if (explicitSelectorOverride) {
                reasons.add("explicit selector override");
            }
            if (explicitPolicyOverride) {
                reasons.add("explicit policy override");
            }
            return "drifted (" + String.join("; ", reasons) + ")";
        }
    }

    public static record BundleAlias(String id, String description, List<String> families) {

        public BundleAlias {
            id = normalizeFamilyId(id);
            description = description == null ? "" : description.trim();
            List<String> normalizedFamilies = families == null
                    ? List.of()
                    : families.stream()
                            .map(ModelFamilyBundleManifest::normalizeFamilyId)
                            .filter(family -> !family.isBlank())
                            .distinct()
                            .toList();
            families = List.copyOf(normalizedFamilies);
        }

        public String compactSummary() {
            return "%s(%d families)".formatted(id, families.size());
        }
    }

    public static record BundleAliasCoverage(
            String id,
            String description,
            List<String> families,
            List<String> selectedFamilies,
            List<String> missingFamilies) {

        public BundleAliasCoverage {
            id = normalizeFamilyId(id);
            description = description == null ? "" : description.trim();
            families = normalizeFamilyIds(families);
            selectedFamilies = normalizeFamilyIds(selectedFamilies);
            missingFamilies = normalizeFamilyIds(missingFamilies);
        }

        public boolean complete() {
            return !families.isEmpty() && missingFamilies.isEmpty();
        }

        public boolean partial() {
            return !complete() && !selectedFamilies.isEmpty();
        }

        public int familyCount() {
            return families.size();
        }

        public int selectedCount() {
            return selectedFamilies.size();
        }

        public int missingCount() {
            return missingFamilies.size();
        }

        public String compactSummary() {
            String status = complete() ? "complete" : partial() ? "partial" : "empty";
            return "%s(%d/%d %s)".formatted(id, selectedCount(), familyCount(), status);
        }
    }

    public static ModelFamilyBundleManifest load() {
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader == null) {
            loader = ModelFamilyBundleManifest.class.getClassLoader();
        }
        try (InputStream input = loader.getResourceAsStream(RESOURCE_PATH)) {
            if (input == null) {
                return empty();
            }
            Properties properties = new Properties();
            properties.load(input);
            return fromProperties(properties);
        } catch (Exception ignored) {
            return empty();
        }
    }

    public static ModelFamilyBundleManifest fromProperties(Properties properties) {
        if (properties == null || properties.isEmpty()) {
            return empty();
        }
        List<String> selectors = csvProperty(properties, "selectors");
        List<String> families = csvProperty(properties, "families");
        List<String> profiles = csvProperty(properties, "profiles");
        List<String> availableFamilies = csvProperty(properties, "availableFamilies");
        List<String> availableProfiles = csvProperty(properties, "availableProfiles");
        List<String> availableSelectors = csvProperty(properties, "availableSelectors");
        List<BundlePreset> bundlePresets = bundlePresets(properties);
        List<BundleAlias> bundleAliases = bundleAliases(properties);
        return new ModelFamilyBundleManifest(
                true,
                schemaVersion(properties),
                properties.getProperty("bundleFingerprint", ""),
                detached(properties, selectors, families),
                selectors,
                families,
                profiles,
                availableFamilies,
                availableProfiles,
                availableSelectors,
                properties.getProperty("selectorSource", ""),
                csvProperty(properties, "explicitSelectors"),
                csvProperty(properties, "presetSelectors"),
                csvProperty(properties, "defaultSelectors"),
                properties.getProperty("policySource", ""),
                csvProperty(properties, "presetRequiredFamilies"),
                csvProperty(properties, "presetForbiddenFamilies"),
                csvProperty(properties, "presetRequiredAliases"),
                csvProperty(properties, "presetForbiddenAliases"),
                csvProperty(properties, "explicitRequiredFamilies"),
                csvProperty(properties, "explicitForbiddenFamilies"),
                csvProperty(properties, "explicitRequiredAliases"),
                csvProperty(properties, "explicitForbiddenAliases"),
                properties.getProperty("bundlePreset", ""),
                bundlePolicy(properties),
                fixtureStatus(properties),
                bundlePresets,
                bundleAliases,
                familySelectedMap(properties, families, availableFamilies),
                prefixedFamilyMap(properties, ".profile"),
                prefixedFamilyMap(properties, ".path"));
    }

    public static ModelFamilyBundleManifest empty() {
        return new ModelFamilyBundleManifest(
                false,
                0,
                "-",
                false,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                "",
                List.of(),
                List.of(),
                List.of(),
                "",
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                "",
                BundlePolicy.empty(),
                FixtureStatus.empty(),
                List.of(),
                List.of(),
                Map.of(),
                Map.of(),
                Map.of());
    }

    public List<String> missingDiscovered(Set<String> discoveredFamilyIds) {
        Set<String> discovered = discoveredFamilyIds == null
                ? Set.of()
                : discoveredFamilyIds.stream()
                        .map(ModelFamilyBundleManifest::normalizeFamilyId)
                        .filter(id -> !id.isBlank())
                        .collect(java.util.stream.Collectors.toSet());
        return families.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(id -> !id.isBlank() && !discovered.contains(id))
                .toList();
    }

    public String joinedSelectors() {
        return joinOrDash(selectors);
    }

    public String displaySelectorSource() {
        return switch (selectorSource) {
            case "explicit" -> "explicit -Pgollek.modelFamilies";
            case "preset" -> hasBundlePreset() ? "preset " + bundlePreset : "preset";
            case "default" -> "default";
            default -> "unknown";
        };
    }

    public String joinedExplicitSelectors() {
        return joinOrDash(explicitSelectors);
    }

    public String joinedPresetSelectors() {
        return joinOrDash(presetSelectors);
    }

    public String joinedDefaultSelectors() {
        return joinOrDash(defaultSelectors);
    }

    public String displayPolicySource() {
        return switch (policySource) {
            case "mixed" -> "mixed preset+explicit";
            case "preset" -> hasBundlePreset() ? "preset " + bundlePreset : "preset";
            case "explicit" -> "explicit Gradle policy flags";
            case "none" -> "none";
            default -> "unknown";
        };
    }

    public List<String> requestedAliases() {
        return requestedSelectors(bundleAliases.stream()
                .map(BundleAlias::id)
                .toList());
    }

    public String joinedRequestedAliases() {
        return joinOrDash(requestedAliases());
    }

    public List<String> requestedProfiles() {
        return requestedSelectors(availableProfiles);
    }

    public String joinedRequestedProfiles() {
        return joinOrDash(requestedProfiles());
    }

    public List<String> requestedFamilies() {
        return requestedSelectors(availableFamilies);
    }

    public String joinedRequestedFamilies() {
        return joinOrDash(requestedFamilies());
    }

    public List<String> reservedSelectors() {
        return selectors.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(selector -> selector.equals("all") || selector.equals("none"))
                .distinct()
                .toList();
    }

    public List<String> unknownSelectors() {
        Set<String> knownSelectors = java.util.stream.Stream.of(
                        availableFamilies.stream(),
                        availableProfiles.stream(),
                        bundleAliases.stream().map(BundleAlias::id),
                        java.util.stream.Stream.of("all", "none"))
                .flatMap(stream -> stream)
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .collect(java.util.stream.Collectors.toSet());
        return selectors.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(selector -> !selector.isBlank() && !knownSelectors.contains(selector))
                .distinct()
                .toList();
    }

    public String displayFingerprint() {
        return bundleFingerprint;
    }

    public String joinedFamilies() {
        return joinOrDash(families);
    }

    public String displayFamilies() {
        return detached ? "(detached)" : joinedFamilies();
    }

    public String joinedProfiles() {
        return joinOrDash(profiles);
    }

    public String displayProfiles() {
        return detached ? "(detached)" : joinedProfiles();
    }

    public boolean isFamilySelected(String familyId) {
        String normalized = normalizeFamilyId(familyId);
        return Boolean.TRUE.equals(familySelected.get(normalized));
    }

    public List<String> omittedFamilies() {
        if (availableFamilies.isEmpty()) {
            return List.of();
        }
        return availableFamilies.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(id -> !id.isBlank() && !isFamilySelected(id))
                .toList();
    }

    public String joinedOmittedFamiliesWithProfiles() {
        List<String> omitted = omittedFamiliesWithProfiles();
        if (omitted.isEmpty()) {
            return "none";
        }
        return omitted.stream()
                .reduce((a, b) -> a + ", " + b)
                .orElse("none");
    }

    public List<String> omittedFamiliesWithProfiles() {
        return omittedFamilies().stream()
                .map(this::familyWithProfile)
                .toList();
    }

    public String joinedAvailableFamilies() {
        return joinOrDash(availableFamilies);
    }

    public String joinedAvailableProfiles() {
        return joinOrDash(availableProfiles);
    }

    public String joinedAvailableSelectors() {
        if (!availableSelectors.isEmpty()) {
            return joinOrDash(availableSelectors);
        }
        return joinOrDash(java.util.stream.Stream.concat(availableProfiles.stream(), availableFamilies.stream())
                .distinct()
                .sorted()
                .toList());
    }

    public boolean hasBundlePreset() {
        return bundlePreset != null && !bundlePreset.isBlank();
    }

    public String displayBundlePreset() {
        if (!hasBundlePreset()) {
            return "none";
        }
        return activeBundlePreset()
                .map(preset -> preset.id() + " - " + preset.description())
                .orElse(bundlePreset);
    }

    public Optional<BundlePreset> activeBundlePreset() {
        if (!hasBundlePreset()) {
            return Optional.empty();
        }
        return bundlePresets.stream()
                .filter(preset -> preset.id().equals(bundlePreset))
                .findFirst();
    }

    public String displayBundlePresetPolicyStatus() {
        if (!hasBundlePreset()) {
            return "none";
        }
        return activeBundlePreset()
                .map(preset -> "%s (%d violation(s), %d selected families)".formatted(
                        preset.policyStatusLabel(),
                        preset.policyViolationCount(),
                        preset.selectedCount()))
                .orElse("unknown (preset metadata missing)");
    }

    public BundlePresetConformance activeBundlePresetConformance() {
        if (!hasBundlePreset()) {
            return BundlePresetConformance.none();
        }

        Optional<BundlePreset> preset = activeBundlePreset();
        if (preset.isEmpty()) {
            return new BundlePresetConformance(
                    bundlePreset,
                    false,
                    false,
                    false,
                    "explicit".equals(selectorSource),
                    isExplicitPolicySource(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of());
        }

        BundlePreset activePreset = preset.get();
        List<String> currentRequiredFamilies = mergeNormalized(presetRequiredFamilies, explicitRequiredFamilies);
        List<String> currentForbiddenFamilies = mergeNormalized(presetForbiddenFamilies, explicitForbiddenFamilies);
        List<String> currentRequiredAliases = mergeNormalized(presetRequiredAliases, explicitRequiredAliases);
        List<String> currentForbiddenAliases = mergeNormalized(presetForbiddenAliases, explicitForbiddenAliases);
        boolean selectorsMatch = sameValues(selectors, activePreset.selectors());
        boolean policyInputsMatch = sameValues(currentRequiredFamilies, activePreset.requiredFamilies())
                && sameValues(currentForbiddenFamilies, activePreset.forbiddenFamilies())
                && sameValues(currentRequiredAliases, activePreset.requiredAliases())
                && sameValues(currentForbiddenAliases, activePreset.forbiddenAliases());

        return new BundlePresetConformance(
                activePreset.id(),
                true,
                selectorsMatch,
                policyInputsMatch,
                "explicit".equals(selectorSource),
                isExplicitPolicySource(),
                valuesMissingFrom(selectors, activePreset.selectors()),
                valuesMissingFrom(activePreset.selectors(), selectors),
                valuesMissingFrom(currentRequiredFamilies, activePreset.requiredFamilies()),
                valuesMissingFrom(activePreset.requiredFamilies(), currentRequiredFamilies),
                valuesMissingFrom(currentForbiddenFamilies, activePreset.forbiddenFamilies()),
                valuesMissingFrom(activePreset.forbiddenFamilies(), currentForbiddenFamilies),
                valuesMissingFrom(currentRequiredAliases, activePreset.requiredAliases()),
                valuesMissingFrom(activePreset.requiredAliases(), currentRequiredAliases),
                valuesMissingFrom(currentForbiddenAliases, activePreset.forbiddenAliases()),
                valuesMissingFrom(activePreset.forbiddenAliases(), currentForbiddenAliases));
    }

    public String displayActiveBundlePresetConformance() {
        return activeBundlePresetConformance().compactSummary();
    }

    public String displayBundlePolicyStatus() {
        return bundlePolicy.compactStatus();
    }

    public String displayFixtureStatus() {
        return fixtureStatus.compactStatus();
    }

    public List<String> availableBundlePresets() {
        return bundlePresets.stream()
                .map(BundlePreset::id)
                .distinct()
                .toList();
    }

    public String joinedAvailableBundlePresets() {
        return joinOrDash(availableBundlePresets());
    }

    public String joinedBundlePresets() {
        if (bundlePresets.isEmpty()) {
            return "none";
        }
        return bundlePresets.stream()
                .map(BundlePreset::compactSummary)
                .reduce((a, b) -> a + ", " + b)
                .orElse("none");
    }

    public String joinedBundleAliases() {
        if (bundleAliases.isEmpty()) {
            return "none";
        }
        return bundleAliases.stream()
                .map(BundleAlias::compactSummary)
                .reduce((a, b) -> a + ", " + b)
                .orElse("none");
    }

    public List<BundleAliasCoverage> bundleAliasCoverage() {
        return bundleAliases.stream()
                .map(alias -> {
                    List<String> selected = alias.families().stream()
                            .filter(this::isFamilySelected)
                            .toList();
                    List<String> missing = alias.families().stream()
                            .filter(family -> !isFamilySelected(family))
                            .toList();
                    return new BundleAliasCoverage(alias.id(), alias.description(), alias.families(), selected, missing);
                })
                .toList();
    }

    public List<BundleAliasCoverage> completeBundleAliases() {
        return bundleAliasCoverage().stream()
                .filter(BundleAliasCoverage::complete)
                .toList();
    }

    public List<BundleAliasCoverage> partialBundleAliases() {
        return bundleAliasCoverage().stream()
                .filter(BundleAliasCoverage::partial)
                .toList();
    }

    public String joinedCompleteBundleAliases() {
        return joinedAliasCoverage(completeBundleAliases());
    }

    public String joinedPartialBundleAliases() {
        return joinedAliasCoverage(partialBundleAliases());
    }

    private static String joinedAliasCoverage(List<BundleAliasCoverage> coverage) {
        if (coverage.isEmpty()) {
            return "none";
        }
        return coverage.stream()
                .map(BundleAliasCoverage::compactSummary)
                .reduce((a, b) -> a + ", " + b)
                .orElse("none");
    }

    private static List<BundleAlias> bundleAliases(Properties properties) {
        return csvProperty(properties, "bundleAliases").stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(aliasId -> !aliasId.isBlank())
                .distinct()
                .map(aliasId -> new BundleAlias(
                        aliasId,
                        properties.getProperty("bundleAlias." + aliasId + ".description", ""),
                        csvProperty(properties, "bundleAlias." + aliasId + ".families")))
                .toList();
    }

    private static BundlePolicy bundlePolicy(Properties properties) {
        return new BundlePolicy(
                csvProperty(properties, "requiredFamilies"),
                csvProperty(properties, "forbiddenFamilies"),
                csvProperty(properties, "requiredAliases"),
                csvProperty(properties, "forbiddenAliases"),
                nullableBooleanProperty(properties, "policyPassed"),
                intProperty(properties, "policyViolationCount"),
                new BundlePolicyViolations(
                        csvProperty(properties, "missingRequiredFamilies"),
                        csvProperty(properties, "selectedForbiddenFamilies"),
                        prefixedAliasFamilyMap(properties, "missingRequiredAlias."),
                        prefixedAliasFamilyMap(properties, "selectedForbiddenAlias.")));
    }

    private static FixtureStatus fixtureStatus(Properties properties) {
        return new FixtureStatus(
                csvProperty(properties, "fixtureRequiredSelectors"),
                csvProperty(properties, "fixtureRequiredFamilies"),
                nullableBooleanProperty(properties, "fixturePassed"),
                properties.getProperty("fixtureRequiredFingerprint", ""),
                properties.getProperty("fixtureInventoryFingerprint", ""),
                intProperty(properties, "fixtureAvailableFamilyCount"),
                intProperty(properties, "fixtureFamilyCount"),
                intProperty(properties, "fixtureRequiredFamilyCount"),
                intProperty(properties, "fixtureRequiredPassedCount"),
                intProperty(properties, "fixtureMissingRequiredCount"),
                intProperty(properties, "fixtureProblemFamilyCount"),
                csvProperty(properties, "fixtureMissingRequiredFamilies"),
                csvProperty(properties, "fixtureProblemFamilies"));
    }

    private static List<BundlePreset> bundlePresets(Properties properties) {
        List<String> presetIds = csvProperty(properties, "bundlePresets");
        if (presetIds.isEmpty()) {
            presetIds = csvProperty(properties, "availableBundlePresets");
        }
        return presetIds.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(presetId -> !presetId.isBlank())
                .distinct()
                .map(presetId -> new BundlePreset(
                        presetId,
                        properties.getProperty("bundlePreset." + presetId + ".description", ""),
                        csvProperty(properties, "bundlePreset." + presetId + ".selectors"),
                        csvProperty(properties, "bundlePreset." + presetId + ".requiredFamilies"),
                        csvProperty(properties, "bundlePreset." + presetId + ".forbiddenFamilies"),
                        csvProperty(properties, "bundlePreset." + presetId + ".requiredAliases"),
                        csvProperty(properties, "bundlePreset." + presetId + ".forbiddenAliases"),
                        csvProperty(properties, "bundlePreset." + presetId + ".selectedFamilies"),
                        nullableBooleanProperty(properties, "bundlePreset." + presetId + ".policyPassed"),
                        intProperty(properties, "bundlePreset." + presetId + ".policyViolationCount"),
                        new BundlePolicyViolations(
                                csvProperty(properties, "bundlePreset." + presetId + ".missingRequiredFamilies"),
                                csvProperty(properties, "bundlePreset." + presetId + ".selectedForbiddenFamilies"),
                                prefixedAliasFamilyMap(properties,
                                        "bundlePreset." + presetId + ".missingRequiredAlias."),
                                prefixedAliasFamilyMap(properties,
                                        "bundlePreset." + presetId + ".selectedForbiddenAlias."))))
                .toList();
    }

    private List<String> requestedSelectors(List<String> available) {
        Set<String> availableSet = available.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .collect(java.util.stream.Collectors.toSet());
        return selectors.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(selector -> !selector.isBlank() && availableSet.contains(selector))
                .distinct()
                .toList();
    }

    private static Map<String, String> prefixedFamilyMap(Properties properties, String suffix) {
        Map<String, String> values = new LinkedHashMap<>();
        String prefix = "family.";
        for (String key : properties.stringPropertyNames().stream().sorted().toList()) {
            if (!key.startsWith(prefix) || !key.endsWith(suffix)) {
                continue;
            }
            String familyId = normalizeFamilyId(key.substring(prefix.length(), key.length() - suffix.length()));
            String value = Objects.toString(properties.getProperty(key), "").trim();
            if (!familyId.isBlank() && !value.isBlank()) {
                values.put(familyId, value);
            }
        }
        return Map.copyOf(values);
    }

    private static List<String> csvProperty(Properties properties, String key) {
        String value = properties.getProperty(key, "");
        if (value.isBlank()) {
            return List.of();
        }
        return Arrays.stream(value.split(","))
                .map(String::trim)
                .filter(item -> !item.isBlank())
                .toList();
    }

    private static Boolean nullableBooleanProperty(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            return null;
        }
        return Boolean.parseBoolean(value.trim());
    }

    private static int intProperty(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            return 0;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private static Map<String, List<String>> prefixedAliasFamilyMap(Properties properties, String prefix) {
        Map<String, List<String>> values = new LinkedHashMap<>();
        String suffix = ".families";
        for (String key : properties.stringPropertyNames().stream().sorted().toList()) {
            if (!key.startsWith(prefix) || !key.endsWith(suffix)) {
                continue;
            }
            String aliasId = normalizeFamilyId(key.substring(prefix.length(), key.length() - suffix.length()));
            List<String> families = csvProperty(properties, key);
            if (!aliasId.isBlank() && !families.isEmpty()) {
                values.put(aliasId, families);
            }
        }
        return Map.copyOf(values);
    }

    private static boolean detached(Properties properties, List<String> selectors, List<String> families) {
        String explicit = properties.getProperty("detached");
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit.trim());
        }
        return selectors.stream().anyMatch("none"::equalsIgnoreCase) && families.isEmpty();
    }

    private static int schemaVersion(Properties properties) {
        String value = properties.getProperty("schemaVersion");
        if (value == null || value.isBlank()) {
            return 0;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private static Map<String, Boolean> familySelectedMap(
            Properties properties,
            List<String> families,
            List<String> availableFamilies) {
        Map<String, Boolean> selected = new LinkedHashMap<>();
        Set<String> bundled = families.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(id -> !id.isBlank())
                .collect(java.util.stream.Collectors.toSet());
        for (String family : availableFamilies) {
            String familyId = normalizeFamilyId(family);
            if (!familyId.isBlank()) {
                selected.put(familyId, bundled.contains(familyId));
            }
        }
        for (String familyId : bundled) {
            selected.putIfAbsent(familyId, true);
        }
        for (String key : properties.stringPropertyNames().stream().sorted().toList()) {
            String suffix = ".selected";
            if (!key.startsWith("family.") || !key.endsWith(suffix)) {
                continue;
            }
            String familyId = normalizeFamilyId(key.substring("family.".length(), key.length() - suffix.length()));
            if (!familyId.isBlank()) {
                selected.put(familyId, Boolean.parseBoolean(properties.getProperty(key, "false").trim()));
            }
        }
        return Map.copyOf(selected);
    }

    private String familyWithProfile(String familyId) {
        String normalized = normalizeFamilyId(familyId);
        String profile = familyProfiles.get(normalized);
        return profile == null || profile.isBlank() ? normalized : normalized + "[" + profile + "]";
    }

    private static String normalizeFamilyId(String id) {
        String normalized = Objects.toString(id, "").trim().toLowerCase(Locale.ROOT);
        return normalized.startsWith("model-family/") ? normalized.substring("model-family/".length()) : normalized;
    }

    private static String normalizeSelectorSource(String source) {
        String normalized = Objects.toString(source, "").trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "explicit", "preset", "default" -> normalized;
            default -> "unknown";
        };
    }

    private static String normalizePolicySource(String source) {
        String normalized = Objects.toString(source, "").trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "none", "preset", "explicit", "mixed" -> normalized;
            default -> "unknown";
        };
    }

    private boolean isExplicitPolicySource() {
        return policySource.equals("explicit") || policySource.equals("mixed");
    }

    private static List<String> normalizeFamilyIds(List<String> ids) {
        if (ids == null) {
            return List.of();
        }
        return ids.stream()
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(id -> !id.isBlank())
                .distinct()
                .toList();
    }

    private static List<String> mergeNormalized(List<String> first, List<String> second) {
        return java.util.stream.Stream.concat(
                first == null ? java.util.stream.Stream.empty() : first.stream(),
                second == null ? java.util.stream.Stream.empty() : second.stream())
                .map(ModelFamilyBundleManifest::normalizeFamilyId)
                .filter(id -> !id.isBlank())
                .distinct()
                .toList();
    }

    private static boolean sameValues(List<String> first, List<String> second) {
        return normalizedSet(first).equals(normalizedSet(second));
    }

    private static List<String> valuesMissingFrom(List<String> values, List<String> expected) {
        Set<String> expectedSet = normalizedSet(expected);
        return normalizeFamilyIds(values).stream()
                .filter(value -> !expectedSet.contains(value))
                .toList();
    }

    private static Set<String> normalizedSet(List<String> values) {
        return new LinkedHashSet<>(normalizeFamilyIds(values));
    }

    private static Map<String, List<String>> normalizeAliasFamilyMap(Map<String, List<String>> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, List<String>> normalized = new LinkedHashMap<>();
        values.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> {
                    String aliasId = normalizeFamilyId(entry.getKey());
                    List<String> families = normalizeFamilyIds(entry.getValue());
                    if (!aliasId.isBlank() && !families.isEmpty()) {
                        normalized.put(aliasId, families);
                    }
                });
        return Map.copyOf(normalized);
    }

    private static String joinOrDash(List<String> values) {
        return values.isEmpty() ? "-" : String.join(", ", values);
    }
}
