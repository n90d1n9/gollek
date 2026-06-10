package tech.kayys.gollek.spi.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Aggregate runtime compatibility signal for a set of attached model families.
 */
public record ModelFamilyRuntimeCompatibilitySummary(
        String runtimeId,
        int familyCount,
        int compatibleFamilyCount,
        int blockedFamilyCount,
        int attentionFamilyCount,
        int architectureAdapterReadyCount,
        int tokenizerReadyCount,
        int tokenizerFileInspectionAvailableCount,
        List<String> compatibleFamilyIds,
        List<String> blockedFamilyIds,
        Map<String, Integer> problemCounts) {

    public ModelFamilyRuntimeCompatibilitySummary {
        runtimeId = runtimeId == null || runtimeId.isBlank() ? "unknown" : runtimeId.trim();
        compatibleFamilyIds = copyStrings(compatibleFamilyIds);
        blockedFamilyIds = copyStrings(blockedFamilyIds);
        problemCounts = problemCounts == null ? Map.of() : Map.copyOf(problemCounts);
    }

    public static ModelFamilyRuntimeCompatibilitySummary from(
            List<ModelFamilyRuntimeCompatibility> compatibilities) {
        String runtimeId = compatibilities == null || compatibilities.isEmpty()
                ? "unknown"
                : compatibilities.stream()
                        .filter(Objects::nonNull)
                        .map(ModelFamilyRuntimeCompatibility::runtimeId)
                        .filter(value -> value != null && !value.isBlank())
                        .findFirst()
                        .orElse("unknown");
        return from(runtimeId, compatibilities);
    }

    public static ModelFamilyRuntimeCompatibilitySummary from(
            String runtimeId,
            List<ModelFamilyRuntimeCompatibility> compatibilities) {
        List<ModelFamilyRuntimeCompatibility> safeCompatibilities = compatibilities == null
                ? List.of()
                : compatibilities.stream()
                        .filter(Objects::nonNull)
                        .toList();
        List<String> compatibleFamilyIds = safeCompatibilities.stream()
                .filter(ModelFamilyRuntimeCompatibility::compatible)
                .map(ModelFamilyRuntimeCompatibilitySummary::familyId)
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
        List<String> blockedFamilyIds = safeCompatibilities.stream()
                .filter(compatibility -> !compatibility.compatible())
                .map(ModelFamilyRuntimeCompatibilitySummary::familyId)
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
        Map<String, Integer> problemCounts = new TreeMap<>();
        safeCompatibilities.stream()
                .flatMap(compatibility -> compatibility.problemCodes().stream())
                .forEach(problem -> problemCounts.merge(problem, 1, Integer::sum));

        return new ModelFamilyRuntimeCompatibilitySummary(
                runtimeId,
                safeCompatibilities.size(),
                compatibleFamilyIds.size(),
                blockedFamilyIds.size(),
                (int) safeCompatibilities.stream()
                        .filter(ModelFamilyRuntimeCompatibility::requiresAttention)
                        .count(),
                (int) safeCompatibilities.stream()
                        .filter(ModelFamilyRuntimeCompatibility::architectureAdapterReady)
                        .count(),
                (int) safeCompatibilities.stream()
                        .filter(ModelFamilyRuntimeCompatibility::tokenizerReady)
                        .count(),
                (int) safeCompatibilities.stream()
                        .filter(ModelFamilyRuntimeCompatibility::tokenizerFileInspectionAvailable)
                        .count(),
                compatibleFamilyIds,
                blockedFamilyIds,
                new LinkedHashMap<>(problemCounts));
    }

    public boolean allCompatible() {
        return familyCount > 0 && blockedFamilyCount == 0;
    }

    public boolean empty() {
        return familyCount == 0;
    }

    private static String familyId(ModelFamilyRuntimeCompatibility compatibility) {
        return compatibility.modelFamily().primaryFamilyId().orElse("");
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
