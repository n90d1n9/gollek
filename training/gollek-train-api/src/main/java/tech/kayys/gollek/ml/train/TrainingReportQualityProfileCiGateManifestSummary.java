package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Compact manifest verification summary for CLI output, dashboards, and release gates.
 */
public record TrainingReportQualityProfileCiGateManifestSummary(
        boolean passed,
        String status,
        Path directory,
        String profileId,
        int artifactCount,
        int failureCount,
        List<String> failedCategories,
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> failureCountsByCategory,
        Optional<TrainingReportQualityProfileCiGateManifestFailure> primaryFailure) {
    public TrainingReportQualityProfileCiGateManifestSummary {
        if (status == null || status.isBlank()) {
            throw new IllegalArgumentException("status must not be blank");
        }
        status = status.trim().toLowerCase();
        directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
        profileId = profileId == null ? "" : profileId.trim();
        if (artifactCount < 0) {
            throw new IllegalArgumentException("artifactCount must be non-negative");
        }
        if (failureCount < 0) {
            throw new IllegalArgumentException("failureCount must be non-negative");
        }
        failedCategories = failedCategories == null ? List.of() : List.copyOf(failedCategories);
        failureCountsByCategory = failureCountsByCategory == null
                ? Map.of()
                : Map.copyOf(failureCountsByCategory);
        primaryFailure = primaryFailure == null ? Optional.empty() : primaryFailure;
    }

    public static TrainingReportQualityProfileCiGateManifestSummary from(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        List<TrainingReportQualityProfileCiGateManifestFailure> failures = verification.structuredFailures();
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts =
                verification.failureCountsByCategory();
        return new TrainingReportQualityProfileCiGateManifestSummary(
                verification.passed(),
                verification.passed() ? "passed" : "failed",
                verification.inspection().directory(),
                verification.inspection().profileId().orElse(""),
                verification.inspection().artifactMaps().size(),
                failures.size(),
                failedCategories(counts),
                counts,
                failures.isEmpty() ? Optional.empty() : Optional.of(failures.get(0)));
    }

    public boolean readyForRelease() {
        return passed;
    }

    public boolean hasFailures(TrainingReportQualityProfileCiGateManifestFailureCategory category) {
        return count(category) > 0;
    }

    public int count(TrainingReportQualityProfileCiGateManifestFailureCategory category) {
        return failureCountsByCategory.getOrDefault(category, 0);
    }

    public Optional<String> primaryFailureCategory() {
        return primaryFailure.map(failure -> failure.category().id());
    }

    public Optional<String> primaryFailureMessage() {
        return primaryFailure.map(TrainingReportQualityProfileCiGateManifestFailure::message);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("passed", passed);
        map.put("readyForRelease", readyForRelease());
        map.put("status", status);
        map.put("directory", directory.toString());
        if (!profileId.isBlank()) {
            map.put("profileId", profileId);
        }
        map.put("artifactCount", artifactCount);
        map.put("failureCount", failureCount);
        map.put("failedCategoryCount", failedCategories.size());
        map.put("failedCategories", failedCategories);
        map.put("failureCountsByCategory", failureCountMap());
        primaryFailure.ifPresent(failure -> map.put("primaryFailure", failure.toMap()));
        primaryFailureCategory().ifPresent(value -> map.put("primaryFailureCategory", value));
        primaryFailureMessage().ifPresent(value -> map.put("primaryFailureMessage", value));
        return Map.copyOf(map);
    }

    private Map<String, Object> failureCountMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category
                : TrainingReportQualityProfileCiGateManifestFailureCategory.values()) {
            int count = count(category);
            if (count > 0) {
                map.put(category.id(), count);
            }
        }
        return Map.copyOf(map);
    }

    private static List<String> failedCategories(
            Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts) {
        List<String> categories = new ArrayList<>();
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category
                : TrainingReportQualityProfileCiGateManifestFailureCategory.values()) {
            if (counts.getOrDefault(category, 0) > 0) {
                categories.add(category.id());
            }
        }
        return List.copyOf(categories);
    }
}
