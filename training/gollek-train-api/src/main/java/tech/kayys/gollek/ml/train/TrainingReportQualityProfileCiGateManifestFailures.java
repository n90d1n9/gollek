package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainingReportQualityProfileCiGateManifestFailures {
    private TrainingReportQualityProfileCiGateManifestFailures() {
    }

    static List<TrainingReportQualityProfileCiGateManifestFailure> categorize(List<String> failures) {
        if (failures == null || failures.isEmpty()) {
            return List.of();
        }
        List<TrainingReportQualityProfileCiGateManifestFailure> categorized = new ArrayList<>();
        for (String failure : failures) {
            if (failure == null || failure.isBlank()) {
                continue;
            }
            categorized.add(new TrainingReportQualityProfileCiGateManifestFailure(
                    classify(failure),
                    failure));
        }
        return List.copyOf(categorized);
    }

    static List<String> messages(
            List<String> failures,
            TrainingReportQualityProfileCiGateManifestFailureCategory category) {
        if (failures == null || failures.isEmpty()) {
            return List.of();
        }
        List<String> messages = new ArrayList<>();
        for (TrainingReportQualityProfileCiGateManifestFailure failure : categorize(failures)) {
            if (failure.category() == category) {
                messages.add(failure.message());
            }
        }
        return List.copyOf(messages);
    }

    static Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> countByCategory(
            List<String> failures) {
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts =
                new EnumMap<>(TrainingReportQualityProfileCiGateManifestFailureCategory.class);
        for (TrainingReportQualityProfileCiGateManifestFailure failure : categorize(failures)) {
            counts.merge(failure.category(), 1, Integer::sum);
        }
        return Map.copyOf(counts);
    }

    static Map<String, Object> countMapByCategory(List<String> failures) {
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts = countByCategory(failures);
        Map<String, Object> map = new LinkedHashMap<>();
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category
                : TrainingReportQualityProfileCiGateManifestFailureCategory.values()) {
            Integer count = counts.get(category);
            if (count != null && count > 0) {
                map.put(category.id(), count);
            }
        }
        return Map.copyOf(map);
    }

    private static TrainingReportQualityProfileCiGateManifestFailureCategory classify(String failure) {
        if (failure.contains("checksum mismatch")) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.CHECKSUM;
        }
        if (failure.startsWith("Manifest format mismatch")) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.FORMAT;
        }
        if (failure.startsWith("Manifest profile is unknown")) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.PROFILE;
        }
        if (failure.startsWith("Manifest JSON cannot be rendered")
                || failure.startsWith("Manifest Markdown does not match manifest JSON")) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.MARKDOWN;
        }
        if (TrainingReportQualityProfileCiGateManifestStructure.isFailure(failure)) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.STRUCTURE;
        }
        if (TrainingReportQualityProfileCiGateManifestArtifacts.isFailure(failure)) {
            return TrainingReportQualityProfileCiGateManifestFailureCategory.ARTIFACT;
        }
        return TrainingReportQualityProfileCiGateManifestFailureCategory.UNKNOWN;
    }
}
