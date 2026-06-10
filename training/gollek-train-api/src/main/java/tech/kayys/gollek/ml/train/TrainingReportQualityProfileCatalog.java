package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Objects;

/**
 * Export-friendly catalog for built-in trainer report quality profiles.
 */
public record TrainingReportQualityProfileCatalog(List<TrainingReportQualityProfile> profiles) {
    public static final String FORMAT = "gollek.training-report.quality-profiles.v1";

    public TrainingReportQualityProfileCatalog {
        profiles = profiles == null || profiles.isEmpty()
                ? TrainingReportQualityProfile.defaults()
                : List.copyOf(profiles);
    }

    public static TrainingReportQualityProfileCatalog defaults() {
        return new TrainingReportQualityProfileCatalog(TrainingReportQualityProfile.defaults());
    }

    public static TrainingReportQualityProfileCatalog fromMap(Map<String, ?> map) {
        Objects.requireNonNull(map, "map must not be null");
        Object format = map.get("format");
        if (!FORMAT.equals(String.valueOf(format))) {
            throw new IllegalArgumentException("Unsupported quality profile catalog format: " + format);
        }
        Object profilesValue = map.get("profiles");
        if (!(profilesValue instanceof Iterable<?> iterable)) {
            throw new IllegalArgumentException("Quality profile catalog must include a profiles array");
        }
        List<TrainingReportQualityProfile> profiles = new ArrayList<>();
        for (Object item : iterable) {
            if (!(item instanceof Map<?, ?> profileMap)) {
                throw new IllegalArgumentException("Quality profile entry must be an object");
            }
            Object id = profileMap.get("id");
            profiles.add(TrainingReportQualityProfile.require(String.valueOf(id)));
        }
        if (profiles.isEmpty()) {
            throw new IllegalArgumentException("Quality profile catalog must include at least one profile");
        }
        Object expectedCount = map.get("profileCount");
        if (expectedCount instanceof Number number && number.intValue() != profiles.size()) {
            throw new IllegalArgumentException("Quality profile catalog profileCount does not match profiles array");
        }
        return new TrainingReportQualityProfileCatalog(profiles);
    }

    public List<String> ids() {
        return profiles.stream()
                .map(TrainingReportQualityProfile::id)
                .toList();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", FORMAT);
        map.put("profileCount", profiles.size());
        map.put("profiles", profiles.stream()
                .map(TrainingReportQualityProfile::toMap)
                .toList());
        return Map.copyOf(map);
    }

    public String toJson() {
        return TrainerJson.toJson(toMap());
    }

    public String toMarkdown() {
        return TrainingReportQualityProfileMarkdown.render(this);
    }
}
