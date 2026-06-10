package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Optional;

/**
 * Export-friendly catalog for built-in and custom trainer report quality profiles.
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
        TrainingReportQualityProfileCatalogValidator.Result validation =
                TrainingReportQualityProfileCatalogValidator.validate(map);
        if (!validation.passed()) {
            throw new IllegalArgumentException(validation.message());
        }
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
            profiles.add(TrainingReportQualityProfile.fromMap(
                    TrainingReportMapValues.immutableMap(profileMap)));
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

    public static TrainingReportQualityProfileCatalog readJson(Path file) throws IOException {
        Path resolvedFile = Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Training report quality profile catalog JSON must be an object: "
                    + resolvedFile);
        }
        try {
            return fromMap(TrainingReportMapValues.immutableMap(map));
        } catch (RuntimeException error) {
            throw new IOException("Training report quality profile catalog JSON is invalid: "
                    + resolvedFile, error);
        }
    }

    public List<String> ids() {
        return profiles.stream()
                .map(TrainingReportQualityProfile::id)
                .toList();
    }

    public Optional<TrainingReportQualityProfile> find(String id) {
        if (id == null || id.isBlank()) {
            return Optional.empty();
        }
        String normalized = TrainingReportQualityProfile.normalizeId(id);
        return profiles.stream()
                .filter(profile -> profile.id().equals(normalized))
                .findFirst();
    }

    public TrainingReportQualityProfile require(String id) {
        return find(id)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Unknown training report quality profile in catalog: " + id
                                + ". Available profiles: " + ids()));
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

    public TrainingReportQualityProfileCatalogValidator.Result validate() {
        return TrainingReportQualityProfileCatalogValidator.validate(this);
    }
}
