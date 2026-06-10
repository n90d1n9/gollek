package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Typed runtime-profile delta summary for baseline-vs-candidate trainer reports.
 */
public record TrainingReportRuntimeRegressionSummary(
        Optional<Entry> primaryGroupAverage,
        Optional<Entry> primaryHotspotAverage) {
    public TrainingReportRuntimeRegressionSummary {
        primaryGroupAverage = primaryGroupAverage == null ? Optional.empty() : primaryGroupAverage;
        primaryHotspotAverage = primaryHotspotAverage == null ? Optional.empty() : primaryHotspotAverage;
    }

    public static TrainingReportRuntimeRegressionSummary empty() {
        return new TrainingReportRuntimeRegressionSummary(Optional.empty(), Optional.empty());
    }

    public static TrainingReportRuntimeRegressionSummary fromMap(Map<String, ?> map) {
        if (map == null || map.isEmpty()) {
            return empty();
        }
        return new TrainingReportRuntimeRegressionSummary(
                Entry.fromObject(map.get("primaryGroupAverage")),
                Entry.fromObject(map.get("primaryHotspotAverage")));
    }

    public boolean available() {
        return primaryGroupAverage.isPresent() || primaryHotspotAverage.isPresent();
    }

    public boolean regressed() {
        return primaryGroupAverage.map(Entry::regressed).orElse(false)
                || primaryHotspotAverage.map(Entry::regressed).orElse(false);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available());
        map.put("regressed", regressed());
        map.put("primaryGroupAverage", primaryGroupAverage.map(Entry::toMap).orElse(Map.of()));
        map.put("primaryHotspotAverage", primaryHotspotAverage.map(Entry::toMap).orElse(Map.of()));
        return Map.copyOf(map);
    }

    public record Entry(
            String key,
            String kind,
            double baselineAverageMillis,
            double candidateAverageMillis,
            double ratio,
            double threshold) {
        public Entry {
            key = key == null ? "" : key.trim();
            kind = kind == null ? "" : kind.trim();
            baselineAverageMillis = Math.max(0.0, baselineAverageMillis);
            candidateAverageMillis = Math.max(0.0, candidateAverageMillis);
            ratio = Math.max(0.0, ratio);
            threshold = Math.max(0.0, threshold);
        }

        static Optional<Entry> fromObject(Object value) {
            if (!(value instanceof Map<?, ?> map) || map.isEmpty()) {
                return Optional.empty();
            }
            return Optional.of(new Entry(
                    TrainingReportValues.stringValue(map.get("key"), ""),
                    TrainingReportValues.stringValue(map.get("kind"), ""),
                    TrainingReportValues.optionalDouble(map.get("baselineAverageMillis")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("candidateAverageMillis")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("ratio")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("threshold")).orElse(0.0)));
        }

        public boolean regressed() {
            return ratio >= threshold && threshold > 0.0;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("key", key);
            map.put("kind", kind);
            map.put("baselineAverageMillis", baselineAverageMillis);
            map.put("candidateAverageMillis", candidateAverageMillis);
            map.put("ratio", ratio);
            map.put("threshold", threshold);
            map.put("regressed", regressed());
            return Map.copyOf(map);
        }
    }
}
