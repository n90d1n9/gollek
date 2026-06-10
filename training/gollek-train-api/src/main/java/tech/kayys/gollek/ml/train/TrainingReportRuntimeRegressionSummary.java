package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Typed runtime-profile delta summary for baseline-vs-candidate trainer reports.
 */
public record TrainingReportRuntimeRegressionSummary(
        Optional<Entry> primaryGroupAverage,
        Optional<Entry> primaryHotspotAverage,
        Optional<EfficiencyEntry> accountedWallTime,
        Optional<EfficiencyEntry> wallClockOverhead,
        Optional<EfficiencyEntry> dominantBottleneck) {
    public TrainingReportRuntimeRegressionSummary {
        primaryGroupAverage = primaryGroupAverage == null ? Optional.empty() : primaryGroupAverage;
        primaryHotspotAverage = primaryHotspotAverage == null ? Optional.empty() : primaryHotspotAverage;
        accountedWallTime = accountedWallTime == null ? Optional.empty() : accountedWallTime;
        wallClockOverhead = wallClockOverhead == null ? Optional.empty() : wallClockOverhead;
        dominantBottleneck = dominantBottleneck == null ? Optional.empty() : dominantBottleneck;
    }

    public TrainingReportRuntimeRegressionSummary(
            Optional<Entry> primaryGroupAverage,
            Optional<Entry> primaryHotspotAverage) {
        this(
                primaryGroupAverage,
                primaryHotspotAverage,
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    public static TrainingReportRuntimeRegressionSummary empty() {
        return new TrainingReportRuntimeRegressionSummary(
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    public static TrainingReportRuntimeRegressionSummary fromMap(Map<String, ?> map) {
        if (map == null || map.isEmpty()) {
            return empty();
        }
        return new TrainingReportRuntimeRegressionSummary(
                Entry.fromObject(map.get("primaryGroupAverage")),
                Entry.fromObject(map.get("primaryHotspotAverage")),
                EfficiencyEntry.fromObject(map.get("accountedWallTime")),
                EfficiencyEntry.fromObject(map.get("wallClockOverhead")),
                EfficiencyEntry.fromObject(map.get("dominantBottleneck")));
    }

    public boolean available() {
        return primaryGroupAverage.isPresent()
                || primaryHotspotAverage.isPresent()
                || accountedWallTime.isPresent()
                || wallClockOverhead.isPresent()
                || dominantBottleneck.isPresent();
    }

    public boolean regressed() {
        return primaryGroupAverage.map(Entry::regressed).orElse(false)
                || primaryHotspotAverage.map(Entry::regressed).orElse(false)
                || accountedWallTime.map(EfficiencyEntry::regressed).orElse(false)
                || wallClockOverhead.map(EfficiencyEntry::regressed).orElse(false)
                || dominantBottleneck.map(EfficiencyEntry::regressed).orElse(false);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available());
        map.put("regressed", regressed());
        map.put("primaryGroupAverage", primaryGroupAverage.map(Entry::toMap).orElse(Map.of()));
        map.put("primaryHotspotAverage", primaryHotspotAverage.map(Entry::toMap).orElse(Map.of()));
        map.put("accountedWallTime", accountedWallTime.map(EfficiencyEntry::toMap).orElse(Map.of()));
        map.put("wallClockOverhead", wallClockOverhead.map(EfficiencyEntry::toMap).orElse(Map.of()));
        map.put("dominantBottleneck", dominantBottleneck.map(EfficiencyEntry::toMap).orElse(Map.of()));
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

    public record EfficiencyEntry(
            String key,
            String kind,
            String direction,
            double baselineValue,
            double candidateValue,
            double delta,
            double threshold,
            String unit) {
        public EfficiencyEntry {
            key = key == null ? "" : key.trim();
            kind = kind == null ? "" : kind.trim();
            direction = direction == null ? "" : direction.trim();
            baselineValue = finiteOrZero(baselineValue);
            candidateValue = finiteOrZero(candidateValue);
            delta = finiteOrZero(delta);
            threshold = Math.max(0.0, finiteOrZero(threshold));
            unit = unit == null || unit.isBlank() ? "value" : unit.trim();
        }

        static Optional<EfficiencyEntry> fromObject(Object value) {
            if (!(value instanceof Map<?, ?> map) || map.isEmpty()) {
                return Optional.empty();
            }
            return Optional.of(new EfficiencyEntry(
                    TrainingReportValues.stringValue(map.get("key"), ""),
                    TrainingReportValues.stringValue(map.get("kind"), ""),
                    TrainingReportValues.stringValue(map.get("direction"), ""),
                    TrainingReportValues.optionalDouble(map.get("baselineValue")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("candidateValue")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("delta")).orElse(0.0),
                    TrainingReportValues.optionalDouble(map.get("threshold")).orElse(0.0),
                    TrainingReportValues.stringValue(map.get("unit"), "value")));
        }

        public boolean regressed() {
            return threshold > 0.0
                    && ("higher_is_worse".equals(direction) ? delta >= threshold : -delta >= threshold);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("key", key);
            map.put("kind", kind);
            map.put("direction", direction);
            map.put("baselineValue", baselineValue);
            map.put("candidateValue", candidateValue);
            map.put("delta", delta);
            map.put("threshold", threshold);
            map.put("unit", unit);
            map.put("regressed", regressed());
            return Map.copyOf(map);
        }

        private static double finiteOrZero(double value) {
            return Double.isFinite(value) ? value : 0.0;
        }
    }
}
