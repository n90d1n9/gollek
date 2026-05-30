package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.mapValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.normalizedString;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalInt;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Type-safe view over a named train or validation metric trend in a canonical report.
 */
public record TrainingReportMetricSummary(
        String name,
        boolean available,
        int count,
        OptionalDouble first,
        OptionalInt firstEpoch,
        OptionalDouble latest,
        OptionalInt latestEpoch,
        OptionalDouble min,
        OptionalInt minEpoch,
        OptionalDouble max,
        OptionalInt maxEpoch,
        OptionalDouble deltaFromFirst,
        OptionalDouble relativeDeltaFromFirst,
        String trend) {
    public TrainingReportMetricSummary {
        name = normalizedString(name, "");
        count = Math.max(0, count);
        first = first == null ? OptionalDouble.empty() : first;
        firstEpoch = firstEpoch == null ? OptionalInt.empty() : firstEpoch;
        latest = latest == null ? OptionalDouble.empty() : latest;
        latestEpoch = latestEpoch == null ? OptionalInt.empty() : latestEpoch;
        min = min == null ? OptionalDouble.empty() : min;
        minEpoch = minEpoch == null ? OptionalInt.empty() : minEpoch;
        max = max == null ? OptionalDouble.empty() : max;
        maxEpoch = maxEpoch == null ? OptionalInt.empty() : maxEpoch;
        deltaFromFirst = deltaFromFirst == null ? OptionalDouble.empty() : deltaFromFirst;
        relativeDeltaFromFirst = relativeDeltaFromFirst == null ? OptionalDouble.empty() : relativeDeltaFromFirst;
        trend = normalizedString(trend, "unknown");
    }

    public static TrainingReportMetricSummary empty(String name) {
        return new TrainingReportMetricSummary(
                name,
                false,
                0,
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                "unknown");
    }

    public static TrainingReportMetricSummary fromMap(String name, Map<String, ?> summary) {
        if (summary == null || summary.isEmpty()) {
            return empty(name);
        }
        return new TrainingReportMetricSummary(
                name,
                booleanValue(summary.get("available")),
                intValue(summary.get("count"), 0),
                optionalDouble(summary.get("first")),
                optionalInt(summary.get("firstEpoch")),
                optionalDouble(summary.get("latest")),
                optionalInt(summary.get("latestEpoch")),
                optionalDouble(summary.get("min")),
                optionalInt(summary.get("minEpoch")),
                optionalDouble(summary.get("max")),
                optionalInt(summary.get("maxEpoch")),
                optionalDouble(summary.get("deltaFromFirst")),
                optionalDouble(summary.get("relativeDeltaFromFirst")),
                stringValue(summary.get("trend"), "unknown"));
    }

    public static Map<String, TrainingReportMetricSummary> fromMaps(Map<String, ?> summaries) {
        if (summaries == null || summaries.isEmpty()) {
            return Map.of();
        }
        Map<String, TrainingReportMetricSummary> typed = new LinkedHashMap<>();
        for (String name : summaries.keySet().stream().map(String::valueOf).sorted().toList()) {
            typed.put(name, fromMap(name, mapValue(summaries, name)));
        }
        return Map.copyOf(typed);
    }

    public boolean increased() {
        return "increased".equals(trend);
    }

    public boolean decreased() {
        return "decreased".equals(trend);
    }

    public boolean flat() {
        return "flat".equals(trend);
    }

    public boolean latestIsMax() {
        return available
                && latest.isPresent()
                && max.isPresent()
                && Double.compare(latest.getAsDouble(), max.getAsDouble()) == 0;
    }

    public boolean latestIsMin() {
        return available
                && latest.isPresent()
                && min.isPresent()
                && Double.compare(latest.getAsDouble(), min.getAsDouble()) == 0;
    }

    public boolean hasName(String metricName) {
        return Objects.equals(name, normalizedString(metricName, ""));
    }
}
