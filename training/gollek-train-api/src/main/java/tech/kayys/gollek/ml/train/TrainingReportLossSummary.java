package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.normalizedString;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalInt;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Type-safe view over train or validation loss history in a canonical report.
 */
public record TrainingReportLossSummary(
        boolean available,
        int count,
        OptionalDouble first,
        OptionalInt firstEpoch,
        OptionalDouble latest,
        OptionalInt latestEpoch,
        OptionalDouble best,
        OptionalInt bestEpoch,
        OptionalDouble deltaFromFirst,
        OptionalDouble relativeDeltaFromFirst,
        String trend) {
    public TrainingReportLossSummary {
        count = Math.max(0, count);
        first = first == null ? OptionalDouble.empty() : first;
        firstEpoch = firstEpoch == null ? OptionalInt.empty() : firstEpoch;
        latest = latest == null ? OptionalDouble.empty() : latest;
        latestEpoch = latestEpoch == null ? OptionalInt.empty() : latestEpoch;
        best = best == null ? OptionalDouble.empty() : best;
        bestEpoch = bestEpoch == null ? OptionalInt.empty() : bestEpoch;
        deltaFromFirst = deltaFromFirst == null ? OptionalDouble.empty() : deltaFromFirst;
        relativeDeltaFromFirst = relativeDeltaFromFirst == null ? OptionalDouble.empty() : relativeDeltaFromFirst;
        trend = normalizedString(trend, "unknown");
    }

    public static TrainingReportLossSummary empty() {
        return new TrainingReportLossSummary(
                false,
                0,
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

    public static TrainingReportLossSummary fromMap(Map<String, ?> summary) {
        if (summary == null || summary.isEmpty()) {
            return empty();
        }
        return new TrainingReportLossSummary(
                booleanValue(summary.get("available")),
                intValue(summary.get("count"), 0),
                optionalDouble(summary.get("first")),
                optionalInt(summary.get("firstEpoch")),
                optionalDouble(summary.get("latest")),
                optionalInt(summary.get("latestEpoch")),
                optionalDouble(summary.get("best")),
                optionalInt(summary.get("bestEpoch")),
                optionalDouble(summary.get("deltaFromFirst")),
                optionalDouble(summary.get("relativeDeltaFromFirst")),
                stringValue(summary.get("trend"), "unknown"));
    }

    public boolean improved() {
        return "improved".equals(trend);
    }

    public boolean worsened() {
        return "worsened".equals(trend);
    }

    public boolean flat() {
        return "flat".equals(trend);
    }
}
