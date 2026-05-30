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
 * Type-safe view over train/validation generalization gap history.
 */
public record TrainingReportGeneralizationSummary(
        boolean available,
        int count,
        OptionalDouble firstGap,
        OptionalInt firstEpoch,
        OptionalDouble latestGap,
        OptionalInt latestEpoch,
        OptionalDouble maxGap,
        OptionalInt maxGapEpoch,
        OptionalDouble gapDeltaFromFirst,
        String gapTrend,
        boolean gapIncreasing,
        OptionalDouble latestValidationToTrainLossRatio,
        boolean latestValidationLossAboveTrainLoss) {
    public TrainingReportGeneralizationSummary {
        count = Math.max(0, count);
        firstGap = firstGap == null ? OptionalDouble.empty() : firstGap;
        firstEpoch = firstEpoch == null ? OptionalInt.empty() : firstEpoch;
        latestGap = latestGap == null ? OptionalDouble.empty() : latestGap;
        latestEpoch = latestEpoch == null ? OptionalInt.empty() : latestEpoch;
        maxGap = maxGap == null ? OptionalDouble.empty() : maxGap;
        maxGapEpoch = maxGapEpoch == null ? OptionalInt.empty() : maxGapEpoch;
        gapDeltaFromFirst = gapDeltaFromFirst == null ? OptionalDouble.empty() : gapDeltaFromFirst;
        gapTrend = normalizedString(gapTrend, "unknown");
        latestValidationToTrainLossRatio = latestValidationToTrainLossRatio == null
                ? OptionalDouble.empty()
                : latestValidationToTrainLossRatio;
    }

    public static TrainingReportGeneralizationSummary empty() {
        return new TrainingReportGeneralizationSummary(
                false,
                0,
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalInt.empty(),
                OptionalDouble.empty(),
                "unknown",
                false,
                OptionalDouble.empty(),
                false);
    }

    public static TrainingReportGeneralizationSummary fromMap(Map<String, ?> summary) {
        if (summary == null || summary.isEmpty()) {
            return empty();
        }
        return new TrainingReportGeneralizationSummary(
                booleanValue(summary.get("available")),
                intValue(summary.get("count"), 0),
                optionalDouble(summary.get("firstGap")),
                optionalInt(summary.get("firstEpoch")),
                optionalDouble(summary.get("latestGap")),
                optionalInt(summary.get("latestEpoch")),
                optionalDouble(summary.get("maxGap")),
                optionalInt(summary.get("maxGapEpoch")),
                optionalDouble(summary.get("gapDeltaFromFirst")),
                stringValue(summary.get("gapTrend"), "unknown"),
                booleanValue(summary.get("gapIncreasing")),
                optionalDouble(summary.get("latestValidationToTrainLossRatio")),
                booleanValue(summary.get("latestValidationLossAboveTrainLoss")));
    }

    public boolean widening() {
        return "increasing".equals(gapTrend) || gapIncreasing;
    }

    public boolean narrowing() {
        return "decreasing".equals(gapTrend);
    }

    public boolean flat() {
        return "flat".equals(gapTrend);
    }
}
