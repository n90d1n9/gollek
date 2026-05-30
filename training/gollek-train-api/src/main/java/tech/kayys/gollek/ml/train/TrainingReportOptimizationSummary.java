package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.mapValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.normalizedString;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalInt;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Type-safe view over optimizer telemetry persisted in a canonical training report.
 */
public record TrainingReportOptimizationSummary(
        boolean available,
        int count,
        OptionalInt firstEpoch,
        OptionalInt lastEpoch,
        Component gradients,
        Component parameters,
        ParameterUpdates parameterUpdates,
        Latest latest) {
    public TrainingReportOptimizationSummary {
        count = Math.max(0, count);
        firstEpoch = firstEpoch == null ? OptionalInt.empty() : firstEpoch;
        lastEpoch = lastEpoch == null ? OptionalInt.empty() : lastEpoch;
        gradients = gradients == null ? Component.empty() : gradients;
        parameters = parameters == null ? Component.empty() : parameters;
        parameterUpdates = parameterUpdates == null ? ParameterUpdates.empty() : parameterUpdates;
        latest = latest == null ? Latest.empty() : latest;
    }

    public static TrainingReportOptimizationSummary empty() {
        return new TrainingReportOptimizationSummary(
                false,
                0,
                OptionalInt.empty(),
                OptionalInt.empty(),
                Component.empty(),
                Component.empty(),
                ParameterUpdates.empty(),
                Latest.empty());
    }

    public static TrainingReportOptimizationSummary fromMap(Map<String, ?> summary) {
        if (summary == null || summary.isEmpty()) {
            return empty();
        }
        return new TrainingReportOptimizationSummary(
                booleanValue(summary.get("available")),
                intValue(summary.get("count"), 0),
                optionalInt(summary.get("firstEpoch")),
                optionalInt(summary.get("lastEpoch")),
                Component.fromMap(mapValue(summary, "gradients")),
                Component.fromMap(mapValue(summary, "parameters")),
                ParameterUpdates.fromMap(mapValue(summary, "parameterUpdates")),
                Latest.fromMap(mapValue(summary, "latest")));
    }

    public record Component(
            Series l2Norm,
            Series zeroFraction,
            Series nonFiniteFraction,
            CountSummary nonFiniteCount) {
        public Component {
            l2Norm = l2Norm == null ? Series.empty() : l2Norm;
            zeroFraction = zeroFraction == null ? Series.empty() : zeroFraction;
            nonFiniteFraction = nonFiniteFraction == null ? Series.empty() : nonFiniteFraction;
            nonFiniteCount = nonFiniteCount == null ? CountSummary.empty() : nonFiniteCount;
        }

        public static Component empty() {
            return new Component(Series.empty(), Series.empty(), Series.empty(), CountSummary.empty());
        }

        public static Component fromMap(Map<String, ?> map) {
            if (map == null || map.isEmpty()) {
                return empty();
            }
            return new Component(
                    Series.fromMap(mapValue(map, "l2Norm")),
                    Series.fromMap(mapValue(map, "zeroFraction")),
                    Series.fromMap(mapValue(map, "nonFiniteFraction")),
                    CountSummary.fromMap(mapValue(map, "nonFiniteCount")));
        }
    }

    public record ParameterUpdates(
            boolean enabled,
            Series l2Norm,
            Series zeroFraction,
            Series nonFiniteFraction,
            CountSummary nonFiniteCount,
            Series toParameterL2Ratio,
            Series toGradientL2Ratio) {
        public ParameterUpdates {
            l2Norm = l2Norm == null ? Series.empty() : l2Norm;
            zeroFraction = zeroFraction == null ? Series.empty() : zeroFraction;
            nonFiniteFraction = nonFiniteFraction == null ? Series.empty() : nonFiniteFraction;
            nonFiniteCount = nonFiniteCount == null ? CountSummary.empty() : nonFiniteCount;
            toParameterL2Ratio = toParameterL2Ratio == null ? Series.empty() : toParameterL2Ratio;
            toGradientL2Ratio = toGradientL2Ratio == null ? Series.empty() : toGradientL2Ratio;
        }

        public static ParameterUpdates empty() {
            return new ParameterUpdates(
                    false,
                    Series.empty(),
                    Series.empty(),
                    Series.empty(),
                    CountSummary.empty(),
                    Series.empty(),
                    Series.empty());
        }

        public static ParameterUpdates fromMap(Map<String, ?> map) {
            if (map == null || map.isEmpty()) {
                return empty();
            }
            return new ParameterUpdates(
                    booleanValue(map.get("enabled")),
                    Series.fromMap(mapValue(map, "l2Norm")),
                    Series.fromMap(mapValue(map, "zeroFraction")),
                    Series.fromMap(mapValue(map, "nonFiniteFraction")),
                    CountSummary.fromMap(mapValue(map, "nonFiniteCount")),
                    Series.fromMap(mapValue(map, "toParameterL2Ratio")),
                    Series.fromMap(mapValue(map, "toGradientL2Ratio")));
        }
    }

    public record Series(
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
            String trend) {
        public Series {
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
            trend = normalizedString(trend, "unknown");
        }

        public static Series empty() {
            return new Series(
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
                    "unknown");
        }

        public static Series fromMap(Map<String, ?> map) {
            if (map == null || map.isEmpty()) {
                return empty();
            }
            return new Series(
                    booleanValue(map.get("available")),
                    intValue(map.get("count"), 0),
                    optionalDouble(map.get("first")),
                    optionalInt(map.get("firstEpoch")),
                    optionalDouble(map.get("latest")),
                    optionalInt(map.get("latestEpoch")),
                    optionalDouble(map.get("min")),
                    optionalInt(map.get("minEpoch")),
                    optionalDouble(map.get("max")),
                    optionalInt(map.get("maxEpoch")),
                    optionalDouble(map.get("deltaFromFirst")),
                    stringValue(map.get("trend"), "unknown"));
        }
    }

    public record CountSummary(
            boolean available,
            int count,
            long total,
            long max,
            OptionalInt maxEpoch) {
        public CountSummary {
            count = Math.max(0, count);
            total = Math.max(0L, total);
            max = Math.max(0L, max);
            maxEpoch = maxEpoch == null ? OptionalInt.empty() : maxEpoch;
        }

        public static CountSummary empty() {
            return new CountSummary(false, 0, 0L, 0L, OptionalInt.empty());
        }

        public static CountSummary fromMap(Map<String, ?> map) {
            if (map == null || map.isEmpty()) {
                return empty();
            }
            return new CountSummary(
                    booleanValue(map.get("available")),
                    intValue(map.get("count"), 0),
                    longValue(map.get("total"), 0L),
                    longValue(map.get("max"), 0L),
                    optionalInt(map.get("maxEpoch")));
        }
    }

    public record Latest(
            boolean available,
            OptionalInt epoch,
            OptionalDouble gradientL2Norm,
            OptionalDouble gradientZeroFraction,
            OptionalLong gradientNonFiniteCount,
            OptionalDouble parameterL2Norm,
            OptionalLong parameterNonFiniteCount,
            boolean parameterUpdateDiagnosticsEnabled,
            OptionalDouble parameterUpdateL2Norm,
            OptionalDouble parameterUpdateToParameterL2Ratio,
            OptionalLong parameterUpdateNonFiniteCount) {
        public Latest {
            epoch = epoch == null ? OptionalInt.empty() : epoch;
            gradientL2Norm = gradientL2Norm == null ? OptionalDouble.empty() : gradientL2Norm;
            gradientZeroFraction = gradientZeroFraction == null ? OptionalDouble.empty() : gradientZeroFraction;
            gradientNonFiniteCount = gradientNonFiniteCount == null ? OptionalLong.empty() : gradientNonFiniteCount;
            parameterL2Norm = parameterL2Norm == null ? OptionalDouble.empty() : parameterL2Norm;
            parameterNonFiniteCount = parameterNonFiniteCount == null ? OptionalLong.empty() : parameterNonFiniteCount;
            parameterUpdateL2Norm = parameterUpdateL2Norm == null ? OptionalDouble.empty() : parameterUpdateL2Norm;
            parameterUpdateToParameterL2Ratio = parameterUpdateToParameterL2Ratio == null
                    ? OptionalDouble.empty()
                    : parameterUpdateToParameterL2Ratio;
            parameterUpdateNonFiniteCount = parameterUpdateNonFiniteCount == null
                    ? OptionalLong.empty()
                    : parameterUpdateNonFiniteCount;
        }

        public static Latest empty() {
            return new Latest(
                    false,
                    OptionalInt.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty(),
                    false,
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty());
        }

        public static Latest fromMap(Map<String, ?> map) {
            if (map == null || map.isEmpty()) {
                return empty();
            }
            return new Latest(
                    booleanValue(map.get("available")),
                    optionalInt(map.get("epoch")),
                    optionalDouble(map.get("gradientL2Norm")),
                    optionalDouble(map.get("gradientZeroFraction")),
                    optionalLong(map.get("gradientNonFiniteCount")),
                    optionalDouble(map.get("parameterL2Norm")),
                    optionalLong(map.get("parameterNonFiniteCount")),
                    booleanValue(map.get("parameterUpdateDiagnosticsEnabled")),
                    optionalDouble(map.get("parameterUpdateL2Norm")),
                    optionalDouble(map.get("parameterUpdateToParameterL2Ratio")),
                    optionalLong(map.get("parameterUpdateNonFiniteCount")));
        }
    }
}
