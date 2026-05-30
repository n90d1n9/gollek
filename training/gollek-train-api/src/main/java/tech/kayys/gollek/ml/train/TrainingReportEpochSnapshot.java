package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.mapValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalInt;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Type-safe view over one canonical training-report history row.
 */
public record TrainingReportEpochSnapshot(
        OptionalInt epoch,
        OptionalDouble trainLoss,
        OptionalDouble validationLoss,
        OptionalDouble learningRate,
        OptionalLong optimizerStepCount,
        OptionalLong schedulerStepCount,
        Map<String, Double> trainMetrics,
        Map<String, Object> trainMetricDetails,
        Map<String, Double> validationMetrics,
        Map<String, Object> validationMetricDetails,
        Optimization optimization,
        MixedPrecision mixedPrecision,
        Map<String, Object> raw) {
    public TrainingReportEpochSnapshot {
        epoch = epoch == null ? OptionalInt.empty() : epoch;
        trainLoss = trainLoss == null ? OptionalDouble.empty() : trainLoss;
        validationLoss = validationLoss == null ? OptionalDouble.empty() : validationLoss;
        learningRate = learningRate == null ? OptionalDouble.empty() : learningRate;
        optimizerStepCount = optimizerStepCount == null ? OptionalLong.empty() : optimizerStepCount;
        schedulerStepCount = schedulerStepCount == null ? OptionalLong.empty() : schedulerStepCount;
        trainMetrics = trainMetrics == null ? Map.of() : Map.copyOf(trainMetrics);
        trainMetricDetails = trainMetricDetails == null ? Map.of() : Map.copyOf(trainMetricDetails);
        validationMetrics = validationMetrics == null ? Map.of() : Map.copyOf(validationMetrics);
        validationMetricDetails = validationMetricDetails == null ? Map.of() : Map.copyOf(validationMetricDetails);
        optimization = optimization == null ? Optimization.empty() : optimization;
        mixedPrecision = mixedPrecision == null ? MixedPrecision.empty() : mixedPrecision;
        raw = raw == null ? Map.of() : Map.copyOf(raw);
    }

    public static TrainingReportEpochSnapshot fromMap(Map<String, ?> row) {
        if (row == null || row.isEmpty()) {
            return empty();
        }
        return new TrainingReportEpochSnapshot(
                optionalInt(row.get("epoch")),
                optionalDouble(row.get("trainLoss")),
                optionalDouble(row.get("validationLoss")),
                optionalDouble(row.get("learningRate")),
                optionalLong(row.get("optimizerStepCount")),
                optionalLong(row.get("schedulerStepCount")),
                doubleMap(mapValue(row, "trainMetrics")),
                mapValue(row, "trainMetricDetails"),
                doubleMap(mapValue(row, "validationMetrics")),
                mapValue(row, "validationMetricDetails"),
                Optimization.fromMap(row),
                MixedPrecision.fromMap(row),
                immutableMap(row));
    }

    public static TrainingReportEpochSnapshot empty() {
        return new TrainingReportEpochSnapshot(
                OptionalInt.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                Optimization.empty(),
                MixedPrecision.empty(),
                Map.of());
    }

    public static List<TrainingReportEpochSnapshot> fromMaps(List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) {
            return List.of();
        }
        return rows.stream()
                .map(TrainingReportEpochSnapshot::fromMap)
                .toList();
    }

    public boolean hasTraining() {
        return trainLoss.isPresent() || !trainMetrics.isEmpty();
    }

    public boolean hasValidation() {
        return validationLoss.isPresent() || !validationMetrics.isEmpty();
    }

    public OptionalDouble trainMetric(String metricName) {
        return metric(trainMetrics, metricName);
    }

    public OptionalDouble validationMetric(String metricName) {
        return metric(validationMetrics, metricName);
    }

    public OptionalDouble generalizationGap() {
        if (trainLoss.isEmpty() || validationLoss.isEmpty()) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(validationLoss.getAsDouble() - trainLoss.getAsDouble());
    }

    public OptionalDouble validationToTrainLossRatio() {
        if (trainLoss.isEmpty() || validationLoss.isEmpty()) {
            return OptionalDouble.empty();
        }
        double train = trainLoss.getAsDouble();
        if (train == 0.0d) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(validationLoss.getAsDouble() / train);
    }

    private static OptionalDouble metric(Map<String, Double> metrics, String metricName) {
        if (metricName == null || metricName.isBlank()) {
            return OptionalDouble.empty();
        }
        Double value = metrics.get(metricName.trim());
        return value == null ? OptionalDouble.empty() : OptionalDouble.of(value.doubleValue());
    }

    private static Map<String, Double> doubleMap(Map<String, ?> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        Map<String, Double> values = new LinkedHashMap<>();
        for (String name : source.keySet().stream().map(String::valueOf).sorted().toList()) {
            optionalDouble(source.get(name)).ifPresent(value -> values.put(name, value));
        }
        return Map.copyOf(values);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    public record Optimization(
            OptionalDouble gradientL2NormBeforeClip,
            OptionalDouble gradientL2Norm,
            OptionalDouble gradientZeroFraction,
            OptionalLong gradientNonFiniteCount,
            OptionalDouble gradientClipScale,
            boolean gradientClipped,
            OptionalDouble parameterL2Norm,
            OptionalDouble parameterZeroFraction,
            OptionalLong parameterNonFiniteCount,
            boolean parameterUpdateDiagnosticsEnabled,
            OptionalDouble parameterUpdateL2Norm,
            OptionalDouble parameterUpdateZeroFraction,
            OptionalDouble parameterUpdateToParameterL2Ratio,
            OptionalLong parameterUpdateNonFiniteCount) {
        public Optimization {
            gradientL2NormBeforeClip = gradientL2NormBeforeClip == null
                    ? OptionalDouble.empty()
                    : gradientL2NormBeforeClip;
            gradientL2Norm = gradientL2Norm == null ? OptionalDouble.empty() : gradientL2Norm;
            gradientZeroFraction = gradientZeroFraction == null ? OptionalDouble.empty() : gradientZeroFraction;
            gradientNonFiniteCount = gradientNonFiniteCount == null ? OptionalLong.empty() : gradientNonFiniteCount;
            gradientClipScale = gradientClipScale == null ? OptionalDouble.empty() : gradientClipScale;
            parameterL2Norm = parameterL2Norm == null ? OptionalDouble.empty() : parameterL2Norm;
            parameterZeroFraction = parameterZeroFraction == null ? OptionalDouble.empty() : parameterZeroFraction;
            parameterNonFiniteCount = parameterNonFiniteCount == null ? OptionalLong.empty() : parameterNonFiniteCount;
            parameterUpdateL2Norm = parameterUpdateL2Norm == null ? OptionalDouble.empty() : parameterUpdateL2Norm;
            parameterUpdateZeroFraction = parameterUpdateZeroFraction == null
                    ? OptionalDouble.empty()
                    : parameterUpdateZeroFraction;
            parameterUpdateToParameterL2Ratio = parameterUpdateToParameterL2Ratio == null
                    ? OptionalDouble.empty()
                    : parameterUpdateToParameterL2Ratio;
            parameterUpdateNonFiniteCount = parameterUpdateNonFiniteCount == null
                    ? OptionalLong.empty()
                    : parameterUpdateNonFiniteCount;
        }

        public static Optimization fromMap(Map<String, ?> row) {
            if (row == null || row.isEmpty()) {
                return empty();
            }
            return new Optimization(
                    optionalDouble(row.get("gradientL2NormBeforeClip")),
                    optionalDouble(row.get("gradientL2Norm")),
                    optionalDouble(row.get("gradientZeroFraction")),
                    optionalLong(row.get("gradientNonFiniteCount")),
                    optionalDouble(row.get("gradientClipScale")),
                    booleanValue(row.get("gradientClipped")),
                    optionalDouble(row.get("parameterL2Norm")),
                    optionalDouble(row.get("parameterZeroFraction")),
                    optionalLong(row.get("parameterNonFiniteCount")),
                    booleanValue(row.get("parameterUpdateDiagnosticsEnabled")),
                    optionalDouble(row.get("parameterUpdateL2Norm")),
                    optionalDouble(row.get("parameterUpdateZeroFraction")),
                    optionalDouble(row.get("parameterUpdateToParameterL2Ratio")),
                    optionalLong(row.get("parameterUpdateNonFiniteCount")));
        }

        public static Optimization empty() {
            return new Optimization(
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty(),
                    OptionalDouble.empty(),
                    false,
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty(),
                    false,
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalLong.empty());
        }

        public boolean hasGradientDiagnostics() {
            return gradientL2Norm.isPresent()
                    || gradientZeroFraction.isPresent()
                    || gradientNonFiniteCount.isPresent();
        }

        public boolean hasParameterDiagnostics() {
            return parameterL2Norm.isPresent()
                    || parameterZeroFraction.isPresent()
                    || parameterNonFiniteCount.isPresent();
        }

        public boolean hasParameterUpdateDiagnostics() {
            return parameterUpdateDiagnosticsEnabled
                    || parameterUpdateL2Norm.isPresent()
                    || parameterUpdateToParameterL2Ratio.isPresent()
                    || parameterUpdateNonFiniteCount.isPresent();
        }
    }

    public record MixedPrecision(
            boolean enabled,
            OptionalDouble lossScale,
            boolean overflowDetected,
            int overflowSkipCount) {
        public MixedPrecision {
            lossScale = lossScale == null ? OptionalDouble.empty() : lossScale;
            overflowSkipCount = Math.max(0, overflowSkipCount);
        }

        public static MixedPrecision fromMap(Map<String, ?> row) {
            if (row == null || row.isEmpty()) {
                return empty();
            }
            return new MixedPrecision(
                    booleanValue(row.get("mixedPrecisionEnabled")),
                    optionalDouble(row.get("mixedPrecisionLossScale")),
                    booleanValue(row.get("mixedPrecisionOverflowDetected")),
                    intValue(row.get("mixedPrecisionOverflowSkipCount"), 0));
        }

        public static MixedPrecision empty() {
            return new MixedPrecision(false, OptionalDouble.empty(), false, 0);
        }
    }
}
