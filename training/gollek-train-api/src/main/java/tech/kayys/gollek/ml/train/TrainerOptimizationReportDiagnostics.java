package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Report-level optimizer health checks derived from epoch telemetry.
 */
final class TrainerOptimizationReportDiagnostics {
    private TrainerOptimizationReportDiagnostics() {
    }

    static void addFindings(
            List<TrainingReportDiagnostics.Finding> findings,
            List<Map<String, Object>> history,
            TrainingReportDiagnostics.Options options) {
        if (findings == null || history == null || history.isEmpty()) {
            return;
        }
        addNonFiniteFinding(findings, history);
        LatestOptimization latest = latestOptimization(history);
        if (latest == null) {
            return;
        }
        addZeroGradientFinding(findings, latest, options);
        addMissingUpdateFinding(findings, latest, options);
        addOversizedUpdateFinding(findings, latest, options);
    }

    private static void addNonFiniteFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            List<Map<String, Object>> history) {
        NonFiniteAggregate aggregate = nonFiniteAggregate(history);
        if (aggregate.totalNonFiniteCount() <= 0L) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("epochs", aggregate.epochs());
        evidence.put("gradientNonFiniteCount", aggregate.gradient().nonFiniteCount());
        evidence.put("gradientNanCount", aggregate.gradient().nanCount());
        evidence.put("gradientPositiveInfinityCount", aggregate.gradient().positiveInfinityCount());
        evidence.put("gradientNegativeInfinityCount", aggregate.gradient().negativeInfinityCount());
        evidence.put("gradientMaxNonFiniteFraction", aggregate.gradient().maxNonFiniteFraction());
        evidence.put("parameterNonFiniteCount", aggregate.parameters().nonFiniteCount());
        evidence.put("parameterNanCount", aggregate.parameters().nanCount());
        evidence.put("parameterPositiveInfinityCount", aggregate.parameters().positiveInfinityCount());
        evidence.put("parameterNegativeInfinityCount", aggregate.parameters().negativeInfinityCount());
        evidence.put("parameterMaxNonFiniteFraction", aggregate.parameters().maxNonFiniteFraction());
        evidence.put("parameterUpdateNonFiniteCount", aggregate.updates().nonFiniteCount());
        evidence.put("parameterUpdateNanCount", aggregate.updates().nanCount());
        evidence.put("parameterUpdatePositiveInfinityCount", aggregate.updates().positiveInfinityCount());
        evidence.put("parameterUpdateNegativeInfinityCount", aggregate.updates().negativeInfinityCount());
        evidence.put("parameterUpdateMaxNonFiniteFraction", aggregate.updates().maxNonFiniteFraction());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.CRITICAL,
                "optimization.non_finite_values",
                "Gradient, parameter, or parameter-update telemetry contains non-finite values.",
                evidence));
    }

    private static void addZeroGradientFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            LatestOptimization latest,
            TrainingReportDiagnostics.Options options) {
        if (latest.gradientValueCount() <= 0L
                || !Double.isFinite(latest.gradientZeroFraction())
                || !Double.isFinite(latest.gradientL2Norm())
                || latest.gradientZeroFraction() < options.optimizationZeroGradientFractionThreshold()
                || latest.gradientL2Norm() > options.optimizationTinyNormThreshold()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("epoch", latest.epoch());
        evidence.put("gradientValueCount", latest.gradientValueCount());
        evidence.put("gradientZeroFraction", latest.gradientZeroFraction());
        evidence.put("gradientL2Norm", latest.gradientL2Norm());
        evidence.put("zeroGradientFractionThreshold", options.optimizationZeroGradientFractionThreshold());
        evidence.put("tinyNormThreshold", options.optimizationTinyNormThreshold());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "optimization.zero_gradient",
                "Latest optimizer telemetry shows an all-zero gradient, so the model may not be learning.",
                evidence));
    }

    private static void addMissingUpdateFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            LatestOptimization latest,
            TrainingReportDiagnostics.Options options) {
        if (!latest.parameterUpdateDiagnosticsEnabled()
                || !Double.isFinite(latest.gradientL2Norm())
                || !Double.isFinite(latest.parameterUpdateL2Norm())
                || latest.gradientL2Norm() <= options.optimizationTinyNormThreshold()
                || latest.parameterUpdateL2Norm() > options.optimizationTinyNormThreshold()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("epoch", latest.epoch());
        evidence.put("gradientL2Norm", latest.gradientL2Norm());
        evidence.put("parameterUpdateL2Norm", latest.parameterUpdateL2Norm());
        evidence.put("tinyNormThreshold", options.optimizationTinyNormThreshold());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "optimization.no_parameter_update",
                "Gradient telemetry is non-zero, but parameter-update telemetry is effectively zero.",
                evidence));
    }

    private static void addOversizedUpdateFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            LatestOptimization latest,
            TrainingReportDiagnostics.Options options) {
        if (!latest.parameterUpdateDiagnosticsEnabled()
                || !Double.isFinite(latest.parameterUpdateToParameterL2Ratio())
                || latest.parameterUpdateToParameterL2Ratio() <= options.optimizationMaxUpdateToParameterRatio()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("epoch", latest.epoch());
        evidence.put("parameterUpdateToParameterL2Ratio", latest.parameterUpdateToParameterL2Ratio());
        evidence.put("maxUpdateToParameterRatio", options.optimizationMaxUpdateToParameterRatio());
        evidence.put("parameterUpdateL2Norm", latest.parameterUpdateL2Norm());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "optimization.update_too_large",
                "Latest parameter update is larger than the configured safe parameter-norm ratio.",
                evidence));
    }

    private static LatestOptimization latestOptimization(List<Map<String, Object>> history) {
        LatestOptimization latest = null;
        int latestEpoch = Integer.MIN_VALUE;
        for (Map<String, Object> row : history) {
            if (!hasOptimizationTelemetry(row)) {
                continue;
            }
            int epoch = intValue(row.get("epoch"), Integer.MIN_VALUE);
            if (latest == null || epoch >= latestEpoch) {
                latestEpoch = epoch;
                latest = new LatestOptimization(
                        epoch,
                        longValue(row.get("gradientValueCount")),
                        doubleValue(row.get("gradientL2Norm")),
                        doubleValue(row.get("gradientZeroFraction")),
                        booleanValue(row.get("parameterUpdateDiagnosticsEnabled")),
                        doubleValue(row.get("parameterUpdateL2Norm")),
                        doubleValue(row.get("parameterUpdateToParameterL2Ratio")));
            }
        }
        return latest;
    }

    private static boolean hasOptimizationTelemetry(Map<String, Object> row) {
        return row.containsKey("gradientValueCount")
                || row.containsKey("gradientL2Norm")
                || row.containsKey("parameterUpdateDiagnosticsEnabled")
                || row.containsKey("parameterUpdateL2Norm")
                || row.containsKey("parameterUpdateToParameterL2Ratio");
    }

    private static NonFiniteAggregate nonFiniteAggregate(List<Map<String, Object>> history) {
        NonFiniteCounts gradients = NonFiniteCounts.empty();
        NonFiniteCounts parameters = NonFiniteCounts.empty();
        NonFiniteCounts updates = NonFiniteCounts.empty();
        List<Integer> epochs = new ArrayList<>();
        for (Map<String, Object> row : history) {
            NonFiniteCounts rowGradients = nonFiniteCounts(row, "gradient");
            NonFiniteCounts rowParameters = nonFiniteCounts(row, "parameter");
            NonFiniteCounts rowUpdates = nonFiniteCounts(row, "parameterUpdate");
            long rowTotal = rowGradients.nonFiniteCount()
                    + rowParameters.nonFiniteCount()
                    + rowUpdates.nonFiniteCount();
            if (rowTotal > 0L) {
                epochs.add(intValue(row.get("epoch"), Integer.MAX_VALUE));
            }
            gradients = gradients.plus(rowGradients);
            parameters = parameters.plus(rowParameters);
            updates = updates.plus(rowUpdates);
        }
        return new NonFiniteAggregate(
                gradients,
                parameters,
                updates,
                List.copyOf(epochs));
    }

    private static NonFiniteCounts nonFiniteCounts(Map<String, Object> row, String prefix) {
        return new NonFiniteCounts(
                longValue(row.get(prefix + "NonFiniteCount")),
                longValue(row.get(prefix + "NanCount")),
                longValue(row.get(prefix + "PositiveInfinityCount")),
                longValue(row.get(prefix + "NegativeInfinityCount")),
                finiteOrZero(row.get(prefix + "NonFiniteFraction")));
    }

    private static long longValue(Object value) {
        return Math.max(0L, TrainingReportValues.longValue(value, 0L));
    }

    private static double doubleValue(Object value) {
        return optionalDouble(value).orElse(Double.NaN);
    }

    private static double finiteOrZero(Object value) {
        double doubleValue = doubleValue(value);
        return Double.isFinite(doubleValue) ? Math.max(0.0, doubleValue) : 0.0;
    }

    private record LatestOptimization(
            int epoch,
            long gradientValueCount,
            double gradientL2Norm,
            double gradientZeroFraction,
            boolean parameterUpdateDiagnosticsEnabled,
            double parameterUpdateL2Norm,
            double parameterUpdateToParameterL2Ratio) {
    }

    private record NonFiniteAggregate(
            NonFiniteCounts gradient,
            NonFiniteCounts parameters,
            NonFiniteCounts updates,
            List<Integer> epochs) {
        long totalNonFiniteCount() {
            return gradient.nonFiniteCount()
                    + parameters.nonFiniteCount()
                    + updates.nonFiniteCount();
        }
    }

    private record NonFiniteCounts(
            long nonFiniteCount,
            long nanCount,
            long positiveInfinityCount,
            long negativeInfinityCount,
            double maxNonFiniteFraction) {
        static NonFiniteCounts empty() {
            return new NonFiniteCounts(0L, 0L, 0L, 0L, 0.0);
        }

        NonFiniteCounts plus(NonFiniteCounts other) {
            return new NonFiniteCounts(
                    nonFiniteCount + other.nonFiniteCount,
                    nanCount + other.nanCount,
                    positiveInfinityCount + other.positiveInfinityCount,
                    negativeInfinityCount + other.negativeInfinityCount,
                    Math.max(maxNonFiniteFraction, other.maxNonFiniteFraction));
        }
    }
}
