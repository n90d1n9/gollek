package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.normalizedString;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Dashboard-friendly bundle of chart-ready trainer report series.
 */
public record TrainingReportSeriesBundle(
        TrainingReportSeries trainLoss,
        TrainingReportSeries validationLoss,
        TrainingReportSeries learningRate,
        TrainingReportSeries generalizationGap,
        TrainingReportSeries validationToTrainLossRatio,
        TrainingReportSeries gradientL2Norm,
        TrainingReportSeries parameterUpdateToParameterL2Ratio,
        Map<String, TrainingReportSeries> trainMetrics,
        Map<String, TrainingReportSeries> validationMetrics) {
    public TrainingReportSeriesBundle {
        trainLoss = trainLoss == null ? TrainingReportSeries.empty("trainLoss") : trainLoss;
        validationLoss = validationLoss == null ? TrainingReportSeries.empty("validationLoss") : validationLoss;
        learningRate = learningRate == null ? TrainingReportSeries.empty("learningRate") : learningRate;
        generalizationGap = generalizationGap == null
                ? TrainingReportSeries.empty("generalizationGap")
                : generalizationGap;
        validationToTrainLossRatio = validationToTrainLossRatio == null
                ? TrainingReportSeries.empty("validationToTrainLossRatio")
                : validationToTrainLossRatio;
        gradientL2Norm = gradientL2Norm == null ? TrainingReportSeries.empty("gradientL2Norm") : gradientL2Norm;
        parameterUpdateToParameterL2Ratio = parameterUpdateToParameterL2Ratio == null
                ? TrainingReportSeries.empty("parameterUpdateToParameterL2Ratio")
                : parameterUpdateToParameterL2Ratio;
        trainMetrics = trainMetrics == null ? Map.of() : Map.copyOf(trainMetrics);
        validationMetrics = validationMetrics == null ? Map.of() : Map.copyOf(validationMetrics);
    }

    public static TrainingReportSeriesBundle fromSnapshots(List<TrainingReportEpochSnapshot> snapshots) {
        List<TrainingReportEpochSnapshot> rows = snapshots == null ? List.of() : List.copyOf(snapshots);
        return new TrainingReportSeriesBundle(
                TrainingReportSeries.fromSnapshots("trainLoss", rows, TrainingReportEpochSnapshot::trainLoss),
                TrainingReportSeries.fromSnapshots("validationLoss", rows, TrainingReportEpochSnapshot::validationLoss),
                TrainingReportSeries.fromSnapshots("learningRate", rows, TrainingReportEpochSnapshot::learningRate),
                TrainingReportSeries.fromSnapshots(
                        "generalizationGap",
                        rows,
                        TrainingReportEpochSnapshot::generalizationGap),
                TrainingReportSeries.fromSnapshots(
                        "validationToTrainLossRatio",
                        rows,
                        TrainingReportEpochSnapshot::validationToTrainLossRatio),
                TrainingReportSeries.fromSnapshots(
                        "gradientL2Norm",
                        rows,
                        snapshot -> snapshot.optimization().gradientL2Norm()),
                TrainingReportSeries.fromSnapshots(
                        "parameterUpdateToParameterL2Ratio",
                        rows,
                        snapshot -> snapshot.optimization().parameterUpdateToParameterL2Ratio()),
                metricSeries(rows, "trainMetric.", true),
                metricSeries(rows, "validationMetric.", false));
    }

    public static TrainingReportSeriesBundle empty() {
        return fromSnapshots(List.of());
    }

    public boolean hasTrainMetrics() {
        return !trainMetrics.isEmpty();
    }

    public boolean hasValidationMetrics() {
        return !validationMetrics.isEmpty();
    }

    public boolean hasOptimization() {
        return gradientL2Norm.available() || parameterUpdateToParameterL2Ratio.available();
    }

    public TrainingReportSeries trainMetric(String metricName) {
        String name = normalizedString(metricName, "");
        TrainingReportSeries series = trainMetrics.get(name);
        return series == null ? TrainingReportSeries.empty("trainMetric." + name) : series;
    }

    public TrainingReportSeries validationMetric(String metricName) {
        String name = normalizedString(metricName, "");
        TrainingReportSeries series = validationMetrics.get(name);
        return series == null ? TrainingReportSeries.empty("validationMetric." + name) : series;
    }

    public Map<String, TrainingReportSeries> allSeries() {
        Map<String, TrainingReportSeries> series = new LinkedHashMap<>();
        series.put(trainLoss.name(), trainLoss);
        series.put(validationLoss.name(), validationLoss);
        series.put(learningRate.name(), learningRate);
        series.put(generalizationGap.name(), generalizationGap);
        series.put(validationToTrainLossRatio.name(), validationToTrainLossRatio);
        series.put(gradientL2Norm.name(), gradientL2Norm);
        series.put(parameterUpdateToParameterL2Ratio.name(), parameterUpdateToParameterL2Ratio);
        trainMetrics.forEach((name, metricSeries) -> series.put(metricSeries.name(), metricSeries));
        validationMetrics.forEach((name, metricSeries) -> series.put(metricSeries.name(), metricSeries));
        return Map.copyOf(series);
    }

    public Map<String, TrainingReportSeries> availableSeries() {
        Map<String, TrainingReportSeries> series = new LinkedHashMap<>();
        allSeries().forEach((name, candidate) -> {
            if (candidate.available()) {
                series.put(name, candidate);
            }
        });
        return Map.copyOf(series);
    }

    private static Map<String, TrainingReportSeries> metricSeries(
            List<TrainingReportEpochSnapshot> snapshots,
            String prefix,
            boolean train) {
        TreeSet<String> metricNames = new TreeSet<>();
        for (TrainingReportEpochSnapshot snapshot : snapshots) {
            metricNames.addAll(train ? snapshot.trainMetrics().keySet() : snapshot.validationMetrics().keySet());
        }
        if (metricNames.isEmpty()) {
            return Map.of();
        }
        Map<String, TrainingReportSeries> series = new LinkedHashMap<>();
        for (String name : metricNames) {
            series.put(
                    name,
                    TrainingReportSeries.fromSnapshots(
                            prefix + name,
                            snapshots,
                            snapshot -> train ? snapshot.trainMetric(name) : snapshot.validationMetric(name)));
        }
        return Map.copyOf(series);
    }
}
