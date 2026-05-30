package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Report-level learning-rate health checks derived from epoch telemetry.
 */
final class TrainerLearningRateReportDiagnostics {
    private TrainerLearningRateReportDiagnostics() {
    }

    static void addFindings(
            List<TrainingReportDiagnostics.Finding> findings,
            List<Map<String, Object>> history,
            Map<String, Object> summary,
            TrainingReportDiagnostics.Options options) {
        if (findings == null || history == null || history.isEmpty() || summary == null) {
            return;
        }
        Map<String, Object> learningRate = mapSection(summary, "learningRate");
        if (!available(learningRate)) {
            return;
        }
        addTooSmallLatestLearningRateFinding(findings, learningRate, options);
        addLearningRateSpikeFinding(findings, history, options);
        addFlatLearningRateWithPlateauFinding(findings, learningRate, options);
    }

    private static void addTooSmallLatestLearningRateFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            Map<String, Object> learningRate,
            TrainingReportDiagnostics.Options options) {
        OptionalDouble latest = optionalDouble(learningRate.get("latest"));
        if (latest.isEmpty() || latest.getAsDouble() > options.learningRateTinyThreshold()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        putIfPresent(evidence, "latest", learningRate.get("latest"));
        putIfPresent(evidence, "latestEpoch", learningRate.get("latestEpoch"));
        evidence.put("tinyThreshold", options.learningRateTinyThreshold());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "learning_rate.too_small",
                "Latest learning rate is effectively zero, so additional optimizer steps may not update the model.",
                evidence));
    }

    private static void addLearningRateSpikeFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            List<Map<String, Object>> history,
            TrainingReportDiagnostics.Options options) {
        List<Point> points = finiteLearningRatePoints(history);
        if (points.size() < 2) {
            return;
        }
        Spike largest = null;
        for (int index = 1; index < points.size(); index++) {
            Point previous = points.get(index - 1);
            Point current = points.get(index);
            if (previous.value() <= options.learningRateTinyThreshold()) {
                continue;
            }
            double ratio = current.value() / previous.value();
            if (Double.isFinite(ratio) && ratio >= options.learningRateSpikeRatioThreshold()) {
                Spike spike = new Spike(previous, current, ratio);
                if (largest == null || spike.ratio() > largest.ratio()) {
                    largest = spike;
                }
            }
        }
        if (largest == null) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("previousEpoch", largest.previous().epoch());
        evidence.put("currentEpoch", largest.current().epoch());
        evidence.put("previousLearningRate", largest.previous().value());
        evidence.put("currentLearningRate", largest.current().value());
        evidence.put("ratio", largest.ratio());
        evidence.put("spikeRatioThreshold", options.learningRateSpikeRatioThreshold());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "learning_rate.spiked",
                "Learning rate jumped sharply between consecutive epochs.",
                evidence));
    }

    private static void addFlatLearningRateWithPlateauFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            Map<String, Object> learningRate,
            TrainingReportDiagnostics.Options options) {
        if (!hasCode(findings, "train.loss_plateau")) {
            return;
        }
        int count = intValue(learningRate.get("count"), 0);
        OptionalDouble relativeDelta = optionalDouble(learningRate.get("relativeDeltaFromFirst"));
        boolean flatTrend = "flat".equals(learningRate.get("trend"));
        boolean flatByMagnitude = relativeDelta.isPresent()
                && Math.abs(relativeDelta.getAsDouble()) <= options.learningRateFlatRelativeDeltaThreshold();
        if (count < options.trainLossPlateauMinEpochs() || (!flatTrend && !flatByMagnitude)) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("count", count);
        putIfPresent(evidence, "first", learningRate.get("first"));
        putIfPresent(evidence, "latest", learningRate.get("latest"));
        putIfPresent(evidence, "relativeDeltaFromFirst", learningRate.get("relativeDeltaFromFirst"));
        evidence.put("flatRelativeDeltaThreshold", options.learningRateFlatRelativeDeltaThreshold());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.INFO,
                "learning_rate.flat_with_train_plateau",
                "Training loss plateaued while the learning rate stayed flat.",
                evidence));
    }

    private static List<Point> finiteLearningRatePoints(List<Map<String, Object>> history) {
        List<Point> points = new ArrayList<>();
        for (Map<String, Object> row : history) {
            OptionalDouble learningRate = optionalDouble(row.get("learningRate"));
            if (learningRate.isPresent()) {
                points.add(new Point(intValue(row.get("epoch"), points.size()), learningRate.getAsDouble()));
            }
        }
        return List.copyOf(points);
    }

    private static boolean hasCode(List<TrainingReportDiagnostics.Finding> findings, String code) {
        return findings.stream().anyMatch(finding -> code.equals(finding.code()));
    }

    private static boolean available(Map<String, Object> section) {
        return TrainingReportValues.booleanValue(section.get("available"));
    }

    private static Map<String, Object> mapSection(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            copy.put(String.valueOf(entry.getKey()), TrainerMetadataSupport.immutableSnapshot(entry.getValue()));
        }
        return Map.copyOf(copy);
    }

    private static void putIfPresent(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private record Point(int epoch, double value) {
    }

    private record Spike(Point previous, Point current, double ratio) {
    }
}
