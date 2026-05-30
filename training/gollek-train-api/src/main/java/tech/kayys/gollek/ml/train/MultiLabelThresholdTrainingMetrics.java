package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Per-label multilabel threshold tuning metrics computed from raw prediction scores. */
final class MultiLabelThresholdTrainingMetrics {
    private MultiLabelThresholdTrainingMetrics() {
    }

    static Supplier<TrainingMetric> multiLabelMacroBestF1() {
        return MultiLabelMacroBestF1Metric::new;
    }

    static Supplier<TrainingMetric> multiLabelBestF1Thresholds() {
        return multiLabelMacroBestF1();
    }

    private static final class MultiLabelMacroBestF1Metric implements DetailedTrainingMetric {
        private int labels = -1;
        private long samples;
        private List<List<TrainingMetricScore>> scoresByLabel = List.of();

        @Override
        public String name() {
            return "multilabel_macro_best_f1";
        }

        @Override
        public void reset() {
            labels = -1;
            samples = 0L;
            scoresByLabel = List.of();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape(name(), predictions, targets);
            long[] shape = predictions.shape();
            int currentSamples = TrainingMetricChecks.multiLabelSampleCount(shape);
            int currentLabels = TrainingMetricChecks.multiLabelLabelsPerSample(shape);
            ensureLabelStorage(currentLabels);

            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int row = 0; row < currentSamples; row++) {
                int offset = row * currentLabels;
                for (int label = 0; label < currentLabels; label++) {
                    int index = offset + label;
                    scoresByLabel.get(label).add(new TrainingMetricScore(
                            predictionData[index],
                            TrainingMetricChecks.binaryTarget(targetData[index])));
                }
                samples++;
            }
        }

        @Override
        public double value() {
            return macroValue(TrainingMetricThreshold::f1);
        }

        @Override
        public Map<String, Object> details() {
            List<TrainingMetricThreshold> thresholds = thresholds();
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "multilabel_threshold_optimization");
            details.put("objective", "f1");
            details.put("aggregation", "macro");
            details.put("defined", labels > 0);
            details.put("samples", samples);
            details.put("labelCount", Math.max(0, labels));
            details.put("labels", labelIndexes());
            details.put("macroF1", thresholds.isEmpty() ? null : macroValue(thresholds, TrainingMetricThreshold::f1));
            details.put("macroPrecision",
                    thresholds.isEmpty() ? null : macroValue(thresholds, TrainingMetricThreshold::precision));
            details.put("macroRecall",
                    thresholds.isEmpty() ? null : macroValue(thresholds, TrainingMetricThreshold::recall));
            details.put("perLabelThreshold", perLabel(thresholds, TrainingMetricThreshold::threshold));
            details.put("perLabelF1", perLabel(thresholds, TrainingMetricThreshold::f1));
            details.put("perLabelPrecision", perLabel(thresholds, TrainingMetricThreshold::precision));
            details.put("perLabelRecall", perLabel(thresholds, TrainingMetricThreshold::recall));
            details.put("truePositive", perLabelLong(thresholds, TrainingMetricThreshold::truePositive));
            details.put("trueNegative", perLabelLong(thresholds, TrainingMetricThreshold::trueNegative));
            details.put("falsePositive", perLabelLong(thresholds, TrainingMetricThreshold::falsePositive));
            details.put("falseNegative", perLabelLong(thresholds, TrainingMetricThreshold::falseNegative));
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("binaryLabels", List.of(0, 1));
            details.put("matrixByLabel", matrixByLabel(thresholds));
            return Collections.unmodifiableMap(details);
        }

        private void ensureLabelStorage(int currentLabels) {
            if (labels < 0) {
                labels = currentLabels;
                List<List<TrainingMetricScore>> lists = new ArrayList<>(labels);
                for (int i = 0; i < labels; i++) {
                    lists.add(new ArrayList<>());
                }
                scoresByLabel = lists;
                return;
            }
            if (labels != currentLabels) {
                throw new IllegalArgumentException(
                        name() + " expected " + labels + " labels per sample but got " + currentLabels);
            }
        }

        private List<TrainingMetricThreshold> thresholds() {
            if (labels <= 0) {
                return List.of();
            }
            List<TrainingMetricThreshold> thresholds = new ArrayList<>(labels);
            for (List<TrainingMetricScore> scores : scoresByLabel) {
                thresholds.add(TrainingMetricRanking.bestBinaryF1Threshold(scores));
            }
            return Collections.unmodifiableList(thresholds);
        }

        private double macroValue(ThresholdDoubleValue value) {
            return macroValue(thresholds(), value);
        }

        private double macroValue(List<TrainingMetricThreshold> thresholds, ThresholdDoubleValue value) {
            if (thresholds.isEmpty()) {
                return Double.NaN;
            }
            double total = 0.0;
            for (TrainingMetricThreshold threshold : thresholds) {
                total += threshold.defined() ? value.get(threshold) : 0.0;
            }
            return total / thresholds.size();
        }

        private List<Integer> labelIndexes() {
            if (labels <= 0) {
                return List.of();
            }
            List<Integer> indexes = new ArrayList<>(labels);
            for (int label = 0; label < labels; label++) {
                indexes.add(label);
            }
            return Collections.unmodifiableList(indexes);
        }

        private List<Object> perLabel(List<TrainingMetricThreshold> thresholds, ThresholdObjectValue value) {
            if (thresholds.isEmpty()) {
                return List.of();
            }
            List<Object> values = new ArrayList<>(thresholds.size());
            for (TrainingMetricThreshold threshold : thresholds) {
                values.add(threshold.defined() ? value.get(threshold) : null);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Long> perLabelLong(List<TrainingMetricThreshold> thresholds, ThresholdLongValue value) {
            if (thresholds.isEmpty()) {
                return List.of();
            }
            List<Long> values = new ArrayList<>(thresholds.size());
            for (TrainingMetricThreshold threshold : thresholds) {
                values.add(threshold.defined() ? value.get(threshold) : 0L);
            }
            return Collections.unmodifiableList(values);
        }

        private List<List<List<Long>>> matrixByLabel(List<TrainingMetricThreshold> thresholds) {
            if (thresholds.isEmpty()) {
                return List.of();
            }
            List<List<List<Long>>> matrices = new ArrayList<>(thresholds.size());
            for (TrainingMetricThreshold threshold : thresholds) {
                matrices.add(List.of(
                        List.of(threshold.trueNegative(), threshold.falsePositive()),
                        List.of(threshold.falseNegative(), threshold.truePositive())));
            }
            return Collections.unmodifiableList(matrices);
        }
    }

    @FunctionalInterface
    private interface ThresholdDoubleValue {
        double get(TrainingMetricThreshold threshold);
    }

    @FunctionalInterface
    private interface ThresholdObjectValue {
        Object get(TrainingMetricThreshold threshold);
    }

    @FunctionalInterface
    private interface ThresholdLongValue {
        long get(TrainingMetricThreshold threshold);
    }
}
