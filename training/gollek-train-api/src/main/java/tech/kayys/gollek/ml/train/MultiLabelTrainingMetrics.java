package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Built-in multilabel classification metric implementations. */
final class MultiLabelTrainingMetrics {
    private MultiLabelTrainingMetrics() {
    }

    static Supplier<TrainingMetric> multiLabelExactMatch() {
        return MultiLabelExactMatchMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelExactMatch(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelExactMatchMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss() {
        return MultiLabelHammingLossMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelHammingLossMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelConfusionMatrix() {
        return MultiLabelConfusionMatrixMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelConfusionMatrix(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelConfusionMatrixMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroPrecision() {
        return MultiLabelMicroPrecisionMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelMicroPrecision(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMicroPrecisionMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroRecall() {
        return MultiLabelMicroRecallMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelMicroRecall(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMicroRecallMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroF1() {
        return MultiLabelMicroF1Metric::new;
    }

    static Supplier<TrainingMetric> multiLabelMicroF1(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMicroF1Metric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelSamplePrecision() {
        return MultiLabelSamplePrecisionMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelSamplePrecision(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelSamplePrecisionMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleRecall() {
        return MultiLabelSampleRecallMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelSampleRecall(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelSampleRecallMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleF1() {
        return MultiLabelSampleF1Metric::new;
    }

    static Supplier<TrainingMetric> multiLabelSampleF1(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelSampleF1Metric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleJaccard() {
        return MultiLabelSampleJaccardMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelSampleJaccard(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelSampleJaccardMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision() {
        return MultiLabelMacroPrecisionMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMacroPrecisionMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall() {
        return MultiLabelMacroRecallMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMacroRecallMetric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroF1() {
        return MultiLabelMacroF1Metric::new;
    }

    static Supplier<TrainingMetric> multiLabelMacroF1(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new MultiLabelMacroF1Metric(threshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRocAuc() {
        return MultiLabelMacroRocAucMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelMacroAuroc() {
        return multiLabelMacroRocAuc();
    }

    static Supplier<TrainingMetric> multiLabelMacroAveragePrecision() {
        return MultiLabelMacroAveragePrecisionMetric::new;
    }

    private abstract static class MultiLabelStatsMetric implements TrainingMetric {
        private final float logitThreshold;
        private int labels = -1;
        private long[] truePositive = new long[0];
        private long[] falsePositive = new long[0];
        private long[] falseNegative = new long[0];
        private long exactMatchCount;
        private long sampleCount;
        private long labelCount;
        private long labelMismatchCount;
        private double samplePrecisionSum;
        private double sampleRecallSum;
        private double sampleF1Sum;
        private double sampleJaccardSum;

        MultiLabelStatsMetric() {
            this(0.0f);
        }

        MultiLabelStatsMetric(float logitThreshold) {
            this.logitThreshold = logitThreshold;
        }

        @Override
        public void reset() {
            labels = -1;
            truePositive = new long[0];
            falsePositive = new long[0];
            falseNegative = new long[0];
            exactMatchCount = 0;
            sampleCount = 0;
            labelCount = 0;
            labelMismatchCount = 0;
            samplePrecisionSum = 0.0;
            sampleRecallSum = 0.0;
            sampleF1Sum = 0.0;
            sampleJaccardSum = 0.0;
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
                boolean exactMatch = true;
                long rowTruePositive = 0L;
                long rowFalsePositive = 0L;
                long rowFalseNegative = 0L;
                int offset = row * currentLabels;
                for (int label = 0; label < currentLabels; label++) {
                    int index = offset + label;
                    boolean predictedPositive = predictionData[index] >= logitThreshold;
                    boolean actualPositive = TrainingMetricChecks.binaryTarget(targetData[index]);
                    if (predictedPositive && actualPositive) {
                        truePositive[label]++;
                        rowTruePositive++;
                    } else if (predictedPositive) {
                        falsePositive[label]++;
                        rowFalsePositive++;
                    } else if (actualPositive) {
                        falseNegative[label]++;
                        rowFalseNegative++;
                    }
                    if (predictedPositive != actualPositive) {
                        exactMatch = false;
                        labelMismatchCount++;
                    }
                    labelCount++;
                }
                if (exactMatch) {
                    exactMatchCount++;
                }
                samplePrecisionSum += ratioOrZero(rowTruePositive, rowTruePositive + rowFalsePositive);
                sampleRecallSum += ratioOrZero(rowTruePositive, rowTruePositive + rowFalseNegative);
                sampleF1Sum += ratioOrZero(
                        2 * rowTruePositive,
                        2 * rowTruePositive + rowFalsePositive + rowFalseNegative);
                sampleJaccardSum += ratioOrZero(
                        rowTruePositive,
                        rowTruePositive + rowFalsePositive + rowFalseNegative);
                sampleCount++;
            }
        }

        private void ensureLabelStorage(int currentLabels) {
            if (labels < 0) {
                labels = currentLabels;
                truePositive = new long[labels];
                falsePositive = new long[labels];
                falseNegative = new long[labels];
                return;
            }
            if (labels != currentLabels) {
                throw new IllegalArgumentException(
                        name() + " expected " + labels + " labels per sample but got " + currentLabels);
            }
        }

        protected double exactMatch() {
            return sampleCount == 0 ? Double.NaN : (double) exactMatchCount / sampleCount;
        }

        protected double hammingLoss() {
            return labelCount == 0 ? Double.NaN : (double) labelMismatchCount / labelCount;
        }

        protected double macroPrecision() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = truePositive[label] + falsePositive[label];
                total += denominator == 0 ? 0.0 : (double) truePositive[label] / denominator;
            }
            return total / labels;
        }

        protected double macroRecall() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = truePositive[label] + falseNegative[label];
                total += denominator == 0 ? 0.0 : (double) truePositive[label] / denominator;
            }
            return total / labels;
        }

        protected double macroF1() {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int label = 0; label < labels; label++) {
                long denominator = 2 * truePositive[label] + falsePositive[label] + falseNegative[label];
                total += denominator == 0 ? 0.0 : (double) (2 * truePositive[label]) / denominator;
            }
            return total / labels;
        }

        protected double microPrecision() {
            long truePositive = totalTruePositive();
            long denominator = truePositive + totalFalsePositive();
            return denominator == 0L ? 0.0 : (double) truePositive / denominator;
        }

        protected double microRecall() {
            long truePositive = totalTruePositive();
            long denominator = truePositive + totalFalseNegative();
            return denominator == 0L ? 0.0 : (double) truePositive / denominator;
        }

        protected double microF1() {
            long truePositive = totalTruePositive();
            long falsePositive = totalFalsePositive();
            long falseNegative = totalFalseNegative();
            long denominator = 2 * truePositive + falsePositive + falseNegative;
            return denominator == 0L ? 0.0 : (double) (2 * truePositive) / denominator;
        }

        protected double samplePrecision() {
            return sampleCount == 0L ? Double.NaN : samplePrecisionSum / sampleCount;
        }

        protected double sampleRecall() {
            return sampleCount == 0L ? Double.NaN : sampleRecallSum / sampleCount;
        }

        protected double sampleF1() {
            return sampleCount == 0L ? Double.NaN : sampleF1Sum / sampleCount;
        }

        protected double sampleJaccard() {
            return sampleCount == 0L ? Double.NaN : sampleJaccardSum / sampleCount;
        }

        protected float logitThreshold() {
            return logitThreshold;
        }

        protected int labels() {
            return labels;
        }

        protected long sampleCount() {
            return sampleCount;
        }

        protected long labelCount() {
            return labelCount;
        }

        protected long labelMismatchCount() {
            return labelMismatchCount;
        }

        protected long truePositive(int label) {
            return truePositive[label];
        }

        protected long falsePositive(int label) {
            return falsePositive[label];
        }

        protected long falseNegative(int label) {
            return falseNegative[label];
        }

        protected long trueNegative(int label) {
            return sampleCount - truePositive[label] - falsePositive[label] - falseNegative[label];
        }

        protected long totalTruePositive() {
            return total(truePositive);
        }

        protected long totalFalsePositive() {
            return total(falsePositive);
        }

        protected long totalFalseNegative() {
            return total(falseNegative);
        }

        protected double labelPrecision(int label) {
            long denominator = truePositive[label] + falsePositive[label];
            return denominator == 0L ? 0.0 : (double) truePositive[label] / denominator;
        }

        protected double labelRecall(int label) {
            long denominator = truePositive[label] + falseNegative[label];
            return denominator == 0L ? 0.0 : (double) truePositive[label] / denominator;
        }

        protected double labelF1(int label) {
            long denominator = 2 * truePositive[label] + falsePositive[label] + falseNegative[label];
            return denominator == 0L ? 0.0 : (double) (2 * truePositive[label]) / denominator;
        }

        protected double labelSpecificity(int label) {
            long trueNegative = trueNegative(label);
            long denominator = trueNegative + falsePositive[label];
            return denominator == 0L ? 0.0 : (double) trueNegative / denominator;
        }

        protected double labelBalancedAccuracy(int label) {
            return (labelRecall(label) + labelSpecificity(label)) / 2.0;
        }

        private static long total(long[] values) {
            long total = 0L;
            for (long value : values) {
                total += value;
            }
            return total;
        }

        private static double ratioOrZero(long numerator, long denominator) {
            return denominator == 0L ? 0.0 : (double) numerator / denominator;
        }
    }

    private static final class MultiLabelExactMatchMetric extends MultiLabelStatsMetric {
        MultiLabelExactMatchMetric() {
        }

        MultiLabelExactMatchMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_exact_match";
        }

        @Override
        public double value() {
            return exactMatch();
        }
    }

    private static final class MultiLabelHammingLossMetric extends MultiLabelStatsMetric {
        MultiLabelHammingLossMetric() {
        }

        MultiLabelHammingLossMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_hamming_loss";
        }

        @Override
        public double value() {
            return hammingLoss();
        }
    }

    private static final class MultiLabelConfusionMatrixMetric extends MultiLabelStatsMetric
            implements DetailedTrainingMetric {
        MultiLabelConfusionMatrixMetric() {
        }

        MultiLabelConfusionMatrixMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_confusion_matrix_macro_f1";
        }

        @Override
        public double value() {
            return macroF1();
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "multilabel_confusion_matrix");
            details.put("threshold", logitThreshold());
            details.put("labels", labelIndexes());
            details.put("labelCount", Math.max(0, labels()));
            details.put("samples", sampleCount());
            details.put("totalLabels", labelCount());
            details.put("labelMismatches", labelMismatchCount());
            details.put("exactMatch", exactMatch());
            details.put("hammingLoss", hammingLoss());
            details.put("macroPrecision", macroPrecision());
            details.put("macroRecall", macroRecall());
            details.put("macroF1", macroF1());
            details.put("microPrecision", microPrecision());
            details.put("microRecall", microRecall());
            details.put("microF1", microF1());
            details.put("samplePrecision", samplePrecision());
            details.put("sampleRecall", sampleRecall());
            details.put("sampleF1", sampleF1());
            details.put("sampleJaccard", sampleJaccard());
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("binaryLabels", List.of(0, 1));
            details.put("matrixByLabel", matrixByLabel());
            details.put("truePositive", perLabelLongs(this::truePositive));
            details.put("trueNegative", perLabelLongs(this::trueNegative));
            details.put("falsePositive", perLabelLongs(this::falsePositive));
            details.put("falseNegative", perLabelLongs(this::falseNegative));
            details.put("perLabelPrecision", perLabelDoubles(this::labelPrecision));
            details.put("perLabelRecall", perLabelDoubles(this::labelRecall));
            details.put("perLabelF1", perLabelDoubles(this::labelF1));
            details.put("perLabelSpecificity", perLabelDoubles(this::labelSpecificity));
            details.put("perLabelBalancedAccuracy", perLabelDoubles(this::labelBalancedAccuracy));
            return details;
        }

        private List<Integer> labelIndexes() {
            if (labels() <= 0) {
                return List.of();
            }
            List<Integer> indexes = new ArrayList<>(labels());
            for (int label = 0; label < labels(); label++) {
                indexes.add(label);
            }
            return Collections.unmodifiableList(indexes);
        }

        private List<List<List<Long>>> matrixByLabel() {
            if (labels() <= 0) {
                return List.of();
            }
            List<List<List<Long>>> matrices = new ArrayList<>(labels());
            for (int label = 0; label < labels(); label++) {
                matrices.add(List.of(
                        List.of(trueNegative(label), falsePositive(label)),
                        List.of(falseNegative(label), truePositive(label))));
            }
            return Collections.unmodifiableList(matrices);
        }

        private List<Long> perLabelLongs(LabelLongValue value) {
            if (labels() <= 0) {
                return List.of();
            }
            List<Long> values = new ArrayList<>(labels());
            for (int label = 0; label < labels(); label++) {
                values.add(value.get(label));
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perLabelDoubles(LabelDoubleValue value) {
            if (labels() <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(labels());
            for (int label = 0; label < labels(); label++) {
                values.add(value.get(label));
            }
            return Collections.unmodifiableList(values);
        }
    }

    private static final class MultiLabelMicroPrecisionMetric extends MultiLabelStatsMetric {
        MultiLabelMicroPrecisionMetric() {
        }

        MultiLabelMicroPrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_micro_precision";
        }

        @Override
        public double value() {
            return microPrecision();
        }
    }

    private static final class MultiLabelMicroRecallMetric extends MultiLabelStatsMetric {
        MultiLabelMicroRecallMetric() {
        }

        MultiLabelMicroRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_micro_recall";
        }

        @Override
        public double value() {
            return microRecall();
        }
    }

    private static final class MultiLabelMicroF1Metric extends MultiLabelStatsMetric {
        MultiLabelMicroF1Metric() {
        }

        MultiLabelMicroF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_micro_f1";
        }

        @Override
        public double value() {
            return microF1();
        }
    }

    private static final class MultiLabelSamplePrecisionMetric extends MultiLabelStatsMetric {
        MultiLabelSamplePrecisionMetric() {
        }

        MultiLabelSamplePrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_sample_precision";
        }

        @Override
        public double value() {
            return samplePrecision();
        }
    }

    private static final class MultiLabelSampleRecallMetric extends MultiLabelStatsMetric {
        MultiLabelSampleRecallMetric() {
        }

        MultiLabelSampleRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_sample_recall";
        }

        @Override
        public double value() {
            return sampleRecall();
        }
    }

    private static final class MultiLabelSampleF1Metric extends MultiLabelStatsMetric {
        MultiLabelSampleF1Metric() {
        }

        MultiLabelSampleF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_sample_f1";
        }

        @Override
        public double value() {
            return sampleF1();
        }
    }

    private static final class MultiLabelSampleJaccardMetric extends MultiLabelStatsMetric {
        MultiLabelSampleJaccardMetric() {
        }

        MultiLabelSampleJaccardMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_sample_jaccard";
        }

        @Override
        public double value() {
            return sampleJaccard();
        }
    }

    private static final class MultiLabelMacroPrecisionMetric extends MultiLabelStatsMetric {
        MultiLabelMacroPrecisionMetric() {
        }

        MultiLabelMacroPrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_precision";
        }

        @Override
        public double value() {
            return macroPrecision();
        }
    }

    private static final class MultiLabelMacroRecallMetric extends MultiLabelStatsMetric {
        MultiLabelMacroRecallMetric() {
        }

        MultiLabelMacroRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_recall";
        }

        @Override
        public double value() {
            return macroRecall();
        }
    }

    private static final class MultiLabelMacroF1Metric extends MultiLabelStatsMetric {
        MultiLabelMacroF1Metric() {
        }

        MultiLabelMacroF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "multilabel_macro_f1";
        }

        @Override
        public double value() {
            return macroF1();
        }
    }

    private abstract static class MultiLabelRankingMetric implements TrainingMetric {
        private int labels = -1;
        private List<List<TrainingMetricScore>> scoresByLabel = List.of();

        @Override
        public void reset() {
            labels = -1;
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
            }
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

        protected double macroRocAuc() {
            return macroDefinedScore(true);
        }

        protected double macroAveragePrecision() {
            return macroDefinedScore(false);
        }

        private double macroDefinedScore(boolean rocAuc) {
            if (labels <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            int defined = 0;
            for (List<TrainingMetricScore> labelScores : scoresByLabel) {
                double score = rocAuc
                        ? TrainingMetricRanking.binaryRocAuc(labelScores)
                        : TrainingMetricRanking.binaryAveragePrecision(labelScores);
                if (Double.isFinite(score)) {
                    total += score;
                    defined++;
                }
            }
            return defined == 0 ? Double.NaN : total / defined;
        }
    }

    private static final class MultiLabelMacroRocAucMetric extends MultiLabelRankingMetric {
        @Override
        public String name() {
            return "multilabel_macro_roc_auc";
        }

        @Override
        public double value() {
            return macroRocAuc();
        }
    }

    private static final class MultiLabelMacroAveragePrecisionMetric extends MultiLabelRankingMetric {
        @Override
        public String name() {
            return "multilabel_macro_average_precision";
        }

        @Override
        public double value() {
            return macroAveragePrecision();
        }
    }

    @FunctionalInterface
    private interface LabelLongValue {
        long get(int label);
    }

    @FunctionalInterface
    private interface LabelDoubleValue {
        double get(int label);
    }

}
