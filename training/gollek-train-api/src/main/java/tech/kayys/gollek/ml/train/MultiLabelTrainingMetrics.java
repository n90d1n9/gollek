package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.List;
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
                int offset = row * currentLabels;
                for (int label = 0; label < currentLabels; label++) {
                    int index = offset + label;
                    boolean predictedPositive = predictionData[index] >= logitThreshold;
                    boolean actualPositive = TrainingMetricChecks.binaryTarget(targetData[index]);
                    if (predictedPositive && actualPositive) {
                        truePositive[label]++;
                    } else if (predictedPositive) {
                        falsePositive[label]++;
                    } else if (actualPositive) {
                        falseNegative[label]++;
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

}
