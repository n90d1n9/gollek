package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Built-in binary classification metric implementations. */
final class BinaryTrainingMetrics {
    private BinaryTrainingMetrics() {
    }

    static Supplier<TrainingMetric> binaryAccuracy() {
        return BinaryAccuracyMetric::new;
    }

    static Supplier<TrainingMetric> binaryAccuracy(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryAccuracyMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix() {
        return BinaryConfusionMatrixMetric::new;
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryConfusionMatrixMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryPrecision() {
        return BinaryPrecisionMetric::new;
    }

    static Supplier<TrainingMetric> binaryPrecision(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryPrecisionMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryRecall() {
        return BinaryRecallMetric::new;
    }

    static Supplier<TrainingMetric> binaryRecall(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryRecallMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryF1() {
        return BinaryF1Metric::new;
    }

    static Supplier<TrainingMetric> binaryF1(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryF1Metric(threshold);
    }

    static Supplier<TrainingMetric> binaryRocAuc() {
        return BinaryRocAucMetric::new;
    }

    static Supplier<TrainingMetric> binaryAuroc() {
        return binaryRocAuc();
    }

    static Supplier<TrainingMetric> binaryAveragePrecision() {
        return BinaryAveragePrecisionMetric::new;
    }

    private abstract static class BinaryStatsMetric implements TrainingMetric {
        private final float logitThreshold;
        private long truePositive;
        private long trueNegative;
        private long falsePositive;
        private long falseNegative;
        private long total;

        BinaryStatsMetric() {
            this(0.0f);
        }

        BinaryStatsMetric(float logitThreshold) {
            this.logitThreshold = logitThreshold;
        }

        @Override
        public void reset() {
            truePositive = 0;
            trueNegative = 0;
            falsePositive = 0;
            falseNegative = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                boolean predictedPositive = predictionData[i] >= logitThreshold;
                boolean actualPositive = TrainingMetricChecks.binaryTarget(targetData[i]);
                if (predictedPositive && actualPositive) {
                    truePositive++;
                } else if (predictedPositive) {
                    falsePositive++;
                } else if (actualPositive) {
                    falseNegative++;
                } else {
                    trueNegative++;
                }
                total++;
            }
        }

        protected double binaryAccuracy() {
            return total == 0 ? Double.NaN : (double) (truePositive + trueNegative) / total;
        }

        protected double binaryPrecision() {
            long denominator = truePositive + falsePositive;
            return denominator == 0 ? 0.0 : (double) truePositive / denominator;
        }

        protected double binaryRecall() {
            long denominator = truePositive + falseNegative;
            return denominator == 0 ? 0.0 : (double) truePositive / denominator;
        }

        protected double binaryF1() {
            long denominator = 2 * truePositive + falsePositive + falseNegative;
            return denominator == 0 ? 0.0 : (double) (2 * truePositive) / denominator;
        }

        protected double binarySpecificity() {
            long denominator = trueNegative + falsePositive;
            return denominator == 0 ? 0.0 : (double) trueNegative / denominator;
        }

        protected double binaryNegativePredictiveValue() {
            long denominator = trueNegative + falseNegative;
            return denominator == 0 ? 0.0 : (double) trueNegative / denominator;
        }

        protected double binaryBalancedAccuracy() {
            return total == 0 ? Double.NaN : (binaryRecall() + binarySpecificity()) / 2.0;
        }

        protected Map<String, Object> binaryConfusionDetails(String type) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", type);
            details.put("threshold", logitThreshold);
            details.put("total", total);
            details.put("trueNegative", trueNegative);
            details.put("falsePositive", falsePositive);
            details.put("falseNegative", falseNegative);
            details.put("truePositive", truePositive);
            details.put("accuracy", binaryAccuracy());
            details.put("precision", binaryPrecision());
            details.put("recall", binaryRecall());
            details.put("f1", binaryF1());
            details.put("specificity", binarySpecificity());
            details.put("negativePredictiveValue", binaryNegativePredictiveValue());
            details.put("balancedAccuracy", binaryBalancedAccuracy());
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("labels", List.of(0, 1));
            details.put("matrix", List.of(
                    List.of(trueNegative, falsePositive),
                    List.of(falseNegative, truePositive)));
            return details;
        }
    }

    private static final class BinaryAccuracyMetric extends BinaryStatsMetric {
        BinaryAccuracyMetric() {
        }

        BinaryAccuracyMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_accuracy";
        }

        @Override
        public double value() {
            return binaryAccuracy();
        }
    }

    private static final class BinaryConfusionMatrixMetric extends BinaryStatsMetric implements DetailedTrainingMetric {
        BinaryConfusionMatrixMetric() {
        }

        BinaryConfusionMatrixMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_confusion_matrix_accuracy";
        }

        @Override
        public double value() {
            return binaryAccuracy();
        }

        @Override
        public Map<String, Object> details() {
            return binaryConfusionDetails("binary_confusion_matrix");
        }
    }

    private static final class BinaryPrecisionMetric extends BinaryStatsMetric {
        BinaryPrecisionMetric() {
        }

        BinaryPrecisionMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_precision";
        }

        @Override
        public double value() {
            return binaryPrecision();
        }
    }

    private static final class BinaryRecallMetric extends BinaryStatsMetric {
        BinaryRecallMetric() {
        }

        BinaryRecallMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_recall";
        }

        @Override
        public double value() {
            return binaryRecall();
        }
    }

    private static final class BinaryF1Metric extends BinaryStatsMetric {
        BinaryF1Metric() {
        }

        BinaryF1Metric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_f1";
        }

        @Override
        public double value() {
            return binaryF1();
        }
    }

    private abstract static class BinaryRankingMetric implements TrainingMetric {
        private final List<TrainingMetricScore> scores = new ArrayList<>();

        @Override
        public void reset() {
            scores.clear();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                scores.add(new TrainingMetricScore(predictionData[i], TrainingMetricChecks.binaryTarget(targetData[i])));
            }
        }

        protected List<TrainingMetricScore> scores() {
            return scores;
        }

        protected long positiveCount() {
            return scores.stream().filter(TrainingMetricScore::positive).count();
        }
    }

    private static final class BinaryRocAucMetric extends BinaryRankingMetric {
        @Override
        public String name() {
            return "binary_roc_auc";
        }

        @Override
        public double value() {
            return TrainingMetricRanking.binaryRocAuc(scores());
        }
    }

    private static final class BinaryAveragePrecisionMetric extends BinaryRankingMetric {
        @Override
        public String name() {
            return "binary_average_precision";
        }

        @Override
        public double value() {
            return TrainingMetricRanking.binaryAveragePrecision(scores());
        }
    }

}
