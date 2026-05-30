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

    static Supplier<TrainingMetric> binaryBalancedAccuracy() {
        return BinaryBalancedAccuracyMetric::new;
    }

    static Supplier<TrainingMetric> binaryBalancedAccuracy(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryBalancedAccuracyMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient() {
        return BinaryMatthewsCorrelationCoefficientMetric::new;
    }

    static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryMatthewsCorrelationCoefficientMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryMcc() {
        return binaryMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> binaryMcc(float logitThreshold) {
        return binaryMatthewsCorrelationCoefficient(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryCohensKappa() {
        return BinaryCohensKappaMetric::new;
    }

    static Supplier<TrainingMetric> binaryCohensKappa(float logitThreshold) {
        float threshold = TrainingMetricChecks.requireFiniteLogitThreshold(logitThreshold);
        return () -> new BinaryCohensKappaMetric(threshold);
    }

    static Supplier<TrainingMetric> binaryKappa() {
        return binaryCohensKappa();
    }

    static Supplier<TrainingMetric> binaryKappa(float logitThreshold) {
        return binaryCohensKappa(logitThreshold);
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

    static Supplier<TrainingMetric> binaryBestF1() {
        return BinaryBestF1Metric::new;
    }

    static Supplier<TrainingMetric> binaryBestF1Threshold() {
        return binaryBestF1();
    }

    static Supplier<TrainingMetric> binaryPrecisionAtRecall(double minimumRecall) {
        double checkedMinimumRecall = TrainingMetricChecks.requireUnitInterval("minimumRecall", minimumRecall);
        return () -> new BinaryPrecisionAtRecallMetric(checkedMinimumRecall);
    }

    static Supplier<TrainingMetric> binaryRecallAtPrecision(double minimumPrecision) {
        double checkedMinimumPrecision = TrainingMetricChecks.requireUnitInterval("minimumPrecision", minimumPrecision);
        return () -> new BinaryRecallAtPrecisionMetric(checkedMinimumPrecision);
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

        protected double binaryMatthewsCorrelationCoefficient() {
            if (total == 0) {
                return Double.NaN;
            }
            double numerator = truePositive * (double) trueNegative - falsePositive * (double) falseNegative;
            double denominator = Math.sqrt(
                    (truePositive + falsePositive) * (double) (truePositive + falseNegative)
                            * (trueNegative + falsePositive) * (double) (trueNegative + falseNegative));
            return denominator == 0.0 ? 0.0 : numerator / denominator;
        }

        protected double binaryCohensKappa() {
            if (total == 0) {
                return Double.NaN;
            }
            double expected = binaryExpectedAgreement();
            double denominator = 1.0 - expected;
            return denominator == 0.0 ? 0.0 : (binaryObservedAgreement() - expected) / denominator;
        }

        private double binaryObservedAgreement() {
            return total == 0 ? Double.NaN : (truePositive + trueNegative) / (double) total;
        }

        private double binaryExpectedAgreement() {
            if (total == 0) {
                return Double.NaN;
            }
            double actualNegative = trueNegative + falsePositive;
            double actualPositive = falseNegative + truePositive;
            double predictedNegative = trueNegative + falseNegative;
            double predictedPositive = falsePositive + truePositive;
            double totalSquared = total * (double) total;
            return (actualNegative * predictedNegative + actualPositive * predictedPositive) / totalSquared;
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
            details.put("matthewsCorrelationCoefficient", binaryMatthewsCorrelationCoefficient());
            details.put("cohensKappa", binaryCohensKappa());
            details.put("observedAgreement", binaryObservedAgreement());
            details.put("expectedAgreement", binaryExpectedAgreement());
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

    private static final class BinaryBalancedAccuracyMetric extends BinaryStatsMetric implements DetailedTrainingMetric {
        BinaryBalancedAccuracyMetric() {
        }

        BinaryBalancedAccuracyMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_balanced_accuracy";
        }

        @Override
        public double value() {
            return binaryBalancedAccuracy();
        }

        @Override
        public Map<String, Object> details() {
            return binaryConfusionDetails("binary_balanced_accuracy");
        }
    }

    private static final class BinaryCohensKappaMetric extends BinaryStatsMetric implements DetailedTrainingMetric {
        BinaryCohensKappaMetric() {
        }

        BinaryCohensKappaMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_cohens_kappa";
        }

        @Override
        public double value() {
            return binaryCohensKappa();
        }

        @Override
        public Map<String, Object> details() {
            return binaryConfusionDetails("binary_cohens_kappa");
        }
    }

    private static final class BinaryMatthewsCorrelationCoefficientMetric
            extends BinaryStatsMetric implements DetailedTrainingMetric {
        BinaryMatthewsCorrelationCoefficientMetric() {
        }

        BinaryMatthewsCorrelationCoefficientMetric(float logitThreshold) {
            super(logitThreshold);
        }

        @Override
        public String name() {
            return "binary_matthews_correlation_coefficient";
        }

        @Override
        public double value() {
            return binaryMatthewsCorrelationCoefficient();
        }

        @Override
        public Map<String, Object> details() {
            return binaryConfusionDetails("binary_matthews_correlation_coefficient");
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

    private static final class BinaryBestF1Metric extends BinaryRankingMetric implements DetailedTrainingMetric {
        @Override
        public String name() {
            return "binary_best_f1";
        }

        @Override
        public double value() {
            return TrainingMetricRanking.bestBinaryF1Threshold(scores()).f1();
        }

        @Override
        public Map<String, Object> details() {
            TrainingMetricThreshold best = TrainingMetricRanking.bestBinaryF1Threshold(scores());
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "binary_threshold_optimization");
            details.put("objective", "f1");
            details.put("samples", (long) scores().size());
            if (!best.defined()) {
                details.put("defined", false);
                return details;
            }
            details.put("defined", true);
            details.put("threshold", best.threshold());
            details.put("f1", best.f1());
            details.put("precision", best.precision());
            details.put("recall", best.recall());
            details.put("positives", best.positives());
            details.put("negatives", best.negatives());
            details.put("truePositive", best.truePositive());
            details.put("trueNegative", best.trueNegative());
            details.put("falsePositive", best.falsePositive());
            details.put("falseNegative", best.falseNegative());
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("labels", List.of(0, 1));
            details.put("matrix", List.of(
                    List.of(best.trueNegative(), best.falsePositive()),
                    List.of(best.falseNegative(), best.truePositive())));
            return details;
        }
    }

    private abstract static class BinaryConstrainedThresholdMetric
            extends BinaryRankingMetric implements DetailedTrainingMetric {
        private final String objective;
        private final String constraintName;
        private final double constraintValue;

        BinaryConstrainedThresholdMetric(String objective, String constraintName, double constraintValue) {
            this.objective = objective;
            this.constraintName = constraintName;
            this.constraintValue = constraintValue;
        }

        @Override
        public double value() {
            TrainingMetricThreshold threshold = bestThreshold();
            return threshold.defined() ? objectiveValue(threshold) : Double.NaN;
        }

        @Override
        public Map<String, Object> details() {
            TrainingMetricThreshold threshold = bestThreshold();
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "binary_threshold_constraint");
            details.put("objective", objective);
            details.put("constraint", constraintName);
            details.put("constraintValue", constraintValue);
            details.put("samples", (long) scores().size());
            details.put("positives", positiveCount());
            details.put("negatives", scores().size() - positiveCount());
            if (!threshold.defined()) {
                details.put("defined", false);
                return details;
            }
            details.put("defined", true);
            details.put("threshold", threshold.threshold());
            details.put("precision", threshold.precision());
            details.put("recall", threshold.recall());
            details.put("f1", threshold.f1());
            details.put("truePositive", threshold.truePositive());
            details.put("trueNegative", threshold.trueNegative());
            details.put("falsePositive", threshold.falsePositive());
            details.put("falseNegative", threshold.falseNegative());
            details.put("rowMeaning", "actual_label");
            details.put("columnMeaning", "predicted_label");
            details.put("labels", List.of(0, 1));
            details.put("matrix", List.of(
                    List.of(threshold.trueNegative(), threshold.falsePositive()),
                    List.of(threshold.falseNegative(), threshold.truePositive())));
            return details;
        }

        private TrainingMetricThreshold bestThreshold() {
            if (scores().isEmpty()) {
                return TrainingMetricThreshold.empty();
            }
            List<TrainingMetricScore> values = new ArrayList<>(scores());
            values.sort((left, right) -> Float.compare(right.score(), left.score()));

            long positives = positiveCount();
            if (positives == 0L) {
                return TrainingMetricThreshold.empty();
            }
            long negatives = values.size() - positives;
            TrainingMetricThreshold best = TrainingMetricThreshold.empty();
            long truePositive = 0L;
            long falsePositive = 0L;
            int index = 0;
            while (index < values.size()) {
                float threshold = values.get(index).score();
                int groupEnd = index + 1;
                long groupPositive = values.get(index).positive() ? 1L : 0L;
                long groupNegative = values.get(index).positive() ? 0L : 1L;
                while (groupEnd < values.size()
                        && Float.compare(values.get(groupEnd).score(), threshold) == 0) {
                    if (values.get(groupEnd).positive()) {
                        groupPositive++;
                    } else {
                        groupNegative++;
                    }
                    groupEnd++;
                }

                truePositive += groupPositive;
                falsePositive += groupNegative;
                TrainingMetricThreshold candidate = TrainingMetricThreshold.of(
                        threshold,
                        truePositive,
                        negatives - falsePositive,
                        falsePositive,
                        positives - truePositive);
                if (satisfiesConstraint(candidate) && isBetter(candidate, best)) {
                    best = candidate;
                }
                index = groupEnd;
            }
            return best;
        }

        private boolean isBetter(TrainingMetricThreshold candidate, TrainingMetricThreshold best) {
            if (!best.defined()) {
                return true;
            }
            int objectiveComparison = Double.compare(objectiveValue(candidate), objectiveValue(best));
            if (objectiveComparison != 0) {
                return objectiveComparison > 0;
            }
            int f1Comparison = Double.compare(candidate.f1(), best.f1());
            if (f1Comparison != 0) {
                return f1Comparison > 0;
            }
            return Float.compare(candidate.threshold(), best.threshold()) > 0;
        }

        protected abstract boolean satisfiesConstraint(TrainingMetricThreshold threshold);

        protected abstract double objectiveValue(TrainingMetricThreshold threshold);
    }

    private static final class BinaryPrecisionAtRecallMetric extends BinaryConstrainedThresholdMetric {
        private final double minimumRecall;

        BinaryPrecisionAtRecallMetric(double minimumRecall) {
            super("precision_at_recall", "minimumRecall", minimumRecall);
            this.minimumRecall = minimumRecall;
        }

        @Override
        public String name() {
            return "binary_precision_at_recall";
        }

        @Override
        protected boolean satisfiesConstraint(TrainingMetricThreshold threshold) {
            return threshold.recall() >= minimumRecall;
        }

        @Override
        protected double objectiveValue(TrainingMetricThreshold threshold) {
            return threshold.precision();
        }
    }

    private static final class BinaryRecallAtPrecisionMetric extends BinaryConstrainedThresholdMetric {
        private final double minimumPrecision;

        BinaryRecallAtPrecisionMetric(double minimumPrecision) {
            super("recall_at_precision", "minimumPrecision", minimumPrecision);
            this.minimumPrecision = minimumPrecision;
        }

        @Override
        public String name() {
            return "binary_recall_at_precision";
        }

        @Override
        protected boolean satisfiesConstraint(TrainingMetricThreshold threshold) {
            return threshold.precision() >= minimumPrecision;
        }

        @Override
        protected double objectiveValue(TrainingMetricThreshold threshold) {
            return threshold.recall();
        }
    }

}
