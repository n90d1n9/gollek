package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Metrics that stay meaningful when multiclass labels are imbalanced. */
final class ClassificationImbalanceTrainingMetrics {
    private ClassificationImbalanceTrainingMetrics() {
    }

    static Supplier<TrainingMetric> classificationBalancedAccuracy() {
        return ClassificationBalancedAccuracyMetric::new;
    }

    static Supplier<TrainingMetric> classificationMatthewsCorrelationCoefficient() {
        return ClassificationMatthewsCorrelationCoefficientMetric::new;
    }

    static Supplier<TrainingMetric> classificationMcc() {
        return classificationMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> classificationWeightedPrecision() {
        return ClassificationWeightedPrecisionMetric::new;
    }

    static Supplier<TrainingMetric> classificationWeightedRecall() {
        return ClassificationWeightedRecallMetric::new;
    }

    static Supplier<TrainingMetric> classificationWeightedF1() {
        return ClassificationWeightedF1Metric::new;
    }

    static Supplier<TrainingMetric> classificationCohensKappa() {
        return ClassificationCohensKappaMetric::new;
    }

    static Supplier<TrainingMetric> classificationKappa() {
        return classificationCohensKappa();
    }

    private abstract static class ClassificationConfusionStatsMetric implements DetailedTrainingMetric {
        private int classes = -1;
        private long[][] matrix = new long[0][0];
        private long[] support = new long[0];
        private long[] predictedSupport = new long[0];
        private long correct;
        private long total;

        @Override
        public void reset() {
            classes = -1;
            matrix = new long[0][0];
            support = new long[0];
            predictedSupport = new long[0];
            correct = 0L;
            total = 0L;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int currentClasses = Math.toIntExact(predictionShape[1]);
            ensureClassStorage(currentClasses);

            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int predictedClass = TrainingMetricChecks.argmax(
                        predictionData,
                        row * currentClasses,
                        currentClasses);
                int actualClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                matrix[actualClass][predictedClass]++;
                support[actualClass]++;
                predictedSupport[predictedClass]++;
                if (predictedClass == actualClass) {
                    correct++;
                }
                total++;
            }
        }

        protected Map<String, Object> baseDetails(String type) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", type);
            details.put("classes", Math.max(0, classes));
            details.put("definedClasses", definedClasses());
            details.put("total", total);
            details.put("correct", correct);
            details.put("accuracy", total == 0L ? null : correct / (double) total);
            details.put("balancedAccuracy", Double.isNaN(balancedAccuracy()) ? null : balancedAccuracy());
            details.put(
                    "matthewsCorrelationCoefficient",
                    Double.isNaN(matthewsCorrelationCoefficient()) ? null : matthewsCorrelationCoefficient());
            details.put("weightedPrecision", Double.isNaN(weightedPrecision()) ? null : weightedPrecision());
            details.put("weightedRecall", Double.isNaN(weightedRecall()) ? null : weightedRecall());
            details.put("weightedF1", Double.isNaN(weightedF1()) ? null : weightedF1());
            details.put("cohensKappa", Double.isNaN(cohensKappa()) ? null : cohensKappa());
            details.put("observedAgreement", Double.isNaN(observedAgreement()) ? null : observedAgreement());
            details.put("expectedAgreement", Double.isNaN(expectedAgreement()) ? null : expectedAgreement());
            details.put("rowMeaning", "actual_class");
            details.put("columnMeaning", "predicted_class");
            details.put("labels", classLabels());
            details.put("support", supportList());
            details.put("predictedSupport", predictedSupportList());
            details.put("perClassRecall", perClassRecall());
            details.put("perClassPrecision", perClassPrecision());
            details.put("matrix", matrixRows());
            return details;
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                matrix = new long[classes][classes];
                support = new long[classes];
                predictedSupport = new long[classes];
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        protected double balancedAccuracy() {
            int defined = definedClasses();
            if (defined == 0) {
                return Double.NaN;
            }
            double totalRecall = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                if (support[classIndex] > 0L) {
                    totalRecall += matrix[classIndex][classIndex] / (double) support[classIndex];
                }
            }
            return totalRecall / defined;
        }

        protected double matthewsCorrelationCoefficient() {
            if (total == 0L) {
                return Double.NaN;
            }
            double sampleCount = total;
            double correctCount = correct;
            double supportPredictedProductSum = 0.0;
            double supportSquaredSum = 0.0;
            double predictedSupportSquaredSum = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                supportPredictedProductSum += support[classIndex] * (double) predictedSupport[classIndex];
                supportSquaredSum += support[classIndex] * (double) support[classIndex];
                predictedSupportSquaredSum += predictedSupport[classIndex] * (double) predictedSupport[classIndex];
            }

            double numerator = correctCount * sampleCount - supportPredictedProductSum;
            double denominator = Math.sqrt(
                    (sampleCount * sampleCount - predictedSupportSquaredSum)
                            * (sampleCount * sampleCount - supportSquaredSum));
            return denominator == 0.0 ? 0.0 : numerator / denominator;
        }

        protected double weightedPrecision() {
            if (total == 0L) {
                return Double.NaN;
            }
            double weighted = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                weighted += support[classIndex] * precisionForClass(classIndex);
            }
            return weighted / total;
        }

        protected double weightedRecall() {
            if (total == 0L) {
                return Double.NaN;
            }
            double weighted = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                weighted += support[classIndex] * recallForClass(classIndex);
            }
            return weighted / total;
        }

        protected double weightedF1() {
            if (total == 0L) {
                return Double.NaN;
            }
            double weighted = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                double precision = precisionForClass(classIndex);
                double recall = recallForClass(classIndex);
                double f1 = precision + recall == 0.0 ? 0.0 : (2.0 * precision * recall) / (precision + recall);
                weighted += support[classIndex] * f1;
            }
            return weighted / total;
        }

        protected double cohensKappa() {
            if (total == 0L) {
                return Double.NaN;
            }
            double expected = expectedAgreement();
            double denominator = 1.0 - expected;
            return denominator == 0.0 ? 0.0 : (observedAgreement() - expected) / denominator;
        }

        private double observedAgreement() {
            return total == 0L ? Double.NaN : correct / (double) total;
        }

        private double expectedAgreement() {
            if (total == 0L) {
                return Double.NaN;
            }
            double agreement = 0.0;
            double totalSquared = total * (double) total;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                agreement += support[classIndex] * (double) predictedSupport[classIndex] / totalSquared;
            }
            return agreement;
        }

        private int definedClasses() {
            int defined = 0;
            for (long count : support) {
                if (count > 0L) {
                    defined++;
                }
            }
            return defined;
        }

        private List<Integer> classLabels() {
            if (classes <= 0) {
                return List.of();
            }
            List<Integer> labels = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                labels.add(classIndex);
            }
            return Collections.unmodifiableList(labels);
        }

        private List<Long> supportList() {
            List<Long> values = new ArrayList<>(support.length);
            for (long count : support) {
                values.add(count);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Long> predictedSupportList() {
            List<Long> values = new ArrayList<>(predictedSupport.length);
            for (long count : predictedSupport) {
                values.add(count);
            }
            return Collections.unmodifiableList(values);
        }

        private double precisionForClass(int classIndex) {
            return predictedSupport[classIndex] == 0L
                    ? 0.0
                    : matrix[classIndex][classIndex] / (double) predictedSupport[classIndex];
        }

        private double recallForClass(int classIndex) {
            return support[classIndex] == 0L
                    ? 0.0
                    : matrix[classIndex][classIndex] / (double) support[classIndex];
        }

        private List<Double> perClassRecall() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                values.add(support[classIndex] == 0L
                        ? null
                        : matrix[classIndex][classIndex] / (double) support[classIndex]);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perClassPrecision() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                values.add(predictedSupport[classIndex] == 0L
                        ? null
                        : matrix[classIndex][classIndex] / (double) predictedSupport[classIndex]);
            }
            return Collections.unmodifiableList(values);
        }

        private List<List<Long>> matrixRows() {
            if (classes <= 0) {
                return List.of();
            }
            List<List<Long>> rows = new ArrayList<>(classes);
            for (int row = 0; row < classes; row++) {
                List<Long> values = new ArrayList<>(classes);
                for (int column = 0; column < classes; column++) {
                    values.add(matrix[row][column]);
                }
                rows.add(Collections.unmodifiableList(values));
            }
            return Collections.unmodifiableList(rows);
        }
    }

    private static final class ClassificationBalancedAccuracyMetric extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_balanced_accuracy";
        }

        @Override
        public double value() {
            return balancedAccuracy();
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("classification_balanced_accuracy");
            details.put("averaging", "macro_recall_observed_classes");
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class ClassificationMatthewsCorrelationCoefficientMetric
            extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_matthews_correlation_coefficient";
        }

        @Override
        public double value() {
            return matthewsCorrelationCoefficient();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(baseDetails("classification_matthews_correlation_coefficient"));
        }
    }

    private static final class ClassificationWeightedPrecisionMetric extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_weighted_precision";
        }

        @Override
        public double value() {
            return weightedPrecision();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(baseDetails("classification_weighted_precision"));
        }
    }

    private static final class ClassificationWeightedRecallMetric extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_weighted_recall";
        }

        @Override
        public double value() {
            return weightedRecall();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(baseDetails("classification_weighted_recall"));
        }
    }

    private static final class ClassificationWeightedF1Metric extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_weighted_f1";
        }

        @Override
        public double value() {
            return weightedF1();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(baseDetails("classification_weighted_f1"));
        }
    }

    private static final class ClassificationCohensKappaMetric extends ClassificationConfusionStatsMetric {
        @Override
        public String name() {
            return "classification_cohens_kappa";
        }

        @Override
        public double value() {
            return cohensKappa();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(baseDetails("classification_cohens_kappa"));
        }
    }
}
