package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Built-in multiclass classification metric implementations. */
final class ClassificationTrainingMetrics {
    private ClassificationTrainingMetrics() {
    }

    static Supplier<TrainingMetric> classificationAccuracy() {
        return AccuracyMetric::new;
    }

    static Supplier<TrainingMetric> accuracy() {
        return classificationAccuracy();
    }

    static Supplier<TrainingMetric> classificationConfusionMatrix() {
        return ClassificationConfusionMatrixMetric::new;
    }

    static Supplier<TrainingMetric> confusionMatrix() {
        return classificationConfusionMatrix();
    }

    static Supplier<TrainingMetric> topKAccuracy(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("top-k accuracy requires k >= 1, got " + k);
        }
        return () -> new TopKAccuracyMetric(k);
    }

    static Supplier<TrainingMetric> classificationLogLoss() {
        return ClassificationLogLossMetric::new;
    }

    static Supplier<TrainingMetric> classificationCrossEntropy() {
        return classificationLogLoss();
    }

    static Supplier<TrainingMetric> precision() {
        return PrecisionMetric::new;
    }

    static Supplier<TrainingMetric> recall() {
        return RecallMetric::new;
    }

    static Supplier<TrainingMetric> f1() {
        return F1Metric::new;
    }

    static Supplier<TrainingMetric> macroF1() {
        return f1();
    }

    static Supplier<TrainingMetric> classificationMacroRocAuc() {
        return ClassificationMacroRocAucMetric::new;
    }

    static Supplier<TrainingMetric> classificationMacroAuroc() {
        return classificationMacroRocAuc();
    }

    static Supplier<TrainingMetric> classificationMacroAveragePrecision() {
        return ClassificationMacroAveragePrecisionMetric::new;
    }

    private static final class AccuracyMetric implements TrainingMetric {
        private long correct;
        private long total;

        @Override
        public String name() {
            return "accuracy";
        }

        @Override
        public void reset() {
            correct = 0;
            total = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            if (predictionShape.length != 2) {
                throw new IllegalArgumentException(
                        "accuracy expects predictions shaped [batch, classes], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int classes = Math.toIntExact(predictionShape[1]);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int predictedClass = TrainingMetricChecks.argmax(predictionData, row * classes, classes);
                int targetClass = TrainingMetricChecks.targetClass(targets, row, batch, classes);
                if (predictedClass == targetClass) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }
    }

    private static final class ClassificationConfusionMatrixMetric implements DetailedTrainingMetric {
        private int classes = -1;
        private long[][] matrix = new long[0][0];
        private long correct;
        private long total;

        @Override
        public String name() {
            return "confusion_matrix_accuracy";
        }

        @Override
        public void reset() {
            classes = -1;
            matrix = new long[0][0];
            correct = 0;
            total = 0;
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
                int predictedClass = TrainingMetricChecks.argmax(predictionData, row * currentClasses, currentClasses);
                int actualClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                matrix[actualClass][predictedClass]++;
                if (predictedClass == actualClass) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "classification_confusion_matrix");
            details.put("classes", Math.max(0, classes));
            details.put("total", total);
            details.put("correct", correct);
            details.put("accuracy", value());
            details.put("rowMeaning", "actual_class");
            details.put("columnMeaning", "predicted_class");
            details.put("labels", classLabels());
            details.put("matrix", matrixRows());
            details.put("perClassPrecision", perClassPrecision());
            details.put("perClassRecall", perClassRecall());
            details.put("perClassF1", perClassF1());
            return details;
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                matrix = new long[classes][classes];
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        private List<Integer> classLabels() {
            if (classes <= 0) {
                return List.of();
            }
            List<Integer> labels = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                labels.add(classIndex);
            }
            return labels;
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

        private List<Double> perClassPrecision() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                long predicted = 0L;
                for (int row = 0; row < classes; row++) {
                    predicted += matrix[row][classIndex];
                }
                values.add(predicted == 0L ? 0.0 : (double) matrix[classIndex][classIndex] / predicted);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perClassRecall() {
            if (classes <= 0) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(classes);
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                long actual = 0L;
                for (int column = 0; column < classes; column++) {
                    actual += matrix[classIndex][column];
                }
                values.add(actual == 0L ? 0.0 : (double) matrix[classIndex][classIndex] / actual);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> perClassF1() {
            List<Double> precision = perClassPrecision();
            List<Double> recall = perClassRecall();
            if (precision.isEmpty()) {
                return List.of();
            }
            List<Double> values = new ArrayList<>(precision.size());
            for (int classIndex = 0; classIndex < precision.size(); classIndex++) {
                double p = precision.get(classIndex);
                double r = recall.get(classIndex);
                values.add(p + r == 0.0 ? 0.0 : (2.0 * p * r) / (p + r));
            }
            return Collections.unmodifiableList(values);
        }
    }

    private static final class TopKAccuracyMetric implements TrainingMetric {
        private final int k;
        private long correct;
        private long total;

        private TopKAccuracyMetric(int k) {
            this.k = k;
        }

        @Override
        public String name() {
            return "top" + k + "_accuracy";
        }

        @Override
        public void reset() {
            correct = 0;
            total = 0;
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
            int classes = Math.toIntExact(predictionShape[1]);
            float[] predictionData = predictions.data();
            for (int row = 0; row < batch; row++) {
                int targetClass = TrainingMetricChecks.targetClass(targets, row, batch, classes);
                if (TrainingMetricChecks.containsTopK(predictionData, row * classes, classes, targetClass, Math.min(k, classes))) {
                    correct++;
                }
                total++;
            }
        }

        @Override
        public double value() {
            return total == 0 ? Double.NaN : (double) correct / total;
        }
    }

    private static final class ClassificationLogLossMetric implements DetailedTrainingMetric {
        private int classes = -1;
        private long samples;
        private double totalLogLoss;
        private double correctClassProbabilitySum;
        private double maximumSampleLogLoss;

        @Override
        public String name() {
            return "classification_log_loss";
        }

        @Override
        public void reset() {
            classes = -1;
            samples = 0L;
            totalLogLoss = 0.0;
            correctClassProbabilitySum = 0.0;
            maximumSampleLogLoss = 0.0;
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
                int offset = row * currentClasses;
                double maxLogit = Double.NEGATIVE_INFINITY;
                for (int classIndex = 0; classIndex < currentClasses; classIndex++) {
                    double logit = predictionData[offset + classIndex];
                    if (!Double.isFinite(logit)) {
                        throw new IllegalArgumentException(
                                name() + " expects finite logits, got " + logit
                                        + " at sample " + row + ", class " + classIndex);
                    }
                    maxLogit = Math.max(maxLogit, logit);
                }

                double sumExp = 0.0;
                for (int classIndex = 0; classIndex < currentClasses; classIndex++) {
                    sumExp += Math.exp(predictionData[offset + classIndex] - maxLogit);
                }
                double logSumExp = maxLogit + Math.log(sumExp);
                int targetClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                double sampleLogLoss = logSumExp - predictionData[offset + targetClass];
                totalLogLoss += sampleLogLoss;
                correctClassProbabilitySum += Math.exp(-sampleLogLoss);
                maximumSampleLogLoss = Math.max(maximumSampleLogLoss, sampleLogLoss);
                samples++;
            }
        }

        @Override
        public double value() {
            return samples == 0L ? Double.NaN : totalLogLoss / samples;
        }

        @Override
        public Map<String, Object> details() {
            double logLoss = value();
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "classification_log_loss");
            details.put("classes", Math.max(0, classes));
            details.put("samples", samples);
            details.put("logLoss", samples == 0L ? null : logLoss);
            details.put("crossEntropy", samples == 0L ? null : logLoss);
            details.put("negativeLogLikelihood", samples == 0L ? null : logLoss);
            details.put("perplexity", samples == 0L ? null : finiteExpOrNull(logLoss));
            details.put(
                    "meanCorrectClassProbability",
                    samples == 0L ? null : correctClassProbabilitySum / samples);
            details.put("maximumSampleLogLoss", samples == 0L ? null : maximumSampleLogLoss);
            details.put("input", "logits");
            return Collections.unmodifiableMap(details);
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        private static Double finiteExpOrNull(double value) {
            double result = Math.exp(value);
            return Double.isFinite(result) ? result : null;
        }
    }

    private abstract static class ClassificationStatsMetric implements TrainingMetric {
        private int classes = -1;
        private long[] truePositive = new long[0];
        private long[] falsePositive = new long[0];
        private long[] falseNegative = new long[0];

        @Override
        public void reset() {
            classes = -1;
            truePositive = new long[0];
            falsePositive = new long[0];
            falseNegative = new long[0];
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
                int predictedClass = TrainingMetricChecks.argmax(predictionData, row * currentClasses, currentClasses);
                int actualClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                if (predictedClass == actualClass) {
                    truePositive[actualClass]++;
                } else {
                    falsePositive[predictedClass]++;
                    falseNegative[actualClass]++;
                }
            }
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                truePositive = new long[classes];
                falsePositive = new long[classes];
                falseNegative = new long[classes];
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        protected double macroPrecision() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = truePositive[i] + falsePositive[i];
                total += denominator == 0 ? 0.0 : (double) truePositive[i] / denominator;
            }
            return total / classes;
        }

        protected double macroRecall() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = truePositive[i] + falseNegative[i];
                total += denominator == 0 ? 0.0 : (double) truePositive[i] / denominator;
            }
            return total / classes;
        }

        protected double macroF1() {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int i = 0; i < classes; i++) {
                long denominator = 2 * truePositive[i] + falsePositive[i] + falseNegative[i];
                total += denominator == 0 ? 0.0 : (double) (2 * truePositive[i]) / denominator;
            }
            return total / classes;
        }
    }

    private static final class PrecisionMetric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "precision";
        }

        @Override
        public double value() {
            return macroPrecision();
        }
    }

    private static final class RecallMetric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "recall";
        }

        @Override
        public double value() {
            return macroRecall();
        }
    }

    private static final class F1Metric extends ClassificationStatsMetric {
        @Override
        public String name() {
            return "f1";
        }

        @Override
        public double value() {
            return macroF1();
        }
    }

    private abstract static class ClassificationRankingMetric implements TrainingMetric {
        private int classes = -1;
        private List<List<TrainingMetricScore>> scoresByClass = List.of();

        @Override
        public void reset() {
            classes = -1;
            scoresByClass = List.of();
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
                int actualClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                int offset = row * currentClasses;
                for (int classIndex = 0; classIndex < currentClasses; classIndex++) {
                    scoresByClass.get(classIndex).add(new TrainingMetricScore(
                            predictionData[offset + classIndex],
                            classIndex == actualClass));
                }
            }
        }

        private void ensureClassStorage(int currentClasses) {
            if (classes < 0) {
                classes = currentClasses;
                List<List<TrainingMetricScore>> lists = new ArrayList<>(classes);
                for (int i = 0; i < classes; i++) {
                    lists.add(new ArrayList<>());
                }
                scoresByClass = lists;
                return;
            }
            if (classes != currentClasses) {
                throw new IllegalArgumentException(
                        name() + " expected " + classes + " classes but got " + currentClasses);
            }
        }

        protected double macroRocAuc() {
            return macroDefinedScore(true);
        }

        protected double macroAveragePrecision() {
            return macroDefinedScore(false);
        }

        private double macroDefinedScore(boolean rocAuc) {
            if (classes <= 0) {
                return Double.NaN;
            }
            double total = 0.0;
            int defined = 0;
            for (List<TrainingMetricScore> classScores : scoresByClass) {
                double score = rocAuc
                        ? TrainingMetricRanking.binaryRocAuc(classScores)
                        : TrainingMetricRanking.binaryAveragePrecision(classScores);
                if (Double.isFinite(score)) {
                    total += score;
                    defined++;
                }
            }
            return defined == 0 ? Double.NaN : total / defined;
        }
    }

    private static final class ClassificationMacroRocAucMetric extends ClassificationRankingMetric {
        @Override
        public String name() {
            return "classification_macro_roc_auc";
        }

        @Override
        public double value() {
            return macroRocAuc();
        }
    }

    private static final class ClassificationMacroAveragePrecisionMetric extends ClassificationRankingMetric {
        @Override
        public String name() {
            return "classification_macro_average_precision";
        }

        @Override
        public double value() {
            return macroAveragePrecision();
        }
    }

}
