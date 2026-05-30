package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Multiclass probability calibration metrics computed from logits. */
final class ClassificationCalibrationTrainingMetrics {
    static final int DEFAULT_BINS = 10;

    private ClassificationCalibrationTrainingMetrics() {
    }

    static Supplier<TrainingMetric> classificationBrierScore() {
        return ClassificationBrierScoreMetric::new;
    }

    static Supplier<TrainingMetric> classificationExpectedCalibrationError() {
        return ClassificationExpectedCalibrationErrorMetric::new;
    }

    static Supplier<TrainingMetric> classificationExpectedCalibrationError(int bins) {
        int checkedBins = requireBins(bins);
        return () -> new ClassificationExpectedCalibrationErrorMetric(checkedBins);
    }

    private abstract static class ClassificationCalibrationMetric implements TrainingMetric {
        private final int bins;
        private int classes = -1;
        private long samples;
        private long correct;
        private double brierScoreSum;
        private double confidenceSum;
        private long[] binCount;
        private long[] binCorrect;
        private double[] binConfidenceSum;

        ClassificationCalibrationMetric() {
            this(DEFAULT_BINS);
        }

        ClassificationCalibrationMetric(int bins) {
            this.bins = requireBins(bins);
            this.binCount = new long[bins];
            this.binCorrect = new long[bins];
            this.binConfidenceSum = new double[bins];
        }

        @Override
        public void reset() {
            classes = -1;
            samples = 0L;
            correct = 0L;
            brierScoreSum = 0.0;
            confidenceSum = 0.0;
            binCount = new long[bins];
            binCorrect = new long[bins];
            binConfidenceSum = new double[bins];
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
                int targetClass = TrainingMetricChecks.targetClass(targets, row, batch, currentClasses);
                RowCalibration calibration = rowCalibration(predictionData, offset, currentClasses, targetClass);
                int bin = Math.min(bins - 1, (int) Math.floor(calibration.confidence() * bins));

                samples++;
                if (calibration.correct()) {
                    correct++;
                    binCorrect[bin]++;
                }
                brierScoreSum += calibration.brierScore();
                confidenceSum += calibration.confidence();
                binCount[bin]++;
                binConfidenceSum[bin] += calibration.confidence();
            }
        }

        protected double brierScore() {
            return samples == 0L ? Double.NaN : brierScoreSum / samples;
        }

        protected double expectedCalibrationError() {
            if (samples == 0L) {
                return Double.NaN;
            }
            double total = 0.0;
            for (int bin = 0; bin < bins; bin++) {
                if (binCount[bin] == 0L) {
                    continue;
                }
                total += binWeight(bin) * binGap(bin);
            }
            return total;
        }

        protected Map<String, Object> calibrationDetails() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "classification_calibration");
            details.put("mode", "top_label");
            details.put("bins", bins);
            details.put("classes", Math.max(0, classes));
            details.put("samples", samples);
            details.put("correct", correct);
            details.put("accuracy", samples == 0L ? null : correct / (double) samples);
            details.put("brierScore", samples == 0L ? null : brierScore());
            details.put("expectedCalibrationError", samples == 0L ? null : expectedCalibrationError());
            details.put("maximumCalibrationError", samples == 0L ? null : maximumCalibrationError());
            details.put("meanConfidence", samples == 0L ? null : confidenceSum / samples);
            details.put("binLowerBound", binLowerBounds());
            details.put("binUpperBound", binUpperBounds());
            details.put("binCount", binCounts());
            details.put("binWeight", binWeights());
            details.put("binConfidence", binConfidences());
            details.put("binAccuracy", binAccuracies());
            details.put("binGap", binGaps());
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

        private RowCalibration rowCalibration(float[] logits, int offset, int classes, int targetClass) {
            double maxLogit = logits[offset];
            for (int classIndex = 1; classIndex < classes; classIndex++) {
                maxLogit = Math.max(maxLogit, logits[offset + classIndex]);
            }

            double expSum = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                expSum += Math.exp(logits[offset + classIndex] - maxLogit);
            }

            int predictedClass = 0;
            double confidence = -1.0;
            double brierScore = 0.0;
            for (int classIndex = 0; classIndex < classes; classIndex++) {
                double probability = Math.exp(logits[offset + classIndex] - maxLogit) / expSum;
                if (probability > confidence) {
                    confidence = probability;
                    predictedClass = classIndex;
                }
                double target = classIndex == targetClass ? 1.0 : 0.0;
                double error = probability - target;
                brierScore += error * error;
            }
            return new RowCalibration(confidence, predictedClass == targetClass, brierScore);
        }

        private double maximumCalibrationError() {
            double maximum = 0.0;
            for (int bin = 0; bin < bins; bin++) {
                if (binCount[bin] > 0L) {
                    maximum = Math.max(maximum, binGap(bin));
                }
            }
            return maximum;
        }

        private List<Double> binLowerBounds() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 0; bin < bins; bin++) {
                values.add(bin / (double) bins);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> binUpperBounds() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 1; bin <= bins; bin++) {
                values.add(bin / (double) bins);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Long> binCounts() {
            List<Long> values = new ArrayList<>(bins);
            for (long count : binCount) {
                values.add(count);
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> binWeights() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 0; bin < bins; bin++) {
                values.add(samples == 0L ? 0.0 : binWeight(bin));
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> binConfidences() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 0; bin < bins; bin++) {
                values.add(binCount[bin] == 0L ? null : binConfidence(bin));
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> binAccuracies() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 0; bin < bins; bin++) {
                values.add(binCount[bin] == 0L ? null : binAccuracy(bin));
            }
            return Collections.unmodifiableList(values);
        }

        private List<Double> binGaps() {
            List<Double> values = new ArrayList<>(bins);
            for (int bin = 0; bin < bins; bin++) {
                values.add(binCount[bin] == 0L ? null : binGap(bin));
            }
            return Collections.unmodifiableList(values);
        }

        private double binWeight(int bin) {
            return binCount[bin] / (double) samples;
        }

        private double binConfidence(int bin) {
            return binConfidenceSum[bin] / binCount[bin];
        }

        private double binAccuracy(int bin) {
            return binCorrect[bin] / (double) binCount[bin];
        }

        private double binGap(int bin) {
            return Math.abs(binConfidence(bin) - binAccuracy(bin));
        }
    }

    private static final class ClassificationBrierScoreMetric extends ClassificationCalibrationMetric {
        @Override
        public String name() {
            return "classification_brier_score";
        }

        @Override
        public double value() {
            return brierScore();
        }
    }

    private static final class ClassificationExpectedCalibrationErrorMetric
            extends ClassificationCalibrationMetric implements DetailedTrainingMetric {
        ClassificationExpectedCalibrationErrorMetric() {
        }

        ClassificationExpectedCalibrationErrorMetric(int bins) {
            super(bins);
        }

        @Override
        public String name() {
            return "classification_expected_calibration_error";
        }

        @Override
        public double value() {
            return expectedCalibrationError();
        }

        @Override
        public Map<String, Object> details() {
            return calibrationDetails();
        }
    }

    private record RowCalibration(double confidence, boolean correct, double brierScore) {
    }

    private static int requireBins(int bins) {
        if (bins < 1) {
            throw new IllegalArgumentException("classification calibration bins must be >= 1, got " + bins);
        }
        return bins;
    }
}
