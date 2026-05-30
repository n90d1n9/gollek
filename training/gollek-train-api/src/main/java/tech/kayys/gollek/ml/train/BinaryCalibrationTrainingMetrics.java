package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Binary probability calibration metrics computed from logits. */
final class BinaryCalibrationTrainingMetrics {
    static final int DEFAULT_BINS = 10;

    private BinaryCalibrationTrainingMetrics() {
    }

    static Supplier<TrainingMetric> binaryBrierScore() {
        return BinaryBrierScoreMetric::new;
    }

    static Supplier<TrainingMetric> binaryLogLoss() {
        return BinaryLogLossMetric::new;
    }

    static Supplier<TrainingMetric> binaryCrossEntropy() {
        return binaryLogLoss();
    }

    static Supplier<TrainingMetric> binaryExpectedCalibrationError() {
        return BinaryExpectedCalibrationErrorMetric::new;
    }

    static Supplier<TrainingMetric> binaryExpectedCalibrationError(int bins) {
        int checkedBins = requireBins(bins);
        return () -> new BinaryExpectedCalibrationErrorMetric(checkedBins);
    }

    private abstract static class BinaryCalibrationMetric implements TrainingMetric {
        private final int bins;
        private long samples;
        private long positives;
        private double squaredErrorSum;
        private double confidenceSum;
        private long[] binCount;
        private long[] binPositive;
        private double[] binConfidenceSum;

        BinaryCalibrationMetric() {
            this(DEFAULT_BINS);
        }

        BinaryCalibrationMetric(int bins) {
            this.bins = requireBins(bins);
            this.binCount = new long[bins];
            this.binPositive = new long[bins];
            this.binConfidenceSum = new double[bins];
        }

        @Override
        public void reset() {
            samples = 0L;
            positives = 0L;
            squaredErrorSum = 0.0;
            confidenceSum = 0.0;
            binCount = new long[bins];
            binPositive = new long[bins];
            binConfidenceSum = new double[bins];
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double probability = sigmoid(predictionData[i]);
                boolean positive = TrainingMetricChecks.binaryTarget(targetData[i]);
                double target = positive ? 1.0 : 0.0;
                double error = probability - target;
                int bin = Math.min(bins - 1, (int) Math.floor(probability * bins));
                samples++;
                if (positive) {
                    positives++;
                    binPositive[bin]++;
                }
                squaredErrorSum += error * error;
                confidenceSum += probability;
                binCount[bin]++;
                binConfidenceSum[bin] += probability;
            }
        }

        protected double brierScore() {
            return samples == 0L ? Double.NaN : squaredErrorSum / samples;
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
            details.put("type", "binary_calibration");
            details.put("bins", bins);
            details.put("samples", samples);
            details.put("positives", positives);
            details.put("negatives", samples - positives);
            details.put("brierScore", samples == 0L ? null : brierScore());
            details.put("expectedCalibrationError", samples == 0L ? null : expectedCalibrationError());
            details.put("maximumCalibrationError", samples == 0L ? null : maximumCalibrationError());
            details.put("meanConfidence", samples == 0L ? null : confidenceSum / samples);
            details.put("empiricalPositiveRate", samples == 0L ? null : positives / (double) samples);
            details.put("binLowerBound", binLowerBounds());
            details.put("binUpperBound", binUpperBounds());
            details.put("binCount", binCounts());
            details.put("binWeight", binWeights());
            details.put("binConfidence", binConfidences());
            details.put("binAccuracy", binAccuracies());
            details.put("binGap", binGaps());
            return Collections.unmodifiableMap(details);
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
            return binPositive[bin] / (double) binCount[bin];
        }

        private double binGap(int bin) {
            return Math.abs(binConfidence(bin) - binAccuracy(bin));
        }

        private static double sigmoid(float logit) {
            if (logit >= 0.0f) {
                double exp = Math.exp(-logit);
                return 1.0 / (1.0 + exp);
            }
            double exp = Math.exp(logit);
            return exp / (1.0 + exp);
        }
    }

    private static final class BinaryBrierScoreMetric extends BinaryCalibrationMetric {
        @Override
        public String name() {
            return "binary_brier_score";
        }

        @Override
        public double value() {
            return brierScore();
        }
    }

    private static final class BinaryLogLossMetric implements DetailedTrainingMetric {
        private long samples;
        private long positives;
        private double totalLogLoss;
        private double positiveProbabilitySum;
        private double correctLabelProbabilitySum;
        private double maximumSampleLogLoss;

        @Override
        public String name() {
            return "binary_log_loss";
        }

        @Override
        public void reset() {
            samples = 0L;
            positives = 0L;
            totalLogLoss = 0.0;
            positiveProbabilitySum = 0.0;
            correctLabelProbabilitySum = 0.0;
            maximumSampleLogLoss = 0.0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameElementCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                float logit = requireFiniteLogit(predictionData[i], i);
                boolean positive = TrainingMetricChecks.binaryTarget(targetData[i]);
                double probability = sigmoid(logit);
                double sampleLogLoss = binaryCrossEntropyWithLogits(logit, positive);
                samples++;
                if (positive) {
                    positives++;
                }
                totalLogLoss += sampleLogLoss;
                positiveProbabilitySum += probability;
                correctLabelProbabilitySum += positive ? probability : 1.0 - probability;
                maximumSampleLogLoss = Math.max(maximumSampleLogLoss, sampleLogLoss);
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
            details.put("type", "binary_log_loss");
            details.put("samples", samples);
            details.put("positives", positives);
            details.put("negatives", samples - positives);
            details.put("logLoss", samples == 0L ? null : logLoss);
            details.put("binaryCrossEntropy", samples == 0L ? null : logLoss);
            details.put("negativeLogLikelihood", samples == 0L ? null : logLoss);
            details.put("perplexity", samples == 0L ? null : finiteExpOrNull(logLoss));
            details.put("meanPositiveProbability", samples == 0L ? null : positiveProbabilitySum / samples);
            details.put("empiricalPositiveRate", samples == 0L ? null : positives / (double) samples);
            details.put(
                    "meanCorrectLabelProbability",
                    samples == 0L ? null : correctLabelProbabilitySum / samples);
            details.put("maximumSampleLogLoss", samples == 0L ? null : maximumSampleLogLoss);
            details.put("input", "logits");
            details.put("targetEncoding", "binary_0_1");
            return Collections.unmodifiableMap(details);
        }

        private static double binaryCrossEntropyWithLogits(float logit, boolean positive) {
            double value = logit;
            double target = positive ? 1.0 : 0.0;
            return Math.max(value, 0.0) - value * target + Math.log1p(Math.exp(-Math.abs(value)));
        }

        private static double sigmoid(float logit) {
            if (logit >= 0.0f) {
                double exp = Math.exp(-logit);
                return 1.0 / (1.0 + exp);
            }
            double exp = Math.exp(logit);
            return exp / (1.0 + exp);
        }

        private static float requireFiniteLogit(float logit, int index) {
            if (!Float.isFinite(logit)) {
                throw new IllegalArgumentException(
                        "binary_log_loss expects finite logits, got " + logit + " at index " + index);
            }
            return logit;
        }

        private static Double finiteExpOrNull(double value) {
            double result = Math.exp(value);
            return Double.isFinite(result) ? result : null;
        }
    }

    private static final class BinaryExpectedCalibrationErrorMetric extends BinaryCalibrationMetric
            implements DetailedTrainingMetric {
        BinaryExpectedCalibrationErrorMetric() {
        }

        BinaryExpectedCalibrationErrorMetric(int bins) {
            super(bins);
        }

        @Override
        public String name() {
            return "binary_expected_calibration_error";
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

    private static int requireBins(int bins) {
        if (bins < 1) {
            throw new IllegalArgumentException("binary calibration bins must be >= 1, got " + bins);
        }
        return bins;
    }
}
