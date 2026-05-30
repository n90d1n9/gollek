package tech.kayys.gollek.ml.train;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Built-in regression metric implementations. */
final class RegressionTrainingMetrics {
    private static final double EPSILON = 1.0e-12;

    private RegressionTrainingMetrics() {
    }

    static Supplier<TrainingMetric> meanAbsoluteError() {
        return MeanAbsoluteErrorMetric::new;
    }

    static Supplier<TrainingMetric> mae() {
        return meanAbsoluteError();
    }

    static Supplier<TrainingMetric> meanSquaredError() {
        return MeanSquaredErrorMetric::new;
    }

    static Supplier<TrainingMetric> mse() {
        return meanSquaredError();
    }

    static Supplier<TrainingMetric> rootMeanSquaredError() {
        return RootMeanSquaredErrorMetric::new;
    }

    static Supplier<TrainingMetric> rmse() {
        return rootMeanSquaredError();
    }

    static Supplier<TrainingMetric> meanSquaredLogError() {
        return MeanSquaredLogErrorMetric::new;
    }

    static Supplier<TrainingMetric> msle() {
        return meanSquaredLogError();
    }

    static Supplier<TrainingMetric> rootMeanSquaredLogError() {
        return RootMeanSquaredLogErrorMetric::new;
    }

    static Supplier<TrainingMetric> rmsle() {
        return rootMeanSquaredLogError();
    }

    static Supplier<TrainingMetric> medianAbsoluteError() {
        return MedianAbsoluteErrorMetric::new;
    }

    static Supplier<TrainingMetric> medae() {
        return medianAbsoluteError();
    }

    static Supplier<TrainingMetric> maxError() {
        return MaxErrorMetric::new;
    }

    static Supplier<TrainingMetric> pinballLoss(double quantile) {
        double checkedQuantile = requireQuantile(quantile);
        return () -> new PinballLossMetric(checkedQuantile);
    }

    static Supplier<TrainingMetric> meanPinballLoss(double quantile) {
        return pinballLoss(quantile);
    }

    static Supplier<TrainingMetric> predictionIntervalCoverage() {
        return PredictionIntervalCoverageMetric::new;
    }

    static Supplier<TrainingMetric> picp() {
        return predictionIntervalCoverage();
    }

    static Supplier<TrainingMetric> predictionIntervalMeanWidth() {
        return PredictionIntervalMeanWidthMetric::new;
    }

    static Supplier<TrainingMetric> predictionIntervalNormalizedMeanWidth() {
        return PredictionIntervalNormalizedMeanWidthMetric::new;
    }

    static Supplier<TrainingMetric> r2Score() {
        return R2ScoreMetric::new;
    }

    static Supplier<TrainingMetric> r2() {
        return r2Score();
    }

    static Supplier<TrainingMetric> meanAbsolutePercentageError() {
        return MeanAbsolutePercentageErrorMetric::new;
    }

    static Supplier<TrainingMetric> mape() {
        return meanAbsolutePercentageError();
    }

    static Supplier<TrainingMetric> symmetricMeanAbsolutePercentageError() {
        return SymmetricMeanAbsolutePercentageErrorMetric::new;
    }

    static Supplier<TrainingMetric> smape() {
        return symmetricMeanAbsolutePercentageError();
    }

    static Supplier<TrainingMetric> meanBiasError() {
        return MeanBiasErrorMetric::new;
    }

    static Supplier<TrainingMetric> mbe() {
        return meanBiasError();
    }

    static Supplier<TrainingMetric> explainedVariance() {
        return ExplainedVarianceMetric::new;
    }

    static Supplier<TrainingMetric> explainedVarianceScore() {
        return explainedVariance();
    }

    private static final class MeanAbsoluteErrorMetric implements TrainingMetric {
        private double totalError;
        private long count;

        @Override
        public String name() {
            return "mae";
        }

        @Override
        public void reset() {
            totalError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("mae", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                totalError += Math.abs(predictionData[i] - targetData[i]);
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalError / count;
        }
    }

    private static final class MeanSquaredErrorMetric implements TrainingMetric {
        private double totalSquaredError;
        private long count;

        @Override
        public String name() {
            return "mse";
        }

        @Override
        public void reset() {
            totalSquaredError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("mse", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double diff = predictionData[i] - targetData[i];
                totalSquaredError += diff * diff;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalSquaredError / count;
        }
    }

    private static final class MedianAbsoluteErrorMetric implements DetailedTrainingMetric {
        private final List<Double> absoluteErrors = new ArrayList<>();

        @Override
        public String name() {
            return "median_absolute_error";
        }

        @Override
        public void reset() {
            absoluteErrors.clear();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("median_absolute_error", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                absoluteErrors.add(Math.abs((double) predictionData[i] - targetData[i]));
            }
        }

        @Override
        public double value() {
            int samples = absoluteErrors.size();
            if (samples == 0) {
                return Double.NaN;
            }
            List<Double> sorted = new ArrayList<>(absoluteErrors);
            Collections.sort(sorted);
            int midpoint = samples / 2;
            if ((samples & 1) == 1) {
                return sorted.get(midpoint);
            }
            return (sorted.get(midpoint - 1) + sorted.get(midpoint)) / 2.0;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("median_absolute_error", absoluteErrors.size());
            details.put("exact", Boolean.TRUE);
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class MaxErrorMetric implements DetailedTrainingMetric {
        private long count;
        private double maxAbsoluteError;
        private double signedErrorAtMax;

        @Override
        public String name() {
            return "max_error";
        }

        @Override
        public void reset() {
            count = 0;
            maxAbsoluteError = 0.0;
            signedErrorAtMax = 0.0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("max_error", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double signedError = predictionData[i] - targetData[i];
                double absoluteError = Math.abs(signedError);
                if (count == 0 || absoluteError > maxAbsoluteError) {
                    maxAbsoluteError = absoluteError;
                    signedErrorAtMax = signedError;
                }
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : maxAbsoluteError;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("max_error", count);
            details.put("signedErrorAtMax", count == 0 ? null : signedErrorAtMax);
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class PinballLossMetric implements DetailedTrainingMetric {
        private final double quantile;
        private final String name;
        private double totalLoss;
        private long count;

        PinballLossMetric(double quantile) {
            this.quantile = requireQuantile(quantile);
            this.name = "pinball_loss_" + quantileSuffix(this.quantile);
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void reset() {
            totalLoss = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape(name, predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double residual = targetData[i] - predictionData[i];
                totalLoss += Math.max(quantile * residual, (quantile - 1.0) * residual);
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalLoss / count;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("pinball_loss", count);
            details.put("metricName", name);
            details.put("quantile", quantile);
            return Collections.unmodifiableMap(details);
        }
    }

    private abstract static class PredictionIntervalMetric implements DetailedTrainingMetric {
        private long count;
        private long covered;
        private long crossed;
        private double widthSum;
        private double targetMin = Double.POSITIVE_INFINITY;
        private double targetMax = Double.NEGATIVE_INFINITY;

        @Override
        public void reset() {
            count = 0;
            covered = 0;
            crossed = 0;
            widthSum = 0.0;
            targetMin = Double.POSITIVE_INFINITY;
            targetMax = Double.NEGATIVE_INFINITY;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            int samples = predictionIntervalSampleCount(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int sample = 0; sample < samples; sample++) {
                double lowerRaw = predictionData[sample * 2];
                double upperRaw = predictionData[sample * 2 + 1];
                double target = targetData[sample];
                if (lowerRaw > upperRaw) {
                    crossed++;
                }
                double lower = Math.min(lowerRaw, upperRaw);
                double upper = Math.max(lowerRaw, upperRaw);
                if (target >= lower && target <= upper) {
                    covered++;
                }
                widthSum += upper - lower;
                targetMin = Math.min(targetMin, target);
                targetMax = Math.max(targetMax, target);
                count++;
            }
        }

        protected double coverage() {
            return count == 0 ? Double.NaN : covered / (double) count;
        }

        protected double meanWidth() {
            return count == 0 ? Double.NaN : widthSum / count;
        }

        protected double normalizedMeanWidth() {
            if (count == 0) {
                return Double.NaN;
            }
            double range = targetRange();
            return range <= EPSILON ? Double.NaN : meanWidth() / range;
        }

        protected Map<String, Object> intervalDetails(String type) {
            Map<String, Object> details = baseDetails(type, count);
            details.put("coveredIntervals", covered);
            details.put("coverage", count == 0 ? null : coverage());
            details.put("meanWidth", count == 0 ? null : meanWidth());
            details.put("normalizedMeanWidth", count == 0 || targetRange() <= EPSILON ? null : normalizedMeanWidth());
            details.put("targetRange", count == 0 ? null : targetRange());
            details.put("crossedIntervals", crossed);
            details.put("crossedFraction", count == 0 ? null : crossed / (double) count);
            details.put("boundsReorderedForEvaluation", Boolean.TRUE);
            return Collections.unmodifiableMap(details);
        }

        private double targetRange() {
            return targetMax - targetMin;
        }
    }

    private static final class PredictionIntervalCoverageMetric extends PredictionIntervalMetric {
        @Override
        public String name() {
            return "prediction_interval_coverage";
        }

        @Override
        public double value() {
            return coverage();
        }

        @Override
        public Map<String, Object> details() {
            return intervalDetails("prediction_interval_coverage");
        }
    }

    private static final class PredictionIntervalMeanWidthMetric extends PredictionIntervalMetric {
        @Override
        public String name() {
            return "prediction_interval_mean_width";
        }

        @Override
        public double value() {
            return meanWidth();
        }

        @Override
        public Map<String, Object> details() {
            return intervalDetails("prediction_interval_mean_width");
        }
    }

    private static final class PredictionIntervalNormalizedMeanWidthMetric extends PredictionIntervalMetric {
        @Override
        public String name() {
            return "prediction_interval_normalized_mean_width";
        }

        @Override
        public double value() {
            return normalizedMeanWidth();
        }

        @Override
        public Map<String, Object> details() {
            return intervalDetails("prediction_interval_normalized_mean_width");
        }
    }

    private static final class RootMeanSquaredErrorMetric implements TrainingMetric {
        private double totalSquaredError;
        private long count;

        @Override
        public String name() {
            return "rmse";
        }

        @Override
        public void reset() {
            totalSquaredError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("rmse", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double diff = predictionData[i] - targetData[i];
                totalSquaredError += diff * diff;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : Math.sqrt(totalSquaredError / count);
        }
    }

    private abstract static class SquaredLogErrorMetric implements DetailedTrainingMetric {
        private double totalSquaredLogError;
        private long count;

        @Override
        public void reset() {
            totalSquaredLogError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double prediction = predictionData[i];
                double target = targetData[i];
                if (prediction < 0.0 || target < 0.0) {
                    throw new IllegalArgumentException(name()
                            + " expects non-negative predictions and targets for log1p regression, got prediction "
                            + prediction + " and target " + target + " at flat index " + i);
                }
                double diff = Math.log1p(prediction) - Math.log1p(target);
                totalSquaredLogError += diff * diff;
                count++;
            }
        }

        protected double meanSquaredLogError() {
            return count == 0 ? Double.NaN : totalSquaredLogError / count;
        }

        protected Map<String, Object> logErrorDetails(String type) {
            Map<String, Object> details = baseDetails(type, count);
            details.put("logTransform", "log1p");
            details.put("inputDomain", "predictions >= 0 and targets >= 0");
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class MeanSquaredLogErrorMetric extends SquaredLogErrorMetric {
        @Override
        public String name() {
            return "msle";
        }

        @Override
        public double value() {
            return meanSquaredLogError();
        }

        @Override
        public Map<String, Object> details() {
            return logErrorDetails("msle");
        }
    }

    private static final class RootMeanSquaredLogErrorMetric extends SquaredLogErrorMetric {
        @Override
        public String name() {
            return "rmsle";
        }

        @Override
        public double value() {
            double meanSquaredLogError = meanSquaredLogError();
            return Double.isNaN(meanSquaredLogError) ? Double.NaN : Math.sqrt(meanSquaredLogError);
        }

        @Override
        public Map<String, Object> details() {
            return logErrorDetails("rmsle");
        }
    }

    private static final class MeanAbsolutePercentageErrorMetric implements DetailedTrainingMetric {
        private double totalRatio;
        private long count;
        private long skippedZeroTargets;

        @Override
        public String name() {
            return "mape";
        }

        @Override
        public void reset() {
            totalRatio = 0.0;
            count = 0;
            skippedZeroTargets = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("mape", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = targetData[i];
                double denominator = Math.abs(target);
                if (denominator <= EPSILON) {
                    skippedZeroTargets++;
                    continue;
                }
                totalRatio += Math.abs(predictionData[i] - target) / denominator;
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalRatio / count;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("mape", count);
            details.put("scale", "fraction");
            details.put("skippedZeroTargets", skippedZeroTargets);
            details.put("zeroTargetEpsilon", EPSILON);
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class SymmetricMeanAbsolutePercentageErrorMetric implements DetailedTrainingMetric {
        private double totalRatio;
        private long count;
        private long zeroDenominatorCount;

        @Override
        public String name() {
            return "smape";
        }

        @Override
        public void reset() {
            totalRatio = 0.0;
            count = 0;
            zeroDenominatorCount = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("smape", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double prediction = predictionData[i];
                double target = targetData[i];
                double denominator = (Math.abs(prediction) + Math.abs(target)) / 2.0;
                if (denominator <= EPSILON) {
                    zeroDenominatorCount++;
                } else {
                    totalRatio += Math.abs(prediction - target) / denominator;
                }
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalRatio / count;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("smape", count);
            details.put("scale", "fraction");
            details.put("zeroDenominatorCount", zeroDenominatorCount);
            details.put("zeroDenominatorContribution", 0.0);
            details.put("zeroDenominatorEpsilon", EPSILON);
            return Collections.unmodifiableMap(details);
        }
    }

    private static final class MeanBiasErrorMetric implements TrainingMetric {
        private double totalError;
        private long count;

        @Override
        public String name() {
            return "mbe";
        }

        @Override
        public void reset() {
            totalError = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("mbe", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                totalError += predictionData[i] - targetData[i];
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalError / count;
        }
    }

    private static final class R2ScoreMetric implements TrainingMetric {
        private double sumSquaredError;
        private double targetSum;
        private double targetSquaredSum;
        private long count;

        @Override
        public String name() {
            return "r2";
        }

        @Override
        public void reset() {
            sumSquaredError = 0.0;
            targetSum = 0.0;
            targetSquaredSum = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("r2", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = targetData[i];
                double diff = predictionData[i] - target;
                sumSquaredError += diff * diff;
                targetSum += target;
                targetSquaredSum += target * target;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            if (count == 0) {
                return Double.NaN;
            }
            double totalVariance = targetSquaredSum - (targetSum * targetSum / count);
            if (Math.abs(totalVariance) < 1e-12) {
                return Math.abs(sumSquaredError) < 1e-12 ? 1.0 : 0.0;
            }
            return 1.0 - (sumSquaredError / totalVariance);
        }
    }

    private static final class ExplainedVarianceMetric implements DetailedTrainingMetric {
        private double residualSum;
        private double residualSquaredSum;
        private double targetSum;
        private double targetSquaredSum;
        private long count;

        @Override
        public String name() {
            return "explained_variance";
        }

        @Override
        public void reset() {
            residualSum = 0.0;
            residualSquaredSum = 0.0;
            targetSum = 0.0;
            targetSquaredSum = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape("explained_variance", predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = targetData[i];
                double residual = target - predictionData[i];
                residualSum += residual;
                residualSquaredSum += residual * residual;
                targetSum += target;
                targetSquaredSum += target * target;
            }
            count += predictionData.length;
        }

        @Override
        public double value() {
            if (count == 0) {
                return Double.NaN;
            }
            double targetVariance = targetVariance();
            double residualVariance = residualVariance();
            if (targetVariance <= EPSILON) {
                return residualVariance <= EPSILON ? 1.0 : 0.0;
            }
            return 1.0 - (residualVariance / targetVariance);
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("explained_variance", count);
            details.put("targetVariance", count == 0 ? null : targetVariance());
            details.put("residualVariance", count == 0 ? null : residualVariance());
            details.put("varianceEpsilon", EPSILON);
            return Collections.unmodifiableMap(details);
        }

        private double targetVariance() {
            return variance(targetSquaredSum, targetSum, count);
        }

        private double residualVariance() {
            return variance(residualSquaredSum, residualSum, count);
        }
    }

    private static Map<String, Object> baseDetails(String type, long samples) {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("type", type);
        details.put("samples", samples);
        return details;
    }

    private static double requireQuantile(double quantile) {
        if (!Double.isFinite(quantile) || quantile <= 0.0 || quantile >= 1.0) {
            throw new IllegalArgumentException("quantile must be finite and between 0 and 1 exclusive, got "
                    + quantile);
        }
        return quantile;
    }

    private static String quantileSuffix(double quantile) {
        BigDecimal percentage = BigDecimal.valueOf(quantile)
                .multiply(BigDecimal.valueOf(100.0))
                .stripTrailingZeros();
        return "q" + percentage.toPlainString().replace('.', '_');
    }

    private static int predictionIntervalSampleCount(
            String metricName,
            GradTensor predictions,
            GradTensor targets) {
        long[] predictionShape = predictions.shape();
        if (predictionShape.length == 0 || predictionShape[predictionShape.length - 1] != 2L) {
            throw new IllegalArgumentException(
                    metricName + " expects predictions shaped [..., 2] for lower/upper bounds, got "
                            + java.util.Arrays.toString(predictionShape));
        }
        long boundValues = predictions.numel();
        long samples = boundValues / 2L;
        if (targets.numel() != samples) {
            throw new IllegalArgumentException(
                    metricName + " expects target element count " + samples
                            + " for prediction interval bounds, got " + targets.numel());
        }
        return Math.toIntExact(samples);
    }

    private static double variance(double squaredSum, double sum, long count) {
        if (count == 0) {
            return Double.NaN;
        }
        return Math.max(0.0, squaredSum - (sum * sum / count)) / count;
    }
}
