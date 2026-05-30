package tech.kayys.gollek.ml.train;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.loss.TweedieNllLoss;

/** Count-distribution regression metrics. */
final class CountRegressionTrainingMetrics {
    private static final double DEFAULT_EPS = 1.0e-12;
    private static final double POWER_TOLERANCE = 1.0e-6;

    private CountRegressionTrainingMetrics() {
    }

    static Supplier<TrainingMetric> meanPoissonDeviance() {
        return meanPoissonDeviance(false);
    }

    static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput) {
        return meanPoissonDeviance(logInput, DEFAULT_EPS);
    }

    static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput, double eps) {
        double checkedEps = requirePositiveFinite(eps, "eps");
        return () -> new MeanPoissonDevianceMetric(logInput, checkedEps);
    }

    static Supplier<TrainingMetric> poissonDeviance() {
        return meanPoissonDeviance();
    }

    static Supplier<TrainingMetric> poissonLogRateDeviance() {
        return meanPoissonDeviance(true);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance() {
        return meanTweedieDeviance(TweedieNllLoss.DEFAULT_POWER);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power) {
        return meanTweedieDeviance(power, true);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput) {
        return meanTweedieDeviance(power, logInput, DEFAULT_EPS);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput, double eps) {
        double checkedPower = requireTweediePower(power);
        double checkedEps = requirePositiveFinite(eps, "eps");
        return () -> new MeanTweedieDevianceMetric(checkedPower, logInput, checkedEps);
    }

    static Supplier<TrainingMetric> tweedieDeviance() {
        return meanTweedieDeviance();
    }

    static Supplier<TrainingMetric> compoundPoissonGammaDeviance() {
        return meanTweedieDeviance();
    }

    private static final class MeanPoissonDevianceMetric implements DetailedTrainingMetric {
        private final boolean logInput;
        private final double eps;
        private double totalDeviance;
        private double totalTarget;
        private double totalMean;
        private long count;

        private MeanPoissonDevianceMetric(boolean logInput, double eps) {
            this.logInput = logInput;
            this.eps = eps;
        }

        @Override
        public String name() {
            return "mean_poisson_deviance";
        }

        @Override
        public void reset() {
            totalDeviance = 0.0;
            totalTarget = 0.0;
            totalMean = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = requireNonNegativeFinite(targetData[i], "target", i);
                double mean = meanPrediction(predictionData[i], i);
                totalDeviance += poissonDeviance(target, mean);
                totalTarget += target;
                totalMean += mean;
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalDeviance / count;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "mean_poisson_deviance");
            details.put("samples", count);
            details.put("logInput", logInput);
            details.put("eps", eps);
            details.put("meanTarget", count == 0 ? null : totalTarget / count);
            details.put("meanPrediction", count == 0 ? null : totalMean / count);
            return Collections.unmodifiableMap(details);
        }

        private double meanPrediction(double prediction, int index) {
            if (!Double.isFinite(prediction)) {
                throw new IllegalArgumentException(name()
                        + " predictions must be finite, got " + prediction + " at flat index " + index);
            }
            if (logInput) {
                double mean = Math.exp(prediction);
                if (!Double.isFinite(mean)) {
                    throw new IllegalArgumentException(name()
                            + " log-rate prediction overflows mean rate at flat index " + index
                            + ": " + prediction);
                }
                return Math.max(mean, eps);
            }
            if (prediction < 0.0) {
                throw new IllegalArgumentException(name()
                        + " raw mean predictions must be non-negative, got " + prediction
                        + " at flat index " + index);
            }
            return Math.max(prediction, eps);
        }
    }

    private static final class MeanTweedieDevianceMetric implements DetailedTrainingMetric {
        private final double power;
        private final boolean logInput;
        private final double eps;
        private double totalDeviance;
        private double totalTarget;
        private double totalMean;
        private long count;

        private MeanTweedieDevianceMetric(double power, boolean logInput, double eps) {
            this.power = power;
            this.logInput = logInput;
            this.eps = eps;
        }

        @Override
        public String name() {
            return "mean_tweedie_deviance";
        }

        @Override
        public void reset() {
            totalDeviance = 0.0;
            totalTarget = 0.0;
            totalMean = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            TrainingMetricChecks.requireSameShape(name(), predictions, targets);
            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int i = 0; i < predictionData.length; i++) {
                double target = requireNonNegativeFinite(targetData[i], "target", i);
                double mean = meanPrediction(predictionData[i], i);
                totalDeviance += tweedieDeviance(target, mean, i);
                totalTarget += target;
                totalMean += mean;
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : totalDeviance / count;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", "mean_tweedie_deviance");
            details.put("samples", count);
            details.put("power", power);
            details.put("logInput", logInput);
            details.put("eps", eps);
            details.put("meanTarget", count == 0 ? null : totalTarget / count);
            details.put("meanPrediction", count == 0 ? null : totalMean / count);
            return Collections.unmodifiableMap(details);
        }

        private double meanPrediction(double prediction, int index) {
            if (!Double.isFinite(prediction)) {
                throw new IllegalArgumentException(name()
                        + " predictions must be finite, got " + prediction + " at flat index " + index);
            }
            if (logInput) {
                double mean = Math.exp(prediction);
                if (!Double.isFinite(mean)) {
                    throw new IllegalArgumentException(name()
                            + " log-mean prediction overflows mean at flat index " + index
                            + ": " + prediction);
                }
                return Math.max(mean, eps);
            }
            if (prediction < 0.0) {
                throw new IllegalArgumentException(name()
                        + " raw mean predictions must be non-negative, got " + prediction
                        + " at flat index " + index);
            }
            return Math.max(prediction, eps);
        }

        private double tweedieDeviance(double target, double mean, int index) {
            if (isPoissonPower()) {
                return poissonDeviance(target, mean);
            }
            if (isGammaPower()) {
                if (target <= 0.0) {
                    throw new IllegalArgumentException(name()
                            + " with power=2 requires positive targets for Gamma deviance, got "
                            + target + " at flat index " + index);
                }
                return 2.0 * ((target - mean) / mean - Math.log(target / mean));
            }
            double oneMinusPower = 1.0 - power;
            double twoMinusPower = 2.0 - power;
            return 2.0 * (
                    Math.pow(target, twoMinusPower) / (oneMinusPower * twoMinusPower)
                            - target * Math.pow(mean, oneMinusPower) / oneMinusPower
                            + Math.pow(mean, twoMinusPower) / twoMinusPower);
        }

        private boolean isPoissonPower() {
            return Math.abs(power - 1.0) <= POWER_TOLERANCE;
        }

        private boolean isGammaPower() {
            return Math.abs(power - 2.0) <= POWER_TOLERANCE;
        }
    }

    private static double poissonDeviance(double target, double mean) {
        if (target == 0.0) {
            return 2.0 * mean;
        }
        return 2.0 * (target * Math.log(target / mean) - target + mean);
    }

    private static double requireNonNegativeFinite(double value, String label, int index) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(
                    "mean_poisson_deviance " + label + " values must be finite and non-negative, got "
                            + value + " at flat index " + index);
        }
        return value;
    }

    private static double requireTweediePower(double power) {
        if (!Double.isFinite(power) || power < 1.0 || power > 2.0) {
            throw new IllegalArgumentException("power must be finite and in [1, 2], got: " + power);
        }
        return power;
    }

    private static double requirePositiveFinite(double value, String label) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException(label + " must be finite and > 0, got: " + value);
        }
        return value;
    }
}
