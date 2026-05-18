package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Built-in regression metric implementations. */
final class RegressionTrainingMetrics {
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

    static Supplier<TrainingMetric> r2Score() {
        return R2ScoreMetric::new;
    }

    static Supplier<TrainingMetric> r2() {
        return r2Score();
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

}
