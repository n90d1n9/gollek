package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Sample-wise multilabel ranking metrics computed directly from prediction scores. */
final class MultiLabelRankingTrainingMetrics {
    private MultiLabelRankingTrainingMetrics() {
    }

    static Supplier<TrainingMetric> multiLabelLabelRankingAveragePrecision() {
        return MultiLabelLabelRankingAveragePrecisionMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelRankingLoss() {
        return MultiLabelRankingLossMetric::new;
    }

    static Supplier<TrainingMetric> multiLabelCoverageError() {
        return MultiLabelCoverageErrorMetric::new;
    }

    private abstract static class MultiLabelSampleRankingMetric implements TrainingMetric {
        private int labels = -1;
        private long samples;
        private double labelRankingAveragePrecisionSum;
        private double rankingLossSum;
        private double coverageErrorSum;

        @Override
        public void reset() {
            labels = -1;
            samples = 0L;
            labelRankingAveragePrecisionSum = 0.0;
            rankingLossSum = 0.0;
            coverageErrorSum = 0.0;
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
                RowRanking rowRanking = rowRanking(predictionData, targetData, row * currentLabels, currentLabels);
                labelRankingAveragePrecisionSum += rowRanking.labelRankingAveragePrecision();
                rankingLossSum += rowRanking.rankingLoss();
                coverageErrorSum += rowRanking.coverageError();
                samples++;
            }
        }

        protected double labelRankingAveragePrecision() {
            return samples == 0L ? Double.NaN : labelRankingAveragePrecisionSum / samples;
        }

        protected double rankingLoss() {
            return samples == 0L ? Double.NaN : rankingLossSum / samples;
        }

        protected double coverageError() {
            return samples == 0L ? Double.NaN : coverageErrorSum / samples;
        }

        private void ensureLabelStorage(int currentLabels) {
            if (labels < 0) {
                labels = currentLabels;
                return;
            }
            if (labels != currentLabels) {
                throw new IllegalArgumentException(
                        name() + " expected " + labels + " labels per sample but got " + currentLabels);
            }
        }

        private static RowRanking rowRanking(float[] scores, float[] targets, int offset, int labels) {
            boolean[] positive = new boolean[labels];
            int positives = 0;
            for (int label = 0; label < labels; label++) {
                boolean actualPositive = TrainingMetricChecks.binaryTarget(targets[offset + label]);
                positive[label] = actualPositive;
                if (actualPositive) {
                    positives++;
                }
            }

            if (positives == 0) {
                return new RowRanking(1.0, 0.0, 0.0);
            }
            int negatives = labels - positives;
            double rankingLoss = negatives == 0 ? 0.0 : rankingLoss(scores, offset, positive, positives, negatives);
            return new RowRanking(
                    labelRankingAveragePrecision(scores, offset, positive, positives),
                    rankingLoss,
                    coverageError(scores, offset, positive));
        }

        private static double labelRankingAveragePrecision(
                float[] scores,
                int offset,
                boolean[] positive,
                int positives) {
            double total = 0.0;
            for (int label = 0; label < positive.length; label++) {
                if (!positive[label]) {
                    continue;
                }
                float positiveScore = scores[offset + label];
                int labelsAtOrAbove = 0;
                int positivesAtOrAbove = 0;
                for (int candidate = 0; candidate < positive.length; candidate++) {
                    if (scores[offset + candidate] >= positiveScore) {
                        labelsAtOrAbove++;
                        if (positive[candidate]) {
                            positivesAtOrAbove++;
                        }
                    }
                }
                total += positivesAtOrAbove / (double) labelsAtOrAbove;
            }
            return total / positives;
        }

        private static double rankingLoss(
                float[] scores,
                int offset,
                boolean[] positive,
                int positives,
                int negatives) {
            double incorrectPairs = 0.0;
            for (int positiveLabel = 0; positiveLabel < positive.length; positiveLabel++) {
                if (!positive[positiveLabel]) {
                    continue;
                }
                float positiveScore = scores[offset + positiveLabel];
                for (int negativeLabel = 0; negativeLabel < positive.length; negativeLabel++) {
                    if (positive[negativeLabel]) {
                        continue;
                    }
                    int comparison = Float.compare(positiveScore, scores[offset + negativeLabel]);
                    if (comparison < 0) {
                        incorrectPairs += 1.0;
                    } else if (comparison == 0) {
                        incorrectPairs += 0.5;
                    }
                }
            }
            return incorrectPairs / (positives * (double) negatives);
        }

        private static double coverageError(float[] scores, int offset, boolean[] positive) {
            float lowestPositiveScore = Float.POSITIVE_INFINITY;
            for (int label = 0; label < positive.length; label++) {
                if (positive[label]) {
                    lowestPositiveScore = Math.min(lowestPositiveScore, scores[offset + label]);
                }
            }
            int labelsNeeded = 0;
            for (int label = 0; label < positive.length; label++) {
                if (scores[offset + label] >= lowestPositiveScore) {
                    labelsNeeded++;
                }
            }
            return labelsNeeded;
        }

        @Override
        public String toString() {
            return name() + "[labels=" + labels + ", samples=" + samples
                    + ", lrap=" + labelRankingAveragePrecision()
                    + ", rankingLoss=" + rankingLoss()
                    + ", coverageError=" + coverageError()
                    + "]";
        }
    }

    private static final class MultiLabelLabelRankingAveragePrecisionMetric
            extends MultiLabelSampleRankingMetric {
        @Override
        public String name() {
            return "multilabel_label_ranking_average_precision";
        }

        @Override
        public double value() {
            return labelRankingAveragePrecision();
        }
    }

    private static final class MultiLabelRankingLossMetric extends MultiLabelSampleRankingMetric {
        @Override
        public String name() {
            return "multilabel_ranking_loss";
        }

        @Override
        public double value() {
            return rankingLoss();
        }
    }

    private static final class MultiLabelCoverageErrorMetric extends MultiLabelSampleRankingMetric {
        @Override
        public String name() {
            return "multilabel_coverage_error";
        }

        @Override
        public double value() {
            return coverageError();
        }
    }

    private record RowRanking(
            double labelRankingAveragePrecision,
            double rankingLoss,
            double coverageError) {
    }
}
