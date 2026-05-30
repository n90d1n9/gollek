package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

record TrainingMetricScore(float score, boolean positive) {
}

/** Ranking metric helpers shared by binary, multilabel, and multiclass metrics. */
final class TrainingMetricRanking {
    private TrainingMetricRanking() {
    }

    static double binaryRocAuc(List<TrainingMetricScore> rawScores) {
        List<TrainingMetricScore> values = new ArrayList<>(rawScores);
        long positives = values.stream().filter(TrainingMetricScore::positive).count();
        long negatives = values.size() - positives;
        if (positives == 0 || negatives == 0) {
            return Double.NaN;
        }

        values.sort(Comparator.comparingDouble(TrainingMetricScore::score));
        double positiveRankSum = 0.0;
        int index = 0;
        while (index < values.size()) {
            int groupEnd = index + 1;
            while (groupEnd < values.size()
                    && Float.compare(values.get(groupEnd).score(), values.get(index).score()) == 0) {
                groupEnd++;
            }
            double averageRank = ((index + 1.0) + groupEnd) / 2.0;
            for (int i = index; i < groupEnd; i++) {
                if (values.get(i).positive()) {
                    positiveRankSum += averageRank;
                }
            }
            index = groupEnd;
        }

        double positiveRankBaseline = positives * (positives + 1.0) / 2.0;
        return (positiveRankSum - positiveRankBaseline) / (positives * (double) negatives);
    }

    static double binaryAveragePrecision(List<TrainingMetricScore> rawScores) {
        List<TrainingMetricScore> values = new ArrayList<>(rawScores);
        long positives = values.stream().filter(TrainingMetricScore::positive).count();
        if (positives == 0) {
            return Double.NaN;
        }

        values.sort(Comparator.comparingDouble(TrainingMetricScore::score).reversed());
        long truePositive = 0;
        long falsePositive = 0;
        double ap = 0.0;
        int index = 0;
        while (index < values.size()) {
            int groupEnd = index + 1;
            long groupPositive = values.get(index).positive() ? 1 : 0;
            long groupNegative = values.get(index).positive() ? 0 : 1;
            while (groupEnd < values.size()
                    && Float.compare(values.get(groupEnd).score(), values.get(index).score()) == 0) {
                if (values.get(groupEnd).positive()) {
                    groupPositive++;
                } else {
                    groupNegative++;
                }
                groupEnd++;
            }

            truePositive += groupPositive;
            falsePositive += groupNegative;
            if (groupPositive > 0) {
                double precisionAtThreshold = truePositive / (double) (truePositive + falsePositive);
                double recallIncrease = groupPositive / (double) positives;
                ap += recallIncrease * precisionAtThreshold;
            }
            index = groupEnd;
        }
        return ap;
    }

    static TrainingMetricThreshold bestBinaryF1Threshold(List<TrainingMetricScore> rawScores) {
        if (rawScores.isEmpty()) {
            return TrainingMetricThreshold.empty();
        }

        List<TrainingMetricScore> values = new ArrayList<>(rawScores);
        values.sort(Comparator.comparingDouble(TrainingMetricScore::score).reversed());

        long positives = values.stream().filter(TrainingMetricScore::positive).count();
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
            long falseNegative = positives - truePositive;
            long trueNegative = negatives - falsePositive;
            TrainingMetricThreshold candidate = TrainingMetricThreshold.of(
                    threshold,
                    truePositive,
                    trueNegative,
                    falsePositive,
                    falseNegative);
            if (isBetterF1Threshold(candidate, best)) {
                best = candidate;
            }
            index = groupEnd;
        }
        return best;
    }

    private static boolean isBetterF1Threshold(TrainingMetricThreshold candidate, TrainingMetricThreshold best) {
        if (!best.defined()) {
            return true;
        }
        int f1Comparison = Double.compare(candidate.f1(), best.f1());
        if (f1Comparison != 0) {
            return f1Comparison > 0;
        }
        int precisionComparison = Double.compare(candidate.precision(), best.precision());
        if (precisionComparison != 0) {
            return precisionComparison > 0;
        }
        int recallComparison = Double.compare(candidate.recall(), best.recall());
        if (recallComparison != 0) {
            return recallComparison > 0;
        }
        return Float.compare(candidate.threshold(), best.threshold()) > 0;
    }

}

record TrainingMetricThreshold(
        boolean defined,
        float threshold,
        double f1,
        double precision,
        double recall,
        long truePositive,
        long trueNegative,
        long falsePositive,
        long falseNegative) {
    static TrainingMetricThreshold empty() {
        return new TrainingMetricThreshold(false, 0.0f, Double.NaN, Double.NaN, Double.NaN, 0L, 0L, 0L, 0L);
    }

    static TrainingMetricThreshold of(
            float threshold,
            long truePositive,
            long trueNegative,
            long falsePositive,
            long falseNegative) {
        long precisionDenominator = truePositive + falsePositive;
        long recallDenominator = truePositive + falseNegative;
        long f1Denominator = 2L * truePositive + falsePositive + falseNegative;
        double precision = precisionDenominator == 0L ? 0.0 : truePositive / (double) precisionDenominator;
        double recall = recallDenominator == 0L ? 0.0 : truePositive / (double) recallDenominator;
        double f1 = f1Denominator == 0L ? 0.0 : (2.0 * truePositive) / f1Denominator;
        return new TrainingMetricThreshold(
                true,
                threshold,
                f1,
                precision,
                recall,
                truePositive,
                trueNegative,
                falsePositive,
                falseNegative);
    }

    long total() {
        return truePositive + trueNegative + falsePositive + falseNegative;
    }

    long positives() {
        return truePositive + falseNegative;
    }

    long negatives() {
        return trueNegative + falsePositive;
    }
}
