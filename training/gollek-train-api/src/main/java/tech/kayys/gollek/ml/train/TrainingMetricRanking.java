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

}
