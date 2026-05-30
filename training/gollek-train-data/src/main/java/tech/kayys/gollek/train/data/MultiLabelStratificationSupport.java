package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

final class MultiLabelStratificationSupport {
    private MultiLabelStratificationSupport() {
    }

    static void requireSameSize(BinaryLabelMatrix labels, int sampleCount) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.rows() != sampleCount) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.rows() + " vs " + sampleCount);
        }
    }

    static int[] binaryLabelValues(BinaryLabelMatrix labels) {
        int[] values = new int[labels.values().length];
        for (int i = 0; i < labels.values().length; i++) {
            values[i] = labels.values()[i] >= 0.5f ? 1 : 0;
        }
        return values;
    }

    static int[] columnPositiveCounts(int[] flatLabels, int rows, int columns) {
        int[] counts = new int[columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                counts[column] += flatLabels[row * columns + column];
            }
        }
        return counts;
    }

    static List<Integer> multiLabelStratificationOrder(
            int[] flatLabels,
            int rows,
            int columns,
            int[] totalPositives,
            long seed) {
        List<Integer> order = new ArrayList<>(IntStream.range(0, rows).boxed().toList());
        Map<Integer, Integer> tieBreakers = new HashMap<>();
        Collections.shuffle(order, new Random(seed));
        for (int i = 0; i < order.size(); i++) {
            tieBreakers.put(order.get(i), i);
        }
        order.sort((left, right) -> {
            int rarity = Double.compare(
                    rowRarityScore(flatLabels, right, columns, totalPositives),
                    rowRarityScore(flatLabels, left, columns, totalPositives));
            if (rarity != 0) {
                return rarity;
            }
            int cardinality = Integer.compare(
                    rowCardinality(flatLabels, right, columns),
                    rowCardinality(flatLabels, left, columns));
            if (cardinality != 0) {
                return cardinality;
            }
            return Integer.compare(tieBreakers.get(left), tieBreakers.get(right));
        });
        return order;
    }

    static int smallestSplit(List<List<Integer>> splitIndices) {
        int smallest = 0;
        for (int split = 1; split < splitIndices.size(); split++) {
            if (splitIndices.get(split).size() < splitIndices.get(smallest).size()) {
                smallest = split;
            }
        }
        return smallest;
    }

    static void addRowCounts(int[] flatLabels, int row, int columns, int[] counts, int delta) {
        for (int column = 0; column < columns; column++) {
            counts[column] += flatLabels[row * columns + column] * delta;
        }
    }

    static int[] foldSizeTargets(int rows, int folds) {
        int[] targets = new int[folds];
        int base = rows / folds;
        int remainder = rows % folds;
        for (int fold = 0; fold < folds; fold++) {
            targets[fold] = base + (fold < remainder ? 1 : 0);
        }
        return targets;
    }

    static int[][] multiLabelFoldTargets(int[] totalPositives, int folds) {
        int[][] targets = new int[folds][totalPositives.length];
        for (int column = 0; column < totalPositives.length; column++) {
            int base = totalPositives[column] / folds;
            int remainder = totalPositives[column] % folds;
            for (int fold = 0; fold < folds; fold++) {
                targets[fold][column] = base + (fold < remainder ? 1 : 0);
            }
        }
        return targets;
    }

    static List<List<Integer>> emptyFoldBuckets(int folds, int[] foldTargets) {
        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>(foldTargets[fold]));
        }
        return validationFolds;
    }

    static int[][] copyCounts(int[][] source) {
        int[][] copy = new int[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }

    static void copyCounts(int[][] source, int[][] target) {
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, target[i], 0, source[i].length);
        }
    }

    static int stratifiedTrainCount(int groupSize, double trainFraction) {
        if (groupSize <= 0) {
            return 0;
        }
        if (groupSize == 1) {
            return 1;
        }
        int trainCount = (int) Math.round(groupSize * trainFraction);
        return Math.max(1, Math.min(groupSize - 1, trainCount));
    }

    static SplitCounts stratifiedThreeWayCounts(
            int groupSize,
            double trainFraction,
            double validationFraction) {
        if (groupSize <= 0) {
            return new SplitCounts(0, 0, 0);
        }
        if (groupSize == 1) {
            return new SplitCounts(1, 0, 0);
        }
        if (groupSize == 2) {
            return new SplitCounts(1, 1, 0);
        }
        return TensorDatasetSplits.threeWayCounts(groupSize, trainFraction, validationFraction);
    }

    private static double rowRarityScore(int[] flatLabels, int row, int columns, int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < columns; column++) {
            if (flatLabels[row * columns + column] == 1) {
                score += totalPositives[column] == 0 ? 0.0 : 1.0 / totalPositives[column];
            }
        }
        return score;
    }

    private static int rowCardinality(int[] flatLabels, int row, int columns) {
        int cardinality = 0;
        for (int column = 0; column < columns; column++) {
            cardinality += flatLabels[row * columns + column];
        }
        return cardinality;
    }

    record StratifiedGroupBucket(List<Integer> indices, int[] labelCounts) {
        int size() {
            return indices.size();
        }

        int labelMassSquared() {
            int total = 0;
            for (int count : labelCounts) {
                total += count * count;
            }
            return total;
        }
    }
}
