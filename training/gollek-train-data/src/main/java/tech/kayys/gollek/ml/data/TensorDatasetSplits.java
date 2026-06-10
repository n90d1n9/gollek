package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

final class TensorDatasetSplits {
    private TensorDatasetSplits() {
    }

    static DataLoader.TensorDatasetSplit randomSplit(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireTrainFraction(trainFraction);
        if (dataset.size() == 0) {
            throw new IllegalArgumentException("dataset must contain at least one sample");
        }

        List<Integer> indices = new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList());
        Collections.shuffle(indices, new Random(seed));
        int trainSize = Math.max(1, Math.min(dataset.size() - 1, (int) Math.round(dataset.size() * trainFraction)));
        return splitByIndexSlices(dataset, indices.subList(0, trainSize), indices.subList(trainSize, indices.size()));
    }

    static DataLoader.TensorDatasetSplit stratifiedSplit(
            DataLoader.TensorDataset dataset,
            int[] labels,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        requireTrainFraction(trainFraction);
        if (labels.length != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.length + " vs " + dataset.size());
        }

        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            groups.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        }

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, entry.getKey())));
            int trainCount = stratifiedTrainCount(group.size(), trainFraction);
            trainIndices.addAll(group.subList(0, trainCount));
            validationIndices.addAll(group.subList(trainCount, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x6A09E667F3BCC909L));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xBB67AE8584CAA73BL));
        return splitByIndexSlices(dataset, trainIndices, validationIndices);
    }

    static DataLoader.TensorDatasetSplit multiLabelStratifiedSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labels, "labels must not be null");
        requireTrainFraction(trainFraction);
        if (labels.rows() != dataset.size()) {
            throw new IllegalArgumentException(
                    "labels and dataset must have same size, got: " + labels.rows() + " vs " + dataset.size());
        }

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        int[] trainTargets = new int[columns];
        for (int column = 0; column < columns; column++) {
            trainTargets[column] = stratifiedTrainCount(totalPositives[column], trainFraction);
        }

        int trainTargetSize = stratifiedTrainCount(rows, trainFraction);
        int validationTargetSize = rows - trainTargetSize;
        int[] trainCounts = new int[columns];
        List<Integer> order = multiLabelStratificationOrder(flatLabels, rows, columns, totalPositives, seed);
        List<Integer> trainIndices = new ArrayList<>(trainTargetSize);
        List<Integer> validationIndices = new ArrayList<>(validationTargetSize);

        for (int row : order) {
            if (trainIndices.size() >= trainTargetSize) {
                validationIndices.add(row);
                continue;
            }
            if (validationIndices.size() >= validationTargetSize) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
                continue;
            }

            double trainPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size() + 1,
                    trainTargetSize, true);
            double validationPenalty = multiLabelSplitPenalty(
                    flatLabels, row, columns, totalPositives, trainCounts, trainTargets, trainIndices.size(),
                    trainTargetSize, false);
            if (trainPenalty <= validationPenalty) {
                trainIndices.add(row);
                addRowCounts(flatLabels, row, columns, trainCounts, 1);
            } else {
                validationIndices.add(row);
            }
        }

        improveMultiLabelAssignment(
                flatLabels, columns, totalPositives, trainTargets, trainIndices, validationIndices, trainCounts);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x3C6EF372FE94F82AL));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xA54FF53A5F1D36F1L));
        return splitByIndexSlices(dataset, trainIndices, validationIndices);
    }

    private static int[] binaryLabelValues(BinaryLabelMatrix labels) {
        int[] values = new int[labels.values().length];
        for (int i = 0; i < labels.values().length; i++) {
            values[i] = labels.values()[i] >= 0.5f ? 1 : 0;
        }
        return values;
    }

    private static int[] columnPositiveCounts(int[] flatLabels, int rows, int columns) {
        int[] counts = new int[columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                counts[column] += flatLabels[row * columns + column];
            }
        }
        return counts;
    }

    private static List<Integer> multiLabelStratificationOrder(
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

    private static double multiLabelSplitPenalty(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[] trainCounts,
            int[] trainTargets,
            int trainSize,
            int trainTargetSize,
            boolean assignToTrain) {
        double penalty = Math.pow(trainSize - trainTargetSize, 2) * columns;
        for (int column = 0; column < columns; column++) {
            int after = trainCounts[column];
            if (assignToTrain) {
                after += flatLabels[row * columns + column];
            }
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            penalty += Math.pow(after - trainTargets[column], 2) * labelWeight;
        }
        return penalty;
    }

    private static void addRowCounts(int[] flatLabels, int row, int columns, int[] counts, int delta) {
        for (int column = 0; column < columns; column++) {
            counts[column] += flatLabels[row * columns + column] * delta;
        }
    }

    private static void improveMultiLabelAssignment(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[] trainTargets,
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            int[] trainCounts) {
        double bestScore = multiLabelAssignmentScore(trainCounts, trainTargets, totalPositives);
        boolean improved;
        do {
            improved = false;
            int bestTrainPosition = -1;
            int bestValidationPosition = -1;
            int[] bestCounts = null;

            for (int trainPosition = 0; trainPosition < trainIndices.size(); trainPosition++) {
                int trainRow = trainIndices.get(trainPosition);
                for (int validationPosition = 0; validationPosition < validationIndices.size(); validationPosition++) {
                    int validationRow = validationIndices.get(validationPosition);
                    int[] candidateCounts = swappedTrainCounts(
                            flatLabels, columns, trainCounts, trainRow, validationRow);
                    double candidateScore = multiLabelAssignmentScore(candidateCounts, trainTargets, totalPositives);
                    if (candidateScore + 1e-9 < bestScore) {
                        bestScore = candidateScore;
                        bestTrainPosition = trainPosition;
                        bestValidationPosition = validationPosition;
                        bestCounts = candidateCounts;
                        improved = true;
                    }
                }
            }

            if (improved) {
                int trainRow = trainIndices.get(bestTrainPosition);
                trainIndices.set(bestTrainPosition, validationIndices.get(bestValidationPosition));
                validationIndices.set(bestValidationPosition, trainRow);
                System.arraycopy(bestCounts, 0, trainCounts, 0, trainCounts.length);
            }
        } while (improved);
    }

    private static int[] swappedTrainCounts(
            int[] flatLabels,
            int columns,
            int[] trainCounts,
            int trainRow,
            int validationRow) {
        int[] candidate = trainCounts.clone();
        addRowCounts(flatLabels, trainRow, columns, candidate, -1);
        addRowCounts(flatLabels, validationRow, columns, candidate, 1);
        return candidate;
    }

    private static double multiLabelAssignmentScore(
            int[] trainCounts,
            int[] trainTargets,
            int[] totalPositives) {
        double score = 0.0;
        for (int column = 0; column < trainTargets.length; column++) {
            double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
            score += Math.pow(trainCounts[column] - trainTargets[column], 2) * labelWeight;
        }
        return score;
    }

    private static int stratifiedTrainCount(int groupSize, double trainFraction) {
        if (groupSize <= 0) {
            return 0;
        }
        if (groupSize == 1) {
            return 1;
        }
        int trainCount = (int) Math.round(groupSize * trainFraction);
        return Math.max(1, Math.min(groupSize - 1, trainCount));
    }

    private static void rebalanceNonEmptySplit(List<Integer> trainIndices, List<Integer> validationIndices) {
        if (trainIndices.isEmpty() && !validationIndices.isEmpty()) {
            trainIndices.add(validationIndices.remove(validationIndices.size() - 1));
        }
        if (validationIndices.isEmpty() && trainIndices.size() > 1) {
            validationIndices.add(trainIndices.remove(trainIndices.size() - 1));
        }
    }

    private static DataLoader.TensorDatasetSplit splitByIndexSlices(
            DataLoader.TensorDataset dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        List<GradTensor[]> train = new ArrayList<>(trainIndices.size());
        List<GradTensor[]> validation = new ArrayList<>(validationIndices.size());
        for (int index : trainIndices) {
            train.add(dataset.get(index));
        }
        for (int index : validationIndices) {
            validation.add(dataset.get(index));
        }
        return new DataLoader.TensorDatasetSplit(samples(train), samples(validation));
    }

    private static DataLoader.TensorDataset samples(List<GradTensor[]> samples) {
        return new DataLoader.TensorDataset(samples.toArray(new GradTensor[0][]));
    }

    private static void requireTrainFraction(double trainFraction) {
        if (trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be between 0 and 1");
        }
    }

    private static long mixSeed(long seed, int label) {
        long value = seed ^ (0x9E3779B97F4A7C15L * (label + 0x632BE5ABL));
        value ^= (value >>> 33);
        value *= 0xff51afd7ed558ccdL;
        value ^= (value >>> 33);
        return value;
    }
}
