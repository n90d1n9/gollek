package tech.kayys.gollek.train.data;

import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.addRowCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.binaryLabelValues;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.columnPositiveCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.copyCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.emptyFoldBuckets;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.foldSizeTargets;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.multiLabelFoldTargets;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.multiLabelStratificationOrder;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.requireSameSize;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.smallestSplit;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.stratifiedThreeWayCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.stratifiedTrainCount;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

final class MultiLabelStratifiedTensorDatasetSplits {
    private MultiLabelStratifiedTensorDatasetSplits() {
    }

    static DataLoader.TensorDatasetSplit stratifiedSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainFraction(trainFraction);
        requireSameSize(labels, dataset.size());

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
        return TensorDatasetSplits.splitByIndices(dataset, trainIndices, validationIndices);
    }

    static DataLoader.TensorDatasetThreeWaySplit stratifiedSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainValidationFractions(trainFraction, validationFraction);
        requireSameSize(labels, dataset.size());

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        SplitCounts rowTargets = TensorDatasetSplits.threeWayCounts(rows, trainFraction, validationFraction);
        int[] splitTargets = new int[] {rowTargets.train(), rowTargets.validation(), rowTargets.test()};
        int[][] labelTargets = new int[3][columns];
        for (int column = 0; column < columns; column++) {
            SplitCounts counts = stratifiedThreeWayCounts(totalPositives[column], trainFraction, validationFraction);
            labelTargets[0][column] = counts.train();
            labelTargets[1][column] = counts.validation();
            labelTargets[2][column] = counts.test();
        }

        int[][] splitCounts = new int[3][columns];
        List<List<Integer>> splitIndices = new ArrayList<>(3);
        splitIndices.add(new ArrayList<>(splitTargets[0]));
        splitIndices.add(new ArrayList<>(splitTargets[1]));
        splitIndices.add(new ArrayList<>(splitTargets[2]));

        for (int row : multiLabelStratificationOrder(flatLabels, rows, columns, totalPositives, seed)) {
            int split = bestMultiLabelThreeWaySplit(
                    flatLabels, row, columns, totalPositives, splitCounts, labelTargets, splitIndices, splitTargets);
            splitIndices.get(split).add(row);
            addRowCounts(flatLabels, row, columns, splitCounts[split], 1);
        }

        improveMultiLabelThreeWayAssignment(
                flatLabels, columns, totalPositives, labelTargets, splitIndices, splitCounts);
        Collections.shuffle(splitIndices.get(0), new Random(seed ^ 0x3C6EF372FE94F82AL));
        Collections.shuffle(splitIndices.get(1), new Random(seed ^ 0xA54FF53A5F1D36F1L));
        Collections.shuffle(splitIndices.get(2), new Random(seed ^ 0x510E527FADE682D1L));
        return TensorDatasetSplits.splitByIndices(dataset, splitIndices.get(0), splitIndices.get(1), splitIndices.get(2));
    }

    static DataLoader.TensorDatasetSplit stratifiedGroupSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            long seed) {
        return MultiLabelStratifiedGroupSplits.stratifiedGroupSplit(dataset, labels, groups, trainFraction, seed);
    }

    static DataLoader.TensorDatasetThreeWaySplit stratifiedGroupSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        return MultiLabelStratifiedGroupSplits.stratifiedGroupSplit(
                dataset, labels, groups, trainFraction, validationFraction, seed);
    }

    static List<DataLoader.TensorDatasetFold> stratifiedKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        requireSameSize(labels, dataset.size());

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        int[] foldTargets = foldSizeTargets(rows, folds);
        int[][] labelTargets = multiLabelFoldTargets(totalPositives, folds);
        int[][] foldCounts = new int[folds][columns];
        List<List<Integer>> validationFolds = emptyFoldBuckets(folds, foldTargets);

        for (int row : multiLabelStratificationOrder(flatLabels, rows, columns, totalPositives, seed)) {
            int fold = bestMultiLabelFold(
                    flatLabels,
                    row,
                    columns,
                    totalPositives,
                    foldCounts,
                    labelTargets,
                    validationFolds,
                    foldTargets);
            validationFolds.get(fold).add(row);
            addRowCounts(flatLabels, row, columns, foldCounts[fold], 1);
        }

        improveMultiLabelFoldAssignment(
                flatLabels, columns, totalPositives, labelTargets, validationFolds, foldCounts);
        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0xC2B2AE3D27D4EB4FL * (fold + 1L))));
        }
        return TensorDatasetSplits.foldsByValidationIndices(dataset, validationFolds);
    }

    static List<DataLoader.TensorDatasetFold> repeatedStratifiedKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        TensorDatasetSplits.requireRepeats(repeats);
        return TensorDatasetSplits.repeatedFolds(
                folds,
                repeats,
                repeat -> stratifiedKFold(dataset, labels, folds, TensorDatasetSplits.repeatSeed(seed, repeat)));
    }

    static List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            long seed) {
        return MultiLabelStratifiedGroupSplits.stratifiedGroupKFold(dataset, labels, groups, folds, seed);
    }

    static List<DataLoader.TensorDatasetFold> repeatedStratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        return MultiLabelStratifiedGroupSplits.repeatedStratifiedGroupKFold(
                dataset, labels, groups, folds, repeats, seed);
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

    private static int bestMultiLabelThreeWaySplit(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[][] splitCounts,
            int[][] labelTargets,
            List<List<Integer>> splitIndices,
            int[] splitTargets) {
        int bestSplit = -1;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int split = 0; split < 3; split++) {
            if (splitIndices.get(split).size() >= splitTargets[split]) {
                continue;
            }
            double penalty = multiLabelThreeWayPenalty(
                    flatLabels,
                    row,
                    columns,
                    totalPositives,
                    splitCounts,
                    labelTargets,
                    splitIndices,
                    splitTargets,
                    split);
            if (penalty + 1e-9 < bestPenalty) {
                bestPenalty = penalty;
                bestSplit = split;
            }
        }
        return bestSplit >= 0 ? bestSplit : smallestSplit(splitIndices);
    }

    private static double multiLabelThreeWayPenalty(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[][] splitCounts,
            int[][] labelTargets,
            List<List<Integer>> splitIndices,
            int[] splitTargets,
            int candidateSplit) {
        double penalty = 0.0;
        for (int split = 0; split < 3; split++) {
            int size = splitIndices.get(split).size() + (split == candidateSplit ? 1 : 0);
            penalty += Math.pow(size - splitTargets[split], 2) * columns;
            for (int column = 0; column < columns; column++) {
                int after = splitCounts[split][column];
                if (split == candidateSplit) {
                    after += flatLabels[row * columns + column];
                }
                double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
                penalty += Math.pow(after - labelTargets[split][column], 2) * labelWeight;
            }
        }
        return penalty;
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

    private static void improveMultiLabelThreeWayAssignment(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[][] labelTargets,
            List<List<Integer>> splitIndices,
            int[][] splitCounts) {
        double bestScore = multiLabelThreeWayAssignmentScore(splitCounts, labelTargets, totalPositives);
        boolean improved;
        do {
            improved = false;
            int bestLeftSplit = -1;
            int bestRightSplit = -1;
            int bestLeftPosition = -1;
            int bestRightPosition = -1;
            int[][] bestCounts = null;

            for (int leftSplit = 0; leftSplit < 2; leftSplit++) {
                for (int rightSplit = leftSplit + 1; rightSplit < 3; rightSplit++) {
                    List<Integer> leftRows = splitIndices.get(leftSplit);
                    List<Integer> rightRows = splitIndices.get(rightSplit);
                    for (int leftPosition = 0; leftPosition < leftRows.size(); leftPosition++) {
                        int leftRow = leftRows.get(leftPosition);
                        for (int rightPosition = 0; rightPosition < rightRows.size(); rightPosition++) {
                            int rightRow = rightRows.get(rightPosition);
                            int[][] candidateCounts = swappedThreeWayCounts(
                                    flatLabels, columns, splitCounts, leftSplit, rightSplit, leftRow, rightRow);
                            double candidateScore = multiLabelThreeWayAssignmentScore(
                                    candidateCounts, labelTargets, totalPositives);
                            if (candidateScore + 1e-9 < bestScore) {
                                bestScore = candidateScore;
                                bestLeftSplit = leftSplit;
                                bestRightSplit = rightSplit;
                                bestLeftPosition = leftPosition;
                                bestRightPosition = rightPosition;
                                bestCounts = candidateCounts;
                                improved = true;
                            }
                        }
                    }
                }
            }

            if (improved) {
                List<Integer> leftRows = splitIndices.get(bestLeftSplit);
                List<Integer> rightRows = splitIndices.get(bestRightSplit);
                int leftRow = leftRows.get(bestLeftPosition);
                leftRows.set(bestLeftPosition, rightRows.get(bestRightPosition));
                rightRows.set(bestRightPosition, leftRow);
                copyCounts(bestCounts, splitCounts);
            } else {
                improved = improveMultiLabelThreeWayCycle(
                        flatLabels, columns, totalPositives, labelTargets, splitIndices, splitCounts, bestScore);
            }
        } while (improved);
    }

    private static boolean improveMultiLabelThreeWayCycle(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[][] labelTargets,
            List<List<Integer>> splitIndices,
            int[][] splitCounts,
            double currentScore) {
        long candidateCount = 1L;
        for (List<Integer> split : splitIndices) {
            candidateCount *= split.size();
        }
        if (candidateCount > 250_000L) {
            return false;
        }

        double bestScore = currentScore;
        int bestTrainPosition = -1;
        int bestValidationPosition = -1;
        int bestTestPosition = -1;
        boolean bestForwardRotation = true;
        int[][] bestCounts = null;

        List<Integer> trainRows = splitIndices.get(0);
        List<Integer> validationRows = splitIndices.get(1);
        List<Integer> testRows = splitIndices.get(2);
        for (int trainPosition = 0; trainPosition < trainRows.size(); trainPosition++) {
            int trainRow = trainRows.get(trainPosition);
            for (int validationPosition = 0; validationPosition < validationRows.size(); validationPosition++) {
                int validationRow = validationRows.get(validationPosition);
                for (int testPosition = 0; testPosition < testRows.size(); testPosition++) {
                    int testRow = testRows.get(testPosition);
                    int[][] forwardCounts = rotatedThreeWayCounts(
                            flatLabels, columns, splitCounts, trainRow, validationRow, testRow, true);
                    double forwardScore = multiLabelThreeWayAssignmentScore(
                            forwardCounts, labelTargets, totalPositives);
                    if (forwardScore + 1e-9 < bestScore) {
                        bestScore = forwardScore;
                        bestTrainPosition = trainPosition;
                        bestValidationPosition = validationPosition;
                        bestTestPosition = testPosition;
                        bestForwardRotation = true;
                        bestCounts = forwardCounts;
                    }

                    int[][] reverseCounts = rotatedThreeWayCounts(
                            flatLabels, columns, splitCounts, trainRow, validationRow, testRow, false);
                    double reverseScore = multiLabelThreeWayAssignmentScore(
                            reverseCounts, labelTargets, totalPositives);
                    if (reverseScore + 1e-9 < bestScore) {
                        bestScore = reverseScore;
                        bestTrainPosition = trainPosition;
                        bestValidationPosition = validationPosition;
                        bestTestPosition = testPosition;
                        bestForwardRotation = false;
                        bestCounts = reverseCounts;
                    }
                }
            }
        }

        if (bestCounts == null) {
            return false;
        }

        int trainRow = trainRows.get(bestTrainPosition);
        int validationRow = validationRows.get(bestValidationPosition);
        int testRow = testRows.get(bestTestPosition);
        if (bestForwardRotation) {
            trainRows.set(bestTrainPosition, testRow);
            validationRows.set(bestValidationPosition, trainRow);
            testRows.set(bestTestPosition, validationRow);
        } else {
            trainRows.set(bestTrainPosition, validationRow);
            validationRows.set(bestValidationPosition, testRow);
            testRows.set(bestTestPosition, trainRow);
        }
        copyCounts(bestCounts, splitCounts);
        return true;
    }

    private static int[][] swappedThreeWayCounts(
            int[] flatLabels,
            int columns,
            int[][] splitCounts,
            int leftSplit,
            int rightSplit,
            int leftRow,
            int rightRow) {
        int[][] candidate = copyCounts(splitCounts);
        addRowCounts(flatLabels, leftRow, columns, candidate[leftSplit], -1);
        addRowCounts(flatLabels, rightRow, columns, candidate[leftSplit], 1);
        addRowCounts(flatLabels, rightRow, columns, candidate[rightSplit], -1);
        addRowCounts(flatLabels, leftRow, columns, candidate[rightSplit], 1);
        return candidate;
    }

    private static int[][] rotatedThreeWayCounts(
            int[] flatLabels,
            int columns,
            int[][] splitCounts,
            int trainRow,
            int validationRow,
            int testRow,
            boolean forward) {
        int[][] candidate = copyCounts(splitCounts);
        if (forward) {
            moveRowCounts(flatLabels, trainRow, columns, candidate, 0, 1);
            moveRowCounts(flatLabels, validationRow, columns, candidate, 1, 2);
            moveRowCounts(flatLabels, testRow, columns, candidate, 2, 0);
        } else {
            moveRowCounts(flatLabels, trainRow, columns, candidate, 0, 2);
            moveRowCounts(flatLabels, testRow, columns, candidate, 2, 1);
            moveRowCounts(flatLabels, validationRow, columns, candidate, 1, 0);
        }
        return candidate;
    }

    private static void moveRowCounts(
            int[] flatLabels,
            int row,
            int columns,
            int[][] splitCounts,
            int fromSplit,
            int toSplit) {
        addRowCounts(flatLabels, row, columns, splitCounts[fromSplit], -1);
        addRowCounts(flatLabels, row, columns, splitCounts[toSplit], 1);
    }

    private static double multiLabelThreeWayAssignmentScore(
            int[][] splitCounts,
            int[][] labelTargets,
            int[] totalPositives) {
        double score = 0.0;
        for (int split = 0; split < 3; split++) {
            for (int column = 0; column < totalPositives.length; column++) {
                double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
                score += Math.pow(splitCounts[split][column] - labelTargets[split][column], 2) * labelWeight;
            }
        }
        return score;
    }

    private static int bestMultiLabelFold(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[][] foldCounts,
            int[][] labelTargets,
            List<List<Integer>> validationFolds,
            int[] foldTargets) {
        int bestFold = -1;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int fold = 0; fold < foldTargets.length; fold++) {
            if (validationFolds.get(fold).size() >= foldTargets[fold]) {
                continue;
            }
            double penalty = multiLabelFoldPenalty(
                    flatLabels,
                    row,
                    columns,
                    totalPositives,
                    foldCounts,
                    labelTargets,
                    validationFolds,
                    foldTargets,
                    fold);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9
                            && validationFolds.get(fold).size() < validationFolds.get(Math.max(bestFold, 0)).size())) {
                bestPenalty = penalty;
                bestFold = fold;
            }
        }
        return bestFold >= 0 ? bestFold : smallestSplit(validationFolds);
    }

    private static double multiLabelFoldPenalty(
            int[] flatLabels,
            int row,
            int columns,
            int[] totalPositives,
            int[][] foldCounts,
            int[][] labelTargets,
            List<List<Integer>> validationFolds,
            int[] foldTargets,
            int candidateFold) {
        double penalty = 0.0;
        for (int fold = 0; fold < foldTargets.length; fold++) {
            int size = validationFolds.get(fold).size() + (fold == candidateFold ? 1 : 0);
            penalty += Math.pow(size - foldTargets[fold], 2) * columns;
            for (int column = 0; column < columns; column++) {
                int after = foldCounts[fold][column];
                if (fold == candidateFold) {
                    after += flatLabels[row * columns + column];
                }
                double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
                penalty += Math.pow(after - labelTargets[fold][column], 2) * labelWeight;
            }
        }
        return penalty;
    }

    private static void improveMultiLabelFoldAssignment(
            int[] flatLabels,
            int columns,
            int[] totalPositives,
            int[][] labelTargets,
            List<List<Integer>> validationFolds,
            int[][] foldCounts) {
        double bestScore = multiLabelFoldAssignmentScore(foldCounts, labelTargets, totalPositives);
        boolean improved;
        do {
            improved = false;
            int bestLeftFold = -1;
            int bestRightFold = -1;
            int bestLeftPosition = -1;
            int bestRightPosition = -1;
            int[][] bestCounts = null;

            for (int leftFold = 0; leftFold < validationFolds.size() - 1; leftFold++) {
                for (int rightFold = leftFold + 1; rightFold < validationFolds.size(); rightFold++) {
                    List<Integer> leftRows = validationFolds.get(leftFold);
                    List<Integer> rightRows = validationFolds.get(rightFold);
                    for (int leftPosition = 0; leftPosition < leftRows.size(); leftPosition++) {
                        int leftRow = leftRows.get(leftPosition);
                        for (int rightPosition = 0; rightPosition < rightRows.size(); rightPosition++) {
                            int rightRow = rightRows.get(rightPosition);
                            int[][] candidateCounts = swappedFoldCounts(
                                    flatLabels, columns, foldCounts, leftFold, rightFold, leftRow, rightRow);
                            double candidateScore = multiLabelFoldAssignmentScore(
                                    candidateCounts, labelTargets, totalPositives);
                            if (candidateScore + 1e-9 < bestScore) {
                                bestScore = candidateScore;
                                bestLeftFold = leftFold;
                                bestRightFold = rightFold;
                                bestLeftPosition = leftPosition;
                                bestRightPosition = rightPosition;
                                bestCounts = candidateCounts;
                                improved = true;
                            }
                        }
                    }
                }
            }

            if (improved) {
                List<Integer> leftRows = validationFolds.get(bestLeftFold);
                List<Integer> rightRows = validationFolds.get(bestRightFold);
                int leftRow = leftRows.get(bestLeftPosition);
                leftRows.set(bestLeftPosition, rightRows.get(bestRightPosition));
                rightRows.set(bestRightPosition, leftRow);
                copyCounts(bestCounts, foldCounts);
            }
        } while (improved);
    }

    private static int[][] swappedFoldCounts(
            int[] flatLabels,
            int columns,
            int[][] foldCounts,
            int leftFold,
            int rightFold,
            int leftRow,
            int rightRow) {
        int[][] candidate = copyCounts(foldCounts);
        addRowCounts(flatLabels, leftRow, columns, candidate[leftFold], -1);
        addRowCounts(flatLabels, rightRow, columns, candidate[leftFold], 1);
        addRowCounts(flatLabels, rightRow, columns, candidate[rightFold], -1);
        addRowCounts(flatLabels, leftRow, columns, candidate[rightFold], 1);
        return candidate;
    }

    private static double multiLabelFoldAssignmentScore(
            int[][] foldCounts,
            int[][] labelTargets,
            int[] totalPositives) {
        double score = 0.0;
        for (int fold = 0; fold < foldCounts.length; fold++) {
            for (int column = 0; column < totalPositives.length; column++) {
                double labelWeight = totalPositives[column] == 0 ? 0.0 : 1.0 + (1.0 / totalPositives[column]);
                score += Math.pow(foldCounts[fold][column] - labelTargets[fold][column], 2) * labelWeight;
            }
        }
        return score;
    }


}
