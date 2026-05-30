package tech.kayys.gollek.train.data;

import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.addRowCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.binaryLabelValues;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.columnPositiveCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.copyCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.requireSameSize;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.stratifiedThreeWayCounts;
import static tech.kayys.gollek.train.data.MultiLabelStratificationSupport.stratifiedTrainCount;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import tech.kayys.gollek.train.data.MultiLabelStratificationSupport.StratifiedGroupBucket;

final class MultiLabelStratifiedGroupSplits {
    private MultiLabelStratifiedGroupSplits() {
    }

    static DataLoader.TensorDatasetSplit stratifiedGroupSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainFraction(trainFraction);
        requireMultiLabelGroupSplitInputs(labels, dataset.size(), groups, 2, "multi-label stratified group split");

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        int trainTargetSize = stratifiedTrainCount(rows, trainFraction);
        int[] splitTargets = new int[] {trainTargetSize, rows - trainTargetSize};
        int[][] labelTargets = new int[2][columns];
        for (int column = 0; column < columns; column++) {
            int trainTarget = stratifiedTrainCount(totalPositives[column], trainFraction);
            labelTargets[0][column] = trainTarget;
            labelTargets[1][column] = totalPositives[column] - trainTarget;
        }

        List<List<Integer>> splitIndices = multiLabelStratifiedGroupPartitions(
                flatLabels,
                columns,
                groups,
                totalPositives,
                splitTargets,
                labelTargets,
                seed,
                "multi-label stratified group split");
        Collections.shuffle(splitIndices.get(0), new Random(seed ^ 0x243F6A8885A308D3L));
        Collections.shuffle(splitIndices.get(1), new Random(seed ^ 0x13198A2E03707344L));
        return TensorDatasetSplits.splitByIndices(dataset, splitIndices.get(0), splitIndices.get(1));
    }

    static DataLoader.TensorDatasetThreeWaySplit stratifiedGroupSplit(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainValidationFractions(trainFraction, validationFraction);
        requireMultiLabelGroupSplitInputs(
                labels, dataset.size(), groups, 3, "multi-label stratified group three-way split");

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

        List<List<Integer>> splitIndices = multiLabelStratifiedGroupPartitions(
                flatLabels,
                columns,
                groups,
                totalPositives,
                splitTargets,
                labelTargets,
                seed,
                "multi-label stratified group three-way split");
        Collections.shuffle(splitIndices.get(0), new Random(seed ^ 0xA4093822299F31D0L));
        Collections.shuffle(splitIndices.get(1), new Random(seed ^ 0x082EFA98EC4E6C89L));
        Collections.shuffle(splitIndices.get(2), new Random(seed ^ 0x452821E638D01377L));
        return TensorDatasetSplits.splitByIndices(dataset, splitIndices.get(0), splitIndices.get(1), splitIndices.get(2));
    }

    static List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(groups, "groups must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        requireSameSize(labels, dataset.size());
        if (groups.length != dataset.size()) {
            throw new IllegalArgumentException(
                    "groups and dataset must have same size, got: " + groups.length + " vs " + dataset.size());
        }

        int rows = labels.rows();
        int columns = labels.columns();
        int[] flatLabels = binaryLabelValues(labels);
        int[] totalPositives = columnPositiveCounts(flatLabels, rows, columns);
        Map<Integer, List<Integer>> indicesByGroup = indicesByGroup(groups);
        if (indicesByGroup.size() < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to group count for multi-label stratified group k-fold, got folds="
                            + folds + ", groups=" + indicesByGroup.size());
        }

        List<StratifiedGroupBucket> buckets = groupBuckets(indicesByGroup, flatLabels, columns, seed);
        List<List<StratifiedGroupBucket>> foldBuckets = new ArrayList<>(folds);
        int[][] foldLabelCounts = new int[folds][columns];
        int[] foldSizes = new int[folds];
        for (int fold = 0; fold < folds; fold++) {
            foldBuckets.add(new ArrayList<>());
        }

        double targetFoldSize = rows / (double) folds;
        for (StratifiedGroupBucket bucket : buckets) {
            int fold = bestMultiLabelStratifiedGroupFold(
                    bucket, foldLabelCounts, foldSizes, totalPositives, targetFoldSize);
            addBucketToFold(bucket, fold, foldBuckets, foldLabelCounts, foldSizes, 1);
        }

        improveMultiLabelStratifiedGroupAssignment(
                foldBuckets, foldLabelCounts, foldSizes, totalPositives, targetFoldSize);

        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            List<Integer> validationIndices = new ArrayList<>(foldSizes[fold]);
            for (StratifiedGroupBucket bucket : foldBuckets.get(fold)) {
                validationIndices.addAll(bucket.indices());
            }
            Collections.shuffle(validationIndices, new Random(seed ^ (0x165667B19E3779F9L * (fold + 1L))));
            validationFolds.add(validationIndices);
        }
        return TensorDatasetSplits.foldsByValidationIndices(dataset, validationFolds);
    }

    static List<DataLoader.TensorDatasetFold> repeatedStratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            BinaryLabelMatrix labels,
            int[] groups,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        TensorDatasetSplits.requireRepeats(repeats);
        return TensorDatasetSplits.repeatedFolds(
                folds,
                repeats,
                repeat -> stratifiedGroupKFold(
                        dataset, labels, groups, folds, TensorDatasetSplits.repeatSeed(seed, repeat)));
    }

    private static void requireMultiLabelGroupSplitInputs(
            BinaryLabelMatrix labels,
            int sampleCount,
            int[] groups,
            int splitCount,
            String splitName) {
        requireSameSize(labels, sampleCount);
        Objects.requireNonNull(groups, "groups must not be null");
        if (groups.length != sampleCount) {
            throw new IllegalArgumentException(
                    "groups and dataset must have same size, got: " + groups.length + " vs " + sampleCount);
        }
        if (new HashSet<Integer>(Arrays.stream(groups).boxed().toList()).size() < splitCount) {
            throw new IllegalArgumentException(
                    splitName + " requires at least " + splitCount + " groups to keep groups leakage-safe");
        }
    }

    private static List<List<Integer>> multiLabelStratifiedGroupPartitions(
            int[] flatLabels,
            int columns,
            int[] groups,
            int[] totalPositives,
            int[] splitTargets,
            int[][] labelTargets,
            long seed,
            String splitName) {
        List<StratifiedGroupBucket> buckets = groupBuckets(indicesByGroup(groups), flatLabels, columns, seed);

        List<List<StratifiedGroupBucket>> splitBuckets = new ArrayList<>(splitTargets.length);
        int[][] splitLabelCounts = new int[splitTargets.length][columns];
        int[] splitSizes = new int[splitTargets.length];
        for (int split = 0; split < splitTargets.length; split++) {
            splitBuckets.add(new ArrayList<>());
        }

        for (StratifiedGroupBucket bucket : buckets) {
            int split = bestMultiLabelGroupPartition(
                    bucket, splitLabelCounts, splitSizes, totalPositives, splitTargets, labelTargets);
            addBucketToFold(bucket, split, splitBuckets, splitLabelCounts, splitSizes, 1);
        }

        improveMultiLabelGroupPartitionAssignment(
                splitBuckets, splitLabelCounts, splitSizes, totalPositives, splitTargets, labelTargets);
        requireNonEmptyGroupPartitions(splitBuckets, splitName);

        List<List<Integer>> splitIndices = new ArrayList<>(splitBuckets.size());
        for (int split = 0; split < splitBuckets.size(); split++) {
            List<Integer> indices = new ArrayList<>(splitSizes[split]);
            for (StratifiedGroupBucket bucket : splitBuckets.get(split)) {
                indices.addAll(bucket.indices());
            }
            splitIndices.add(indices);
        }
        return splitIndices;
    }

    private static Map<Integer, List<Integer>> indicesByGroup(int[] groups) {
        Map<Integer, List<Integer>> indicesByGroup = new LinkedHashMap<>();
        for (int row = 0; row < groups.length; row++) {
            indicesByGroup.computeIfAbsent(groups[row], ignored -> new ArrayList<>()).add(row);
        }
        return indicesByGroup;
    }

    private static List<StratifiedGroupBucket> groupBuckets(
            Map<Integer, List<Integer>> indicesByGroup,
            int[] flatLabels,
            int columns,
            long seed) {
        List<StratifiedGroupBucket> buckets = new ArrayList<>(indicesByGroup.size());
        for (List<Integer> groupIndices : indicesByGroup.values()) {
            int[] labelCounts = new int[columns];
            for (int row : groupIndices) {
                addRowCounts(flatLabels, row, columns, labelCounts, 1);
            }
            buckets.add(new StratifiedGroupBucket(new ArrayList<>(groupIndices), labelCounts));
        }
        Collections.shuffle(buckets, new Random(seed));
        buckets.sort(Comparator
                .comparingInt(StratifiedGroupBucket::labelMassSquared)
                .thenComparingInt(StratifiedGroupBucket::size)
                .reversed());
        return buckets;
    }

    private static void requireNonEmptyGroupPartitions(
            List<List<StratifiedGroupBucket>> splitBuckets,
            String splitName) {
        for (int split = 0; split < splitBuckets.size(); split++) {
            if (splitBuckets.get(split).isEmpty()) {
                throw new IllegalArgumentException(
                        "unable to create non-empty " + splitName + "; split " + split + " received no groups");
            }
        }
    }

    private static int bestMultiLabelGroupPartition(
            StratifiedGroupBucket bucket,
            int[][] splitLabelCounts,
            int[] splitSizes,
            int[] totalPositives,
            int[] splitTargets,
            int[][] labelTargets) {
        int bestSplit = 0;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int split = 0; split < splitSizes.length; split++) {
            double penalty = multiLabelGroupPartitionPenalty(
                    bucket, split, splitLabelCounts, splitSizes, totalPositives, splitTargets, labelTargets);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && splitSizes[split] < splitSizes[bestSplit])) {
                bestPenalty = penalty;
                bestSplit = split;
            }
        }
        return bestSplit;
    }

    private static double multiLabelGroupPartitionPenalty(
            StratifiedGroupBucket bucket,
            int candidateSplit,
            int[][] splitLabelCounts,
            int[] splitSizes,
            int[] totalPositives,
            int[] splitTargets,
            int[][] labelTargets) {
        double penalty = 0.0;
        for (int split = 0; split < splitSizes.length; split++) {
            double afterSize = splitSizes[split] + (split == candidateSplit ? bucket.size() : 0);
            penalty += Math.pow(afterSize - splitTargets[split], 2) / Math.max(1.0, splitTargets[split]);
            for (int column = 0; column < totalPositives.length; column++) {
                if (totalPositives[column] == 0) {
                    continue;
                }
                double after = splitLabelCounts[split][column]
                        + (split == candidateSplit ? bucket.labelCounts()[column] : 0);
                penalty += Math.pow(after - labelTargets[split][column], 2)
                        * (1.0 + (1.0 / totalPositives[column]));
            }
        }
        return penalty;
    }

    private static void improveMultiLabelGroupPartitionAssignment(
            List<List<StratifiedGroupBucket>> splitBuckets,
            int[][] splitLabelCounts,
            int[] splitSizes,
            int[] totalPositives,
            int[] splitTargets,
            int[][] labelTargets) {
        double bestScore = multiLabelGroupPartitionAssignmentScore(
                splitLabelCounts, splitSizes, totalPositives, splitTargets, labelTargets);
        boolean improved;
        do {
            improved = false;
            int bestLeftSplit = -1;
            int bestRightSplit = -1;
            int bestLeftPosition = -1;
            int bestRightPosition = -1;
            int[][] bestCounts = null;
            int[] bestSizes = null;

            for (int leftSplit = 0; leftSplit < splitBuckets.size() - 1; leftSplit++) {
                for (int rightSplit = leftSplit + 1; rightSplit < splitBuckets.size(); rightSplit++) {
                    List<StratifiedGroupBucket> leftBuckets = splitBuckets.get(leftSplit);
                    List<StratifiedGroupBucket> rightBuckets = splitBuckets.get(rightSplit);
                    for (int leftPosition = 0; leftPosition < leftBuckets.size(); leftPosition++) {
                        StratifiedGroupBucket left = leftBuckets.get(leftPosition);
                        for (int rightPosition = 0; rightPosition < rightBuckets.size(); rightPosition++) {
                            StratifiedGroupBucket right = rightBuckets.get(rightPosition);
                            int[][] candidateCounts = swappedGroupBucketCounts(
                                    splitLabelCounts, leftSplit, rightSplit, left, right);
                            int[] candidateSizes = swappedGroupBucketSizes(
                                    splitSizes, leftSplit, rightSplit, left, right);
                            double candidateScore = multiLabelGroupPartitionAssignmentScore(
                                    candidateCounts, candidateSizes, totalPositives, splitTargets, labelTargets);
                            if (candidateScore + 1e-9 < bestScore) {
                                bestScore = candidateScore;
                                bestLeftSplit = leftSplit;
                                bestRightSplit = rightSplit;
                                bestLeftPosition = leftPosition;
                                bestRightPosition = rightPosition;
                                bestCounts = candidateCounts;
                                bestSizes = candidateSizes;
                                improved = true;
                            }
                        }
                    }
                }
            }

            if (improved) {
                List<StratifiedGroupBucket> leftBuckets = splitBuckets.get(bestLeftSplit);
                List<StratifiedGroupBucket> rightBuckets = splitBuckets.get(bestRightSplit);
                StratifiedGroupBucket left = leftBuckets.get(bestLeftPosition);
                leftBuckets.set(bestLeftPosition, rightBuckets.get(bestRightPosition));
                rightBuckets.set(bestRightPosition, left);
                copyCounts(bestCounts, splitLabelCounts);
                System.arraycopy(bestSizes, 0, splitSizes, 0, splitSizes.length);
            }
        } while (improved);
    }

    private static double multiLabelGroupPartitionAssignmentScore(
            int[][] splitLabelCounts,
            int[] splitSizes,
            int[] totalPositives,
            int[] splitTargets,
            int[][] labelTargets) {
        double score = 0.0;
        for (int split = 0; split < splitSizes.length; split++) {
            score += Math.pow(splitSizes[split] - splitTargets[split], 2) / Math.max(1.0, splitTargets[split]);
            for (int column = 0; column < totalPositives.length; column++) {
                if (totalPositives[column] == 0) {
                    continue;
                }
                score += Math.pow(splitLabelCounts[split][column] - labelTargets[split][column], 2)
                        * (1.0 + (1.0 / totalPositives[column]));
            }
        }
        return score;
    }

    private static int bestMultiLabelStratifiedGroupFold(
            StratifiedGroupBucket bucket,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalPositives,
            double targetFoldSize) {
        int bestFold = 0;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            double penalty = multiLabelStratifiedGroupPenalty(
                    bucket, fold, foldLabelCounts, foldSizes, totalPositives, targetFoldSize);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && foldSizes[fold] < foldSizes[bestFold])) {
                bestPenalty = penalty;
                bestFold = fold;
            }
        }
        return bestFold;
    }

    private static double multiLabelStratifiedGroupPenalty(
            StratifiedGroupBucket bucket,
            int candidateFold,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalPositives,
            double targetFoldSize) {
        double penalty = 0.0;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            double afterSize = foldSizes[fold] + (fold == candidateFold ? bucket.size() : 0);
            penalty += Math.pow(afterSize - targetFoldSize, 2) / Math.max(1.0, targetFoldSize);
            for (int column = 0; column < totalPositives.length; column++) {
                if (totalPositives[column] == 0) {
                    continue;
                }
                double target = totalPositives[column] / (double) foldSizes.length;
                double after = foldLabelCounts[fold][column]
                        + (fold == candidateFold ? bucket.labelCounts()[column] : 0);
                penalty += Math.pow(after - target, 2) * (1.0 + (1.0 / totalPositives[column]));
            }
        }
        return penalty;
    }

    private static void addBucketToFold(
            StratifiedGroupBucket bucket,
            int fold,
            List<List<StratifiedGroupBucket>> foldBuckets,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int delta) {
        if (delta > 0) {
            foldBuckets.get(fold).add(bucket);
        } else {
            foldBuckets.get(fold).remove(bucket);
        }
        foldSizes[fold] += bucket.size() * delta;
        for (int column = 0; column < bucket.labelCounts().length; column++) {
            foldLabelCounts[fold][column] += bucket.labelCounts()[column] * delta;
        }
    }

    private static void improveMultiLabelStratifiedGroupAssignment(
            List<List<StratifiedGroupBucket>> foldBuckets,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalPositives,
            double targetFoldSize) {
        double bestScore = multiLabelStratifiedGroupAssignmentScore(
                foldLabelCounts, foldSizes, totalPositives, targetFoldSize);
        boolean improved;
        do {
            improved = false;
            int bestLeftFold = -1;
            int bestRightFold = -1;
            int bestLeftPosition = -1;
            int bestRightPosition = -1;
            int[][] bestCounts = null;
            int[] bestSizes = null;

            for (int leftFold = 0; leftFold < foldBuckets.size() - 1; leftFold++) {
                for (int rightFold = leftFold + 1; rightFold < foldBuckets.size(); rightFold++) {
                    List<StratifiedGroupBucket> leftBuckets = foldBuckets.get(leftFold);
                    List<StratifiedGroupBucket> rightBuckets = foldBuckets.get(rightFold);
                    for (int leftPosition = 0; leftPosition < leftBuckets.size(); leftPosition++) {
                        StratifiedGroupBucket left = leftBuckets.get(leftPosition);
                        for (int rightPosition = 0; rightPosition < rightBuckets.size(); rightPosition++) {
                            StratifiedGroupBucket right = rightBuckets.get(rightPosition);
                            int[][] candidateCounts = swappedGroupBucketCounts(
                                    foldLabelCounts, leftFold, rightFold, left, right);
                            int[] candidateSizes = swappedGroupBucketSizes(foldSizes, leftFold, rightFold, left, right);
                            double candidateScore = multiLabelStratifiedGroupAssignmentScore(
                                    candidateCounts, candidateSizes, totalPositives, targetFoldSize);
                            if (candidateScore + 1e-9 < bestScore) {
                                bestScore = candidateScore;
                                bestLeftFold = leftFold;
                                bestRightFold = rightFold;
                                bestLeftPosition = leftPosition;
                                bestRightPosition = rightPosition;
                                bestCounts = candidateCounts;
                                bestSizes = candidateSizes;
                                improved = true;
                            }
                        }
                    }
                }
            }

            if (improved) {
                List<StratifiedGroupBucket> leftBuckets = foldBuckets.get(bestLeftFold);
                List<StratifiedGroupBucket> rightBuckets = foldBuckets.get(bestRightFold);
                StratifiedGroupBucket left = leftBuckets.get(bestLeftPosition);
                leftBuckets.set(bestLeftPosition, rightBuckets.get(bestRightPosition));
                rightBuckets.set(bestRightPosition, left);
                copyCounts(bestCounts, foldLabelCounts);
                System.arraycopy(bestSizes, 0, foldSizes, 0, foldSizes.length);
            }
        } while (improved);
    }

    private static int[][] swappedGroupBucketCounts(
            int[][] foldLabelCounts,
            int leftFold,
            int rightFold,
            StratifiedGroupBucket left,
            StratifiedGroupBucket right) {
        int[][] candidate = copyCounts(foldLabelCounts);
        for (int column = 0; column < left.labelCounts().length; column++) {
            candidate[leftFold][column] += right.labelCounts()[column] - left.labelCounts()[column];
            candidate[rightFold][column] += left.labelCounts()[column] - right.labelCounts()[column];
        }
        return candidate;
    }

    private static int[] swappedGroupBucketSizes(
            int[] foldSizes,
            int leftFold,
            int rightFold,
            StratifiedGroupBucket left,
            StratifiedGroupBucket right) {
        int[] candidate = foldSizes.clone();
        candidate[leftFold] += right.size() - left.size();
        candidate[rightFold] += left.size() - right.size();
        return candidate;
    }

    private static double multiLabelStratifiedGroupAssignmentScore(
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalPositives,
            double targetFoldSize) {
        double score = 0.0;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            score += Math.pow(foldSizes[fold] - targetFoldSize, 2) / Math.max(1.0, targetFoldSize);
            for (int column = 0; column < totalPositives.length; column++) {
                if (totalPositives[column] == 0) {
                    continue;
                }
                double target = totalPositives[column] / (double) foldSizes.length;
                score += Math.pow(foldLabelCounts[fold][column] - target, 2)
                        * (1.0 + (1.0 / totalPositives[column]));
            }
        }
        return score;
    }
}
