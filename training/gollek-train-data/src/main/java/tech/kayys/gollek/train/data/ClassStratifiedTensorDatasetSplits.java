package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

final class ClassStratifiedTensorDatasetSplits {
    private ClassStratifiedTensorDatasetSplits() {
    }

    static DataLoader.TensorDatasetSplit stratifiedSplit(
            DataLoader.TensorDataset dataset,
            int[] labels,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainFraction(trainFraction);
        requireSameSize(labels, dataset.size(), "labels");

        Map<Integer, List<Integer>> groups = labelGroups(labels);
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
        return TensorDatasetSplits.splitByIndices(dataset, trainIndices, validationIndices);
    }

    static DataLoader.TensorDatasetThreeWaySplit stratifiedSplit(
            DataLoader.TensorDataset dataset,
            int[] labels,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireTrainValidationFractions(trainFraction, validationFraction);
        requireSameSize(labels, dataset.size(), "labels");

        Map<Integer, List<Integer>> groups = labelGroups(labels);
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        List<Integer> testIndices = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, entry.getKey())));
            SplitCounts counts = stratifiedThreeWayCounts(group.size(), trainFraction, validationFraction);
            int trainEnd = counts.train();
            int validationEnd = trainEnd + counts.validation();
            trainIndices.addAll(group.subList(0, trainEnd));
            validationIndices.addAll(group.subList(trainEnd, validationEnd));
            testIndices.addAll(group.subList(validationEnd, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices, testIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ 0x6A09E667F3BCC909L));
        Collections.shuffle(validationIndices, new Random(seed ^ 0xBB67AE8584CAA73BL));
        Collections.shuffle(testIndices, new Random(seed ^ 0x3C6EF372FE94F82AL));
        return TensorDatasetSplits.splitByIndices(dataset, trainIndices, validationIndices, testIndices);
    }

    static List<DataLoader.TensorDatasetFold> stratifiedKFold(
            DataLoader.TensorDataset dataset,
            int[] labels,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        requireSameSize(labels, dataset.size(), "labels");

        Map<Integer, List<Integer>> groups = labelGroups(labels);
        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            if (group.size() < folds) {
                throw new IllegalArgumentException(
                        "each class must contain at least folds samples for stratified k-fold; label "
                                + entry.getKey() + " has " + group.size() + ", folds=" + folds);
            }
            Collections.shuffle(group, new Random(mixSeed(seed, entry.getKey())));
            for (int i = 0; i < group.size(); i++) {
                validationFolds.get(i % folds).add(group.get(i));
            }
        }

        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0x9E3779B97F4A7C15L * (fold + 1L))));
        }
        return TensorDatasetSplits.foldsByValidationIndices(dataset, validationFolds);
    }

    static List<DataLoader.TensorDatasetFold> repeatedStratifiedKFold(
            DataLoader.TensorDataset dataset,
            int[] labels,
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

    static List<DataLoader.TensorDatasetFold> groupKFold(
            DataLoader.TensorDataset dataset,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        requireSameSize(groups, dataset.size(), "groups");

        Map<Integer, List<Integer>> indicesByGroup = new LinkedHashMap<>();
        for (int i = 0; i < groups.length; i++) {
            indicesByGroup.computeIfAbsent(groups[i], ignored -> new ArrayList<>()).add(i);
        }
        if (indicesByGroup.size() < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to group count for group k-fold, got folds="
                            + folds + ", groups=" + indicesByGroup.size());
        }

        List<List<Integer>> groupedIndices = new ArrayList<>(indicesByGroup.size());
        for (List<Integer> indices : indicesByGroup.values()) {
            groupedIndices.add(new ArrayList<>(indices));
        }
        Collections.shuffle(groupedIndices, new Random(seed));
        groupedIndices.sort(Comparator.comparingInt((List<Integer> indices) -> indices.size()).reversed());

        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        int[] foldSizes = new int[folds];
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }
        for (List<Integer> groupIndices : groupedIndices) {
            int targetFold = 0;
            for (int fold = 1; fold < folds; fold++) {
                if (foldSizes[fold] < foldSizes[targetFold]) {
                    targetFold = fold;
                }
            }
            validationFolds.get(targetFold).addAll(groupIndices);
            foldSizes[targetFold] += groupIndices.size();
        }

        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0xD1B54A32D192ED03L * (fold + 1L))));
        }
        return TensorDatasetSplits.foldsByValidationIndices(dataset, validationFolds);
    }

    static List<DataLoader.TensorDatasetFold> stratifiedGroupKFold(
            DataLoader.TensorDataset dataset,
            int[] labels,
            int[] groups,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        TensorDatasetSplits.requireKFold(folds, dataset.size());
        requireSameSize(labels, dataset.size(), "labels");
        requireSameSize(groups, dataset.size(), "groups");

        Map<Integer, Integer> labelColumns = labelColumns(labels);
        Map<Integer, List<Integer>> indicesByGroup = new LinkedHashMap<>();
        int[] totalLabelCounts = new int[labelColumns.size()];
        for (int i = 0; i < labels.length; i++) {
            indicesByGroup.computeIfAbsent(groups[i], ignored -> new ArrayList<>()).add(i);
            totalLabelCounts[labelColumns.get(labels[i])]++;
        }
        if (indicesByGroup.size() < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to group count for stratified group k-fold, got folds="
                            + folds + ", groups=" + indicesByGroup.size());
        }

        List<Bucket> buckets = new ArrayList<>(indicesByGroup.size());
        for (List<Integer> groupIndices : indicesByGroup.values()) {
            int[] labelCounts = new int[labelColumns.size()];
            for (int index : groupIndices) {
                labelCounts[labelColumns.get(labels[index])]++;
            }
            buckets.add(new Bucket(new ArrayList<>(groupIndices), labelCounts));
        }
        Collections.shuffle(buckets, new Random(seed));
        buckets.sort(Comparator.comparingInt(Bucket::labelMassSquared).reversed());

        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        int[][] foldLabelCounts = new int[folds][labelColumns.size()];
        int[] foldSizes = new int[folds];
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }

        double targetFoldSize = dataset.size() / (double) folds;
        for (Bucket bucket : buckets) {
            int targetFold = bestStratifiedGroupFold(
                    bucket, foldLabelCounts, foldSizes, totalLabelCounts, targetFoldSize);
            validationFolds.get(targetFold).addAll(bucket.indices());
            foldSizes[targetFold] += bucket.size();
            for (int column = 0; column < bucket.labelCounts().length; column++) {
                foldLabelCounts[targetFold][column] += bucket.labelCounts()[column];
            }
        }

        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0x94D049BB133111EBL * (fold + 1L))));
        }
        return TensorDatasetSplits.foldsByValidationIndices(dataset, validationFolds);
    }

    private static Map<Integer, List<Integer>> labelGroups(int[] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            groups.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        }
        return groups;
    }

    private static void requireSameSize(int[] values, int sampleCount, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.length != sampleCount) {
            throw new IllegalArgumentException(
                    name + " and dataset must have same size, got: " + values.length + " vs " + sampleCount);
        }
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

    private static SplitCounts stratifiedThreeWayCounts(
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

    private static void rebalanceNonEmptySplit(List<Integer> trainIndices, List<Integer> validationIndices) {
        if (trainIndices.isEmpty() && !validationIndices.isEmpty()) {
            trainIndices.add(validationIndices.remove(validationIndices.size() - 1));
        }
        if (validationIndices.isEmpty() && trainIndices.size() > 1) {
            validationIndices.add(trainIndices.remove(trainIndices.size() - 1));
        }
    }

    private static void rebalanceNonEmptySplit(
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            List<Integer> testIndices) {
        moveOneIfEmpty(trainIndices, validationIndices, testIndices);
        moveOneIfEmpty(validationIndices, trainIndices, testIndices);
        moveOneIfEmpty(testIndices, trainIndices, validationIndices);
    }

    private static void moveOneIfEmpty(
            List<Integer> target,
            List<Integer> firstSource,
            List<Integer> secondSource) {
        if (!target.isEmpty()) {
            return;
        }
        List<Integer> source = firstSource.size() >= secondSource.size() ? firstSource : secondSource;
        if (source.size() <= 1) {
            source = source == firstSource ? secondSource : firstSource;
        }
        if (source.size() <= 1) {
            throw new IllegalArgumentException(
                    "dataset must contain at least three samples for train/validation/test split");
        }
        target.add(source.remove(source.size() - 1));
    }

    private static Map<Integer, Integer> labelColumns(int[] labels) {
        SortedSet<Integer> uniqueLabels = new TreeSet<>();
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            uniqueLabels.add(label);
        }
        Map<Integer, Integer> columns = new LinkedHashMap<>();
        for (int label : uniqueLabels) {
            columns.put(label, columns.size());
        }
        return columns;
    }

    private static int bestStratifiedGroupFold(
            Bucket bucket,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalLabelCounts,
            double targetFoldSize) {
        int bestFold = 0;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            double penalty = stratifiedGroupPenalty(
                    bucket, fold, foldLabelCounts, foldSizes, totalLabelCounts, targetFoldSize);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && foldSizes[fold] < foldSizes[bestFold])) {
                bestPenalty = penalty;
                bestFold = fold;
            }
        }
        return bestFold;
    }

    private static double stratifiedGroupPenalty(
            Bucket bucket,
            int candidateFold,
            int[][] foldLabelCounts,
            int[] foldSizes,
            int[] totalLabelCounts,
            double targetFoldSize) {
        double penalty = 0.0;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            double afterSize = foldSizes[fold] + (fold == candidateFold ? bucket.size() : 0);
            penalty += Math.pow(afterSize - targetFoldSize, 2) / Math.max(1.0, targetFoldSize);
            for (int column = 0; column < totalLabelCounts.length; column++) {
                double target = totalLabelCounts[column] / (double) foldSizes.length;
                double after = foldLabelCounts[fold][column]
                        + (fold == candidateFold ? bucket.labelCounts()[column] : 0);
                penalty += Math.pow(after - target, 2) / Math.max(1.0, totalLabelCounts[column]);
            }
        }
        return penalty;
    }

    private static long mixSeed(long seed, int label) {
        long value = seed ^ (0x9E3779B97F4A7C15L * (label + 0x632BE5ABL));
        value ^= (value >>> 33);
        value *= 0xff51afd7ed558ccdL;
        value ^= (value >>> 33);
        return value;
    }

    private record Bucket(List<Integer> indices, int[] labelCounts) {
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
