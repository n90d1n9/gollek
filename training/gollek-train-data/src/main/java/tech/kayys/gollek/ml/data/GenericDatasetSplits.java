package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

final class GenericDatasetSplits {
    private static final long TRAIN_SHUFFLE_SEED = 0x6A09E667F3BCC909L;
    private static final long VALIDATION_SHUFFLE_SEED = 0xBB67AE8584CAA73BL;
    private static final long TEST_SHUFFLE_SEED = 0x3C6EF372FE94F82AL;

    private GenericDatasetSplits() {
    }

    static <T> List<Dataset.Fold<T>> kFold(Dataset<T> dataset, int folds, long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireKFold(folds, dataset.size());

        List<Integer> indices = new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList());
        Collections.shuffle(indices, new Random(seed));

        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        int baseFoldSize = dataset.size() / folds;
        int remainder = dataset.size() % folds;
        int offset = 0;
        for (int fold = 0; fold < folds; fold++) {
            int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);
            validationFolds.add(new ArrayList<>(indices.subList(offset, offset + foldSize)));
            offset += foldSize;
        }
        return foldsByValidationIndices(dataset, validationFolds);
    }

    static <T> List<Dataset.Fold<T>> repeatedKFold(
            Dataset<T> dataset,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireKFold(folds, dataset.size());
        requireRepeats(repeats);
        return repeatedFolds(folds, repeats, repeat -> kFold(dataset, folds, repeatSeed(seed, repeat)));
    }

    static <T> List<Dataset.Fold<T>> timeSeriesSplit(Dataset<T> dataset, int splits) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        int validationSize = defaultTimeSeriesValidationSize(dataset.size(), splits);
        return timeSeriesSplit(dataset, splits, validationSize, 0, 0);
    }

    static <T> List<Dataset.Fold<T>> timeSeriesSplit(
            Dataset<T> dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireTimeSeriesSplit(splits, dataset.size(), validationSize, gap, maxTrainSize);

        int firstValidationStart = dataset.size() - splits * validationSize;
        List<Dataset.Fold<T>> result = new ArrayList<>(splits);
        for (int fold = 0; fold < splits; fold++) {
            int validationStart = firstValidationStart + fold * validationSize;
            int validationEnd = validationStart + validationSize;
            int trainEnd = validationStart - gap;
            int trainStart = maxTrainSize > 0 ? Math.max(0, trainEnd - maxTrainSize) : 0;

            List<Integer> trainIndices = IntStream.range(trainStart, trainEnd).boxed().toList();
            List<Integer> validationIndices = IntStream.range(validationStart, validationEnd).boxed().toList();
            result.add(new Dataset.Fold<>(
                    fold,
                    splits,
                    new Dataset.IndexedDataset<>(dataset, trainIndices),
                    new Dataset.IndexedDataset<>(dataset, validationIndices)));
        }
        return List.copyOf(result);
    }

    static <T, K> Dataset.Split<T> groupSplit(
            Dataset<T> dataset,
            Function<? super T, ? extends K> groupExtractor,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(groupExtractor, "groupExtractor must not be null");
        requireTrainFraction(trainFraction);
        List<GroupBucket<K>> groups = groupBuckets(dataset, groupExtractor);
        requireGroupCount(groups.size(), 2, "group split");

        int trainTarget = (int) Math.round(dataset.size() * trainFraction);
        trainTarget = Math.max(1, Math.min(dataset.size() - 1, trainTarget));
        List<List<GroupBucket<K>>> partitions =
                partitionBuckets(groups, new int[] {trainTarget, dataset.size() - trainTarget}, seed);
        List<Integer> trainIndices = flattenBuckets(partitions.get(0));
        List<Integer> validationIndices = flattenBuckets(partitions.get(1));
        Collections.shuffle(trainIndices, new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validationIndices, new Random(seed ^ VALIDATION_SHUFFLE_SEED));
        return new Dataset.Split<>(
                new Dataset.IndexedDataset<>(dataset, trainIndices),
                new Dataset.IndexedDataset<>(dataset, validationIndices));
    }

    static <T, K> Dataset.ThreeWaySplit<T> groupSplit(
            Dataset<T> dataset,
            Function<? super T, ? extends K> groupExtractor,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(groupExtractor, "groupExtractor must not be null");
        requireTrainValidationFractions(trainFraction, validationFraction);
        List<GroupBucket<K>> groups = groupBuckets(dataset, groupExtractor);
        requireGroupCount(groups.size(), 3, "group train/validation/test split");

        int[] targets = stratifiedThreeWayCounts(dataset.size(), trainFraction, validationFraction);
        List<List<GroupBucket<K>>> partitions = partitionBuckets(groups, targets, seed);
        List<Integer> trainIndices = flattenBuckets(partitions.get(0));
        List<Integer> validationIndices = flattenBuckets(partitions.get(1));
        List<Integer> testIndices = flattenBuckets(partitions.get(2));
        Collections.shuffle(trainIndices, new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validationIndices, new Random(seed ^ VALIDATION_SHUFFLE_SEED));
        Collections.shuffle(testIndices, new Random(seed ^ TEST_SHUFFLE_SEED));
        return new Dataset.ThreeWaySplit<>(
                new Dataset.IndexedDataset<>(dataset, trainIndices),
                new Dataset.IndexedDataset<>(dataset, validationIndices),
                new Dataset.IndexedDataset<>(dataset, testIndices));
    }

    static <T, K> List<Dataset.Fold<T>> groupKFold(
            Dataset<T> dataset,
            Function<? super T, ? extends K> groupExtractor,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(groupExtractor, "groupExtractor must not be null");
        List<GroupBucket<K>> groups = groupBuckets(dataset, groupExtractor);
        requireGroupFolds(folds, groups.size());

        List<GroupBucket<K>> ordered = shuffledBucketsBySize(groups, seed);
        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        int[] foldSizes = new int[folds];
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }
        for (GroupBucket<K> group : ordered) {
            int targetFold = smallestPartition(foldSizes);
            validationFolds.get(targetFold).addAll(group.indices());
            foldSizes[targetFold] += group.size();
        }

        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0xD1B54A32D192ED03L * (fold + 1L))));
        }
        return foldsByValidationIndices(dataset, validationFolds);
    }

    static <T, K> List<Dataset.Fold<T>> repeatedGroupKFold(
            Dataset<T> dataset,
            Function<? super T, ? extends K> groupExtractor,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(groupExtractor, "groupExtractor must not be null");
        List<GroupBucket<K>> groups = groupBuckets(dataset, groupExtractor);
        requireGroupFolds(folds, groups.size());
        requireRepeats(repeats);
        return repeatedFolds(
                folds,
                repeats,
                repeat -> groupKFold(dataset, groupExtractor, folds, repeatSeed(seed, repeat)));
    }

    static <T, K> Dataset.Split<T> stratifiedSplit(
            Dataset<T> dataset,
            Function<? super T, ? extends K> labelExtractor,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labelExtractor, "labelExtractor must not be null");
        requireTrainFraction(trainFraction);
        if (dataset.size() < 2) {
            throw new IllegalArgumentException("dataset must contain at least two samples for stratified split");
        }

        Map<K, List<Integer>> groups = labelGroups(dataset, labelExtractor);
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        int groupOrdinal = 0;
        for (Map.Entry<K, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, groupOrdinal++, entry.getKey())));
            int trainCount = stratifiedTrainCount(group.size(), trainFraction);
            trainIndices.addAll(group.subList(0, trainCount));
            validationIndices.addAll(group.subList(trainCount, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validationIndices, new Random(seed ^ VALIDATION_SHUFFLE_SEED));
        return new Dataset.Split<>(
                new Dataset.IndexedDataset<>(dataset, trainIndices),
                new Dataset.IndexedDataset<>(dataset, validationIndices));
    }

    static <T, K> Dataset.ThreeWaySplit<T> stratifiedSplit(
            Dataset<T> dataset,
            Function<? super T, ? extends K> labelExtractor,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labelExtractor, "labelExtractor must not be null");
        requireTrainValidationFractions(trainFraction, validationFraction);
        if (dataset.size() < 3) {
            throw new IllegalArgumentException(
                    "dataset must contain at least three samples for stratified train/validation/test split");
        }

        Map<K, List<Integer>> groups = labelGroups(dataset, labelExtractor);
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        List<Integer> testIndices = new ArrayList<>();
        int groupOrdinal = 0;
        for (Map.Entry<K, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, groupOrdinal++, entry.getKey())));
            int[] counts = stratifiedThreeWayCounts(group.size(), trainFraction, validationFraction);
            int trainEnd = counts[0];
            int validationEnd = trainEnd + counts[1];
            trainIndices.addAll(group.subList(0, trainEnd));
            validationIndices.addAll(group.subList(trainEnd, validationEnd));
            testIndices.addAll(group.subList(validationEnd, group.size()));
        }

        rebalanceNonEmptySplit(trainIndices, validationIndices, testIndices);
        Collections.shuffle(trainIndices, new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validationIndices, new Random(seed ^ VALIDATION_SHUFFLE_SEED));
        Collections.shuffle(testIndices, new Random(seed ^ TEST_SHUFFLE_SEED));
        return new Dataset.ThreeWaySplit<>(
                new Dataset.IndexedDataset<>(dataset, trainIndices),
                new Dataset.IndexedDataset<>(dataset, validationIndices),
                new Dataset.IndexedDataset<>(dataset, testIndices));
    }

    static <T, K> List<Dataset.Fold<T>> stratifiedKFold(
            Dataset<T> dataset,
            Function<? super T, ? extends K> labelExtractor,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labelExtractor, "labelExtractor must not be null");
        requireKFold(folds, dataset.size());

        Map<K, List<Integer>> groups = labelGroups(dataset, labelExtractor);
        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }

        int groupOrdinal = 0;
        for (Map.Entry<K, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            if (group.size() < folds) {
                throw new IllegalArgumentException(
                        "each label must contain at least folds samples for stratified k-fold; label "
                                + entry.getKey() + " has " + group.size() + ", folds=" + folds);
            }
            Collections.shuffle(group, new Random(mixSeed(seed, groupOrdinal++, entry.getKey())));
            for (int i = 0; i < group.size(); i++) {
                validationFolds.get(i % folds).add(group.get(i));
            }
        }

        for (int fold = 0; fold < folds; fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (0x9E3779B97F4A7C15L * (fold + 1L))));
        }
        return foldsByValidationIndices(dataset, validationFolds);
    }

    static <T, K> List<Dataset.Fold<T>> repeatedStratifiedKFold(
            Dataset<T> dataset,
            Function<? super T, ? extends K> labelExtractor,
            int folds,
            int repeats,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(labelExtractor, "labelExtractor must not be null");
        requireKFold(folds, dataset.size());
        requireRepeats(repeats);
        return repeatedFolds(
                folds,
                repeats,
                repeat -> stratifiedKFold(dataset, labelExtractor, folds, repeatSeed(seed, repeat)));
    }

    private static <T, K> Map<K, List<Integer>> labelGroups(
            Dataset<T> dataset,
            Function<? super T, ? extends K> labelExtractor) {
        Map<K, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            groups.computeIfAbsent(labelExtractor.apply(dataset.get(i)), ignored -> new ArrayList<>()).add(i);
        }
        return groups;
    }

    private static <T, K> List<GroupBucket<K>> groupBuckets(
            Dataset<T> dataset,
            Function<? super T, ? extends K> groupExtractor) {
        Map<K, List<Integer>> groupedIndices = labelGroups(dataset, groupExtractor);
        List<GroupBucket<K>> buckets = new ArrayList<>(groupedIndices.size());
        for (Map.Entry<K, List<Integer>> entry : groupedIndices.entrySet()) {
            buckets.add(new GroupBucket<>(entry.getKey(), List.copyOf(entry.getValue())));
        }
        return buckets;
    }

    private static <K> List<List<GroupBucket<K>>> partitionBuckets(
            List<GroupBucket<K>> groups,
            int[] targetSizes,
            long seed) {
        List<GroupBucket<K>> ordered = shuffledBucketsBySize(groups, seed);
        List<List<GroupBucket<K>>> partitions = new ArrayList<>(targetSizes.length);
        int[] sizes = new int[targetSizes.length];
        for (int i = 0; i < targetSizes.length; i++) {
            partitions.add(new ArrayList<>());
        }
        for (GroupBucket<K> group : ordered) {
            int partition = bestSizedPartition(group, sizes, targetSizes);
            partitions.get(partition).add(group);
            sizes[partition] += group.size();
        }
        rebalanceEmptyBucketPartitions(partitions, sizes);
        return partitions;
    }

    private static <K> List<GroupBucket<K>> shuffledBucketsBySize(List<GroupBucket<K>> groups, long seed) {
        List<GroupBucket<K>> ordered = new ArrayList<>(groups);
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> Integer.compare(right.size(), left.size()));
        return ordered;
    }

    private static <K> int bestSizedPartition(GroupBucket<K> group, int[] sizes, int[] targetSizes) {
        int best = 0;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int candidate = 0; candidate < sizes.length; candidate++) {
            double penalty = 0.0;
            for (int partition = 0; partition < sizes.length; partition++) {
                int after = sizes[partition] + (partition == candidate ? group.size() : 0);
                penalty += Math.pow(after - targetSizes[partition], 2) / Math.max(1.0, targetSizes[partition]);
            }
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && sizes[candidate] < sizes[best])) {
                bestPenalty = penalty;
                best = candidate;
            }
        }
        return best;
    }

    private static int smallestPartition(int[] sizes) {
        int best = 0;
        for (int i = 1; i < sizes.length; i++) {
            if (sizes[i] < sizes[best]) {
                best = i;
            }
        }
        return best;
    }

    private static <K> void rebalanceEmptyBucketPartitions(
            List<List<GroupBucket<K>>> partitions,
            int[] sizes) {
        for (int target = 0; target < partitions.size(); target++) {
            if (!partitions.get(target).isEmpty()) {
                continue;
            }
            int source = largestMovablePartition(partitions, sizes);
            if (source < 0) {
                throw new IllegalArgumentException("not enough groups to create non-empty partitions");
            }
            GroupBucket<K> moved = partitions.get(source).remove(partitions.get(source).size() - 1);
            sizes[source] -= moved.size();
            partitions.get(target).add(moved);
            sizes[target] += moved.size();
        }
    }

    private static <K> int largestMovablePartition(List<List<GroupBucket<K>>> partitions, int[] sizes) {
        int best = -1;
        for (int i = 0; i < partitions.size(); i++) {
            if (partitions.get(i).size() <= 1) {
                continue;
            }
            if (best < 0 || sizes[i] > sizes[best]) {
                best = i;
            }
        }
        return best;
    }

    private static <K> List<Integer> flattenBuckets(List<GroupBucket<K>> groups) {
        List<Integer> indices = new ArrayList<>();
        for (GroupBucket<K> group : groups) {
            indices.addAll(group.indices());
        }
        return indices;
    }

    private static <T> List<Dataset.Fold<T>> foldsByValidationIndices(
            Dataset<T> dataset,
            List<List<Integer>> validationFolds) {
        List<Dataset.Fold<T>> result = new ArrayList<>(validationFolds.size());
        for (int fold = 0; fold < validationFolds.size(); fold++) {
            List<Integer> validationIndices = validationFolds.get(fold);
            boolean[] validationMask = new boolean[dataset.size()];
            for (int index : validationIndices) {
                validationMask[index] = true;
            }

            List<Integer> trainIndices = new ArrayList<>(dataset.size() - validationIndices.size());
            for (int index = 0; index < dataset.size(); index++) {
                if (!validationMask[index]) {
                    trainIndices.add(index);
                }
            }
            result.add(new Dataset.Fold<>(
                    fold,
                    validationFolds.size(),
                    new Dataset.IndexedDataset<>(dataset, trainIndices),
                    new Dataset.IndexedDataset<>(dataset, validationIndices)));
        }
        return List.copyOf(result);
    }

    private static <T> List<Dataset.Fold<T>> repeatedFolds(
            int folds,
            int repeats,
            FoldFactory<T> foldFactory) {
        int totalFoldCount = totalRepeatedFoldCount(folds, repeats);
        List<Dataset.Fold<T>> result = new ArrayList<>(totalFoldCount);
        for (int repeat = 0; repeat < repeats; repeat++) {
            for (Dataset.Fold<T> fold : foldFactory.create(repeat)) {
                result.add(new Dataset.Fold<>(
                        repeat * folds + fold.foldIndex(),
                        totalFoldCount,
                        fold.train(),
                        fold.validation()));
            }
        }
        return List.copyOf(result);
    }

    private static void requireTrainFraction(double trainFraction) {
        if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
        }
    }

    private static void requireTrainValidationFractions(double trainFraction, double validationFraction) {
        if (!Double.isFinite(trainFraction) || !Double.isFinite(validationFraction)) {
            throw new IllegalArgumentException("split fractions must be finite");
        }
        if (trainFraction <= 0.0 || validationFraction <= 0.0 || trainFraction + validationFraction >= 1.0) {
            throw new IllegalArgumentException(
                    "trainFraction and validationFraction must be positive and leave a positive test fraction");
        }
    }

    private static void requireGroupCount(int groupCount, int required, String splitName) {
        if (groupCount < required) {
            throw new IllegalArgumentException(splitName + " requires at least " + required
                    + " groups, got: " + groupCount);
        }
    }

    private static void requireGroupFolds(int folds, int groupCount) {
        if (folds < 2) {
            throw new IllegalArgumentException("folds must be at least 2, got: " + folds);
        }
        if (groupCount < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to group count, got folds="
                            + folds + ", groups=" + groupCount);
        }
    }

    private static int defaultTimeSeriesValidationSize(int sampleCount, int splits) {
        if (splits < 1) {
            throw new IllegalArgumentException("splits must be at least 1, got: " + splits);
        }
        if (sampleCount < splits + 1) {
            throw new IllegalArgumentException(
                    "dataset must contain at least splits + 1 samples for time-series split, got splits="
                            + splits + ", size=" + sampleCount);
        }
        int validationSize = sampleCount / (splits + 1);
        if (validationSize < 1) {
            throw new IllegalArgumentException("time-series validationSize must be at least 1");
        }
        return validationSize;
    }

    private static void requireTimeSeriesSplit(
            int splits,
            int sampleCount,
            int validationSize,
            int gap,
            int maxTrainSize) {
        if (splits < 1) {
            throw new IllegalArgumentException("splits must be at least 1, got: " + splits);
        }
        if (validationSize < 1) {
            throw new IllegalArgumentException("validationSize must be at least 1, got: " + validationSize);
        }
        if (gap < 0) {
            throw new IllegalArgumentException("gap must be non-negative, got: " + gap);
        }
        if (maxTrainSize < 0) {
            throw new IllegalArgumentException("maxTrainSize must be non-negative, got: " + maxTrainSize);
        }
        long validationTotal = (long) splits * validationSize;
        if (validationTotal >= sampleCount) {
            throw new IllegalArgumentException(
                    "time-series validation windows must leave at least one training sample, got splits="
                            + splits + ", validationSize=" + validationSize + ", size=" + sampleCount);
        }
        int firstValidationStart = sampleCount - (int) validationTotal;
        if (firstValidationStart <= gap) {
            throw new IllegalArgumentException(
                    "gap leaves no training samples before the first validation window, got gap="
                            + gap + ", firstValidationStart=" + firstValidationStart);
        }
    }

    private static void requireKFold(int folds, int sampleCount) {
        if (folds < 2) {
            throw new IllegalArgumentException("folds must be at least 2, got: " + folds);
        }
        if (sampleCount < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to dataset size, got folds="
                            + folds + ", size=" + sampleCount);
        }
    }

    private static void requireRepeats(int repeats) {
        if (repeats < 1) {
            throw new IllegalArgumentException("repeats must be at least 1, got: " + repeats);
        }
    }

    private static int totalRepeatedFoldCount(int folds, int repeats) {
        try {
            return Math.multiplyExact(folds, repeats);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException(
                    "folds * repeats is too large, got folds=" + folds + ", repeats=" + repeats,
                    error);
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

    private static int[] stratifiedThreeWayCounts(
            int groupSize,
            double trainFraction,
            double validationFraction) {
        if (groupSize <= 0) {
            return new int[] {0, 0, 0};
        }
        if (groupSize == 1) {
            return new int[] {1, 0, 0};
        }
        if (groupSize == 2) {
            return new int[] {1, 1, 0};
        }
        int trainCount = (int) Math.round(groupSize * trainFraction);
        int validationCount = (int) Math.round(groupSize * validationFraction);
        trainCount = Math.max(1, Math.min(groupSize - 2, trainCount));
        validationCount = Math.max(1, Math.min(groupSize - trainCount - 1, validationCount));
        return new int[] {trainCount, validationCount, groupSize - trainCount - validationCount};
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

    private static long mixSeed(long seed, int groupOrdinal, Object label) {
        long value = seed ^ (0x9E3779B97F4A7C15L * (groupOrdinal + 1L));
        value ^= ((long) Objects.hashCode(label)) * 0xD1B54A32D192ED03L;
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        return value;
    }

    private static long repeatSeed(long seed, int repeat) {
        long value = seed ^ (0xD1B54A32D192ED03L * (repeat + 1L));
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        return value;
    }

    @FunctionalInterface
    private interface FoldFactory<T> {
        List<Dataset.Fold<T>> create(int repeat);
    }

    private record GroupBucket<K>(K key, List<Integer> indices) {
        int size() {
            return indices.size();
        }
    }
}
