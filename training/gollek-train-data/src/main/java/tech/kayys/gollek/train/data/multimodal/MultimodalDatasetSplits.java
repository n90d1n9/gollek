package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

/**
 * Split helpers for multimodal datasets.
 */
public final class MultimodalDatasetSplits {
    private static final long TRAIN_SHUFFLE_SEED = 0x6A09E667F3BCC909L;
    private static final long VALIDATION_SHUFFLE_SEED = 0xBB67AE8584CAA73BL;
    private static final long FOLD_SHUFFLE_SEED = 0x3C6EF372FE94F82AL;

    private MultimodalDatasetSplits() {
    }

    public static Dataset.Split<List<MultimodalContent>> stratifiedBySignature(
            Dataset<? extends List<MultimodalContent>> dataset,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireTrainFraction(trainFraction);
        if (dataset.size() < 2) {
            throw new IllegalArgumentException("dataset must contain at least two samples for stratified split");
        }

        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int index = 0; index < dataset.size(); index++) {
            groups.computeIfAbsent(signature(dataset.get(index)), ignored -> new ArrayList<>()).add(index);
        }

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        int groupOrdinal = 0;
        for (Map.Entry<String, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, new Random(mixSeed(seed, groupOrdinal++, entry.getKey())));
            int trainCount = stratifiedTrainCount(group.size(), trainFraction);
            trainIndices.addAll(group.subList(0, trainCount));
            validationIndices.addAll(group.subList(trainCount, group.size()));
        }
        rebalanceNonEmpty(trainIndices, validationIndices);
        shufflePartitions(trainIndices, validationIndices, seed);
        return split(dataset, trainIndices, validationIndices);
    }

    public static Dataset.Split<List<MultimodalContent>> groupedBySourcePath(
            Dataset<? extends List<MultimodalContent>> dataset,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireTrainFraction(trainFraction);
        if (dataset.size() < 2) {
            throw new IllegalArgumentException("dataset must contain at least two samples for source-grouped split");
        }

        List<Group> groups = sourcePathGroups(dataset);
        if (groups.size() < 2) {
            throw new IllegalArgumentException("source-grouped split requires at least two independent source groups");
        }
        int trainTarget = trainTarget(dataset.size(), trainFraction);

        List<Group> ordered = new ArrayList<>(groups);
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> Integer.compare(right.size(), left.size()));

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        int trainSize = 0;
        for (Group group : ordered) {
            int trainPenalty = Math.abs((trainSize + group.size()) - trainTarget);
            int validationPenalty = Math.abs(trainSize - trainTarget);
            if (trainPenalty <= validationPenalty && trainSize < trainTarget) {
                trainIndices.addAll(group.indices());
                trainSize += group.size();
            } else {
                validationIndices.addAll(group.indices());
            }
        }
        rebalanceNonEmpty(trainIndices, validationIndices);
        shufflePartitions(trainIndices, validationIndices, seed);
        return split(dataset, trainIndices, validationIndices);
    }

    public static Dataset.Split<List<MultimodalContent>> stratifiedGroupedBySourcePath(
            Dataset<? extends List<MultimodalContent>> dataset,
            double trainFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireTrainFraction(trainFraction);
        if (dataset.size() < 2) {
            throw new IllegalArgumentException(
                    "dataset must contain at least two samples for source-grouped stratified split");
        }

        List<Group> groups = sourcePathGroups(dataset);
        if (groups.size() < 2) {
            throw new IllegalArgumentException(
                    "source-grouped stratified split requires at least two independent source groups");
        }

        List<GroupProfile> ordered = new ArrayList<>(groups.size());
        Map<String, Integer> totalSignatureCounts = new LinkedHashMap<>();
        for (Group group : groups) {
            GroupProfile profile = profile(dataset, group);
            ordered.add(profile);
            profile.signatureCounts().forEach((signature, count) ->
                    totalSignatureCounts.merge(signature, count, Integer::sum));
        }
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> {
            int bySize = Integer.compare(right.size(), left.size());
            return bySize != 0 ? bySize : Integer.compare(right.signatureCounts().size(), left.signatureCounts().size());
        });

        int trainTarget = trainTarget(dataset.size(), trainFraction);
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validationIndices = new ArrayList<>();
        Map<String, Integer> trainSignatureCounts = new LinkedHashMap<>();
        Map<String, Integer> validationSignatureCounts = new LinkedHashMap<>();
        int trainGroupCount = 0;
        int validationGroupCount = 0;

        for (int index = 0; index < ordered.size(); index++) {
            GroupProfile profile = ordered.get(index);
            boolean lastGroup = index == ordered.size() - 1;
            SplitSide side;
            if (lastGroup && trainGroupCount == 0 && validationGroupCount > 0) {
                side = SplitSide.TRAIN;
            } else if (lastGroup && validationGroupCount == 0 && trainGroupCount > 0) {
                side = SplitSide.VALIDATION;
            } else {
                side = bestStratifiedGroupedSplitSide(
                        profile,
                        trainSignatureCounts,
                        validationSignatureCounts,
                        trainIndices.size(),
                        validationIndices.size(),
                        totalSignatureCounts,
                        dataset.size(),
                        trainTarget,
                        trainFraction);
            }

            if (side == SplitSide.TRAIN) {
                trainIndices.addAll(profile.group().indices());
                addSignatureCounts(trainSignatureCounts, profile.signatureCounts());
                trainGroupCount++;
            } else {
                validationIndices.addAll(profile.group().indices());
                addSignatureCounts(validationSignatureCounts, profile.signatureCounts());
                validationGroupCount++;
            }
        }

        if (trainIndices.isEmpty() || validationIndices.isEmpty()) {
            throw new IllegalArgumentException("split could not create non-empty train and validation partitions");
        }
        shufflePartitions(trainIndices, validationIndices, seed);
        return split(dataset, trainIndices, validationIndices);
    }

    public static Dataset.ThreeWaySplit<List<MultimodalContent>> stratifiedGroupedThreeWayBySourcePath(
            Dataset<? extends List<MultimodalContent>> dataset,
            double trainFraction,
            double validationFraction,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireThreeWayFractions(trainFraction, validationFraction);
        if (dataset.size() < 3) {
            throw new IllegalArgumentException(
                    "dataset must contain at least three samples for source-grouped stratified three-way split");
        }

        List<Group> groups = sourcePathGroups(dataset);
        if (groups.size() < 3) {
            throw new IllegalArgumentException(
                    "source-grouped stratified three-way split requires at least three independent source groups");
        }

        List<GroupProfile> ordered = new ArrayList<>(groups.size());
        Map<String, Integer> totalSignatureCounts = new LinkedHashMap<>();
        for (Group group : groups) {
            GroupProfile profile = profile(dataset, group);
            ordered.add(profile);
            profile.signatureCounts().forEach((signature, count) ->
                    totalSignatureCounts.merge(signature, count, Integer::sum));
        }
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> {
            int bySize = Integer.compare(right.size(), left.size());
            return bySize != 0 ? bySize : Integer.compare(right.signatureCounts().size(), left.signatureCounts().size());
        });

        int[] targets = threeWayTargets(dataset.size(), trainFraction, validationFraction);
        double testFraction = 1.0 - trainFraction - validationFraction;
        List<SplitPartition> partitions = List.of(
                new SplitPartition(SplitPart.TRAIN, targets[0], trainFraction),
                new SplitPartition(SplitPart.VALIDATION, targets[1], validationFraction),
                new SplitPartition(SplitPart.TEST, targets[2], testFraction));

        for (int index = 0; index < ordered.size(); index++) {
            GroupProfile profile = ordered.get(index);
            int remainingGroups = ordered.size() - index;
            SplitPartition target = remainingGroups == emptyPartitionCount(partitions)
                    ? firstEmptyPartition(partitions)
                    : bestThreeWayPartition(profile, partitions, totalSignatureCounts);
            target.add(profile);
        }

        SplitPartition train = partitions.get(0);
        SplitPartition validation = partitions.get(1);
        SplitPartition test = partitions.get(2);
        if (train.indices().isEmpty() || validation.indices().isEmpty() || test.indices().isEmpty()) {
            throw new IllegalArgumentException("split could not create non-empty train, validation, and test partitions");
        }
        Collections.shuffle(train.indices(), new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validation.indices(), new Random(seed ^ VALIDATION_SHUFFLE_SEED));
        Collections.shuffle(test.indices(), new Random(seed ^ FOLD_SHUFFLE_SEED));
        return new Dataset.ThreeWaySplit<>(
                new IndexedMultimodalDataset(dataset, train.indices()),
                new IndexedMultimodalDataset(dataset, validation.indices()),
                new IndexedMultimodalDataset(dataset, test.indices()));
    }

    public static List<Dataset.Fold<List<MultimodalContent>>> stratifiedKFoldBySignature(
            Dataset<? extends List<MultimodalContent>> dataset,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireFolds(folds, dataset.size());

        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int index = 0; index < dataset.size(); index++) {
            groups.computeIfAbsent(signature(dataset.get(index)), ignored -> new ArrayList<>()).add(index);
        }

        List<List<Integer>> validationFolds = emptyFoldBuckets(folds);
        int groupOrdinal = 0;
        for (Map.Entry<String, List<Integer>> entry : groups.entrySet()) {
            List<Integer> group = new ArrayList<>(entry.getValue());
            if (group.size() < folds) {
                throw new IllegalArgumentException(
                        "each modality signature must contain at least folds samples; signature "
                                + entry.getKey() + " has " + group.size() + ", folds=" + folds);
            }
            Collections.shuffle(group, new Random(mixSeed(seed, groupOrdinal++, entry.getKey())));
            for (int index = 0; index < group.size(); index++) {
                validationFolds.get(index % folds).add(group.get(index));
            }
        }
        shuffleValidationFolds(validationFolds, seed);
        return foldsByValidationIndices(dataset, validationFolds);
    }

    public static List<Dataset.Fold<List<MultimodalContent>>> groupedKFoldBySourcePath(
            Dataset<? extends List<MultimodalContent>> dataset,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireFolds(folds, dataset.size());
        List<Group> groups = sourcePathGroups(dataset);
        if (groups.size() < folds) {
            throw new IllegalArgumentException(
                    "source-grouped k-fold requires at least folds independent source groups, got groups="
                            + groups.size() + ", folds=" + folds);
        }

        List<Group> ordered = new ArrayList<>(groups);
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> Integer.compare(right.size(), left.size()));

        List<List<Integer>> validationFolds = emptyFoldBuckets(folds);
        int[] foldSizes = new int[folds];
        for (Group group : ordered) {
            int targetFold = smallestFold(foldSizes);
            validationFolds.get(targetFold).addAll(group.indices());
            foldSizes[targetFold] += group.size();
        }
        shuffleValidationFolds(validationFolds, seed);
        return foldsByValidationIndices(dataset, validationFolds);
    }

    public static List<Dataset.Fold<List<MultimodalContent>>> stratifiedGroupedKFoldBySourcePath(
            Dataset<? extends List<MultimodalContent>> dataset,
            int folds,
            long seed) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        requireFolds(folds, dataset.size());
        List<Group> groups = sourcePathGroups(dataset);
        if (groups.size() < folds) {
            throw new IllegalArgumentException(
                    "source-grouped stratified k-fold requires at least folds independent source groups, got groups="
                            + groups.size() + ", folds=" + folds);
        }

        List<GroupProfile> ordered = new ArrayList<>(groups.size());
        Map<String, Integer> totalSignatureCounts = new LinkedHashMap<>();
        for (Group group : groups) {
            GroupProfile profile = profile(dataset, group);
            ordered.add(profile);
            profile.signatureCounts().forEach((signature, count) ->
                    totalSignatureCounts.merge(signature, count, Integer::sum));
        }
        Collections.shuffle(ordered, new Random(seed));
        ordered.sort((left, right) -> {
            int bySize = Integer.compare(right.size(), left.size());
            return bySize != 0 ? bySize : Integer.compare(right.signatureCounts().size(), left.signatureCounts().size());
        });

        List<List<Integer>> validationFolds = emptyFoldBuckets(folds);
        List<Map<String, Integer>> foldSignatureCounts = emptySignatureBuckets(folds);
        int[] foldSizes = new int[folds];
        int[] foldGroupCounts = new int[folds];
        for (GroupProfile profile : ordered) {
            int targetFold = firstEmptyFold(foldGroupCounts);
            if (targetFold < 0) {
                targetFold = bestStratifiedGroupedFold(
                        profile,
                        foldSignatureCounts,
                        foldSizes,
                        totalSignatureCounts,
                        dataset.size());
            }
            validationFolds.get(targetFold).addAll(profile.group().indices());
            Map<String, Integer> signatureBucket = foldSignatureCounts.get(targetFold);
            profile.signatureCounts().forEach((signature, count) ->
                    signatureBucket.merge(signature, count, Integer::sum));
            foldSizes[targetFold] += profile.size();
            foldGroupCounts[targetFold]++;
        }
        shuffleValidationFolds(validationFolds, seed);
        return foldsByValidationIndices(dataset, validationFolds);
    }

    public static Set<String> overlappingSourcePaths(Dataset.Split<List<MultimodalContent>> split) {
        Objects.requireNonNull(split, "split must not be null");
        Set<String> train = sourcePaths(split.train());
        Set<String> validation = sourcePaths(split.validation());
        train.retainAll(validation);
        return Collections.unmodifiableSet(new TreeSet<>(train));
    }

    static List<Integer> sourceIndices(Dataset<List<MultimodalContent>> dataset) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        if (dataset instanceof IndexedMultimodalDataset indexed) {
            return indexed.sourceIndices();
        }
        throw new IllegalArgumentException(
                "dataset partition does not expose source indices; build it with MultimodalDatasetSplits");
    }

    static Dataset.ThreeWaySplit<List<MultimodalContent>> threeWayByIndices(
            Dataset<? extends List<MultimodalContent>> dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            List<Integer> testIndices) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return new Dataset.ThreeWaySplit<>(
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, trainIndices, "trainIndices")),
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, validationIndices, "validationIndices")),
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, testIndices, "testIndices")));
    }

    static Dataset.Split<List<MultimodalContent>> splitByIndices(
            Dataset<? extends List<MultimodalContent>> dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return new Dataset.Split<>(
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, trainIndices, "trainIndices")),
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, validationIndices, "validationIndices")));
    }

    static Dataset.Fold<List<MultimodalContent>> foldByIndices(
            Dataset<? extends List<MultimodalContent>> dataset,
            int foldIndex,
            int foldCount,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return new Dataset.Fold<>(
                foldIndex,
                foldCount,
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, trainIndices, "trainIndices")),
                new IndexedMultimodalDataset(dataset, validateIndices(dataset, validationIndices, "validationIndices")));
    }

    public static String signature(List<MultimodalContent> sample) {
        List<String> names = MultimodalDatasetSupport.modalitySet(sample).stream()
                .map(Enum::name)
                .sorted()
                .toList();
        return String.join("+", names);
    }

    private static List<Group> sourcePathGroups(Dataset<? extends List<MultimodalContent>> dataset) {
        Map<String, List<Integer>> pathToIndices = new LinkedHashMap<>();
        List<Set<String>> samplePaths = new ArrayList<>(dataset.size());
        for (int index = 0; index < dataset.size(); index++) {
            Set<String> paths = sourcePaths(dataset.get(index));
            samplePaths.add(paths);
            for (String path : paths) {
                pathToIndices.computeIfAbsent(path, ignored -> new ArrayList<>()).add(index);
            }
        }

        boolean[] visited = new boolean[dataset.size()];
        List<Group> groups = new ArrayList<>();
        for (int start = 0; start < dataset.size(); start++) {
            if (visited[start]) {
                continue;
            }
            List<Integer> indices = new ArrayList<>();
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            visited[start] = true;
            while (!queue.isEmpty()) {
                int current = queue.removeFirst();
                indices.add(current);
                for (String path : samplePaths.get(current)) {
                    for (int linked : pathToIndices.getOrDefault(path, List.of())) {
                        if (!visited[linked]) {
                            visited[linked] = true;
                            queue.addLast(linked);
                        }
                    }
                }
            }
            groups.add(new Group(indices));
        }
        return groups;
    }

    private static Set<String> sourcePaths(Dataset<? extends List<MultimodalContent>> dataset) {
        Set<String> paths = new LinkedHashSet<>();
        for (int index = 0; index < dataset.size(); index++) {
            paths.addAll(sourcePaths(dataset.get(index)));
        }
        return paths;
    }

    static Set<String> sourcePaths(List<MultimodalContent> sample) {
        Set<String> paths = new LinkedHashSet<>();
        for (MultimodalContent content : MultimodalDatasetSupport.immutableSample(sample, "sample")) {
            if (content.getModality() == ModalityType.TEXT) {
                continue;
            }
            Map<String, Object> metadata = content.getMetadata();
            Object value = metadata == null ? null : metadata.get("sourcePath");
            if (value != null && !value.toString().isBlank()) {
                paths.add(value.toString().trim());
            } else if (content.isRemote()) {
                paths.add(content.getUri().trim());
            }
        }
        return paths;
    }

    private static List<Integer> validateIndices(
            Dataset<? extends List<MultimodalContent>> dataset,
            List<Integer> indices,
            String owner) {
        Objects.requireNonNull(indices, owner + " must not be null");
        List<Integer> copy = new ArrayList<>(indices.size());
        for (Integer index : indices) {
            int value = Objects.requireNonNull(index, owner + " must not contain null indices");
            if (value < 0 || value >= dataset.size()) {
                throw new IllegalArgumentException(
                        owner + " contains out-of-range index " + value + " for dataset size " + dataset.size());
            }
            copy.add(value);
        }
        return copy;
    }

    private static Dataset.Split<List<MultimodalContent>> split(
            Dataset<? extends List<MultimodalContent>> dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        return new Dataset.Split<>(
                new IndexedMultimodalDataset(dataset, trainIndices),
                new IndexedMultimodalDataset(dataset, validationIndices));
    }

    private static void shufflePartitions(List<Integer> trainIndices, List<Integer> validationIndices, long seed) {
        Collections.shuffle(trainIndices, new Random(seed ^ TRAIN_SHUFFLE_SEED));
        Collections.shuffle(validationIndices, new Random(seed ^ VALIDATION_SHUFFLE_SEED));
    }

    private static int stratifiedTrainCount(int size, double trainFraction) {
        if (size <= 1) {
            return size;
        }
        return trainTarget(size, trainFraction);
    }

    private static int trainTarget(int size, double trainFraction) {
        int count = (int) Math.round(size * trainFraction);
        return Math.max(1, Math.min(size - 1, count));
    }

    private static void rebalanceNonEmpty(List<Integer> trainIndices, List<Integer> validationIndices) {
        if (trainIndices.isEmpty() && validationIndices.size() > 1) {
            trainIndices.add(validationIndices.remove(validationIndices.size() - 1));
        }
        if (validationIndices.isEmpty() && trainIndices.size() > 1) {
            validationIndices.add(trainIndices.remove(trainIndices.size() - 1));
        }
        if (trainIndices.isEmpty() || validationIndices.isEmpty()) {
            throw new IllegalArgumentException("split could not create non-empty train and validation partitions");
        }
    }

    private static void requireTrainFraction(double trainFraction) {
        if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
        }
    }

    private static void requireThreeWayFractions(double trainFraction, double validationFraction) {
        requireTrainFraction(trainFraction);
        if (!Double.isFinite(validationFraction) || validationFraction <= 0.0 || validationFraction >= 1.0) {
            throw new IllegalArgumentException("validationFraction must be finite and between 0 and 1");
        }
        if (trainFraction + validationFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction + validationFraction must be less than 1");
        }
    }

    private static int[] threeWayTargets(int size, double trainFraction, double validationFraction) {
        int trainTarget = Math.max(1, (int) Math.round(size * trainFraction));
        trainTarget = Math.min(trainTarget, size - 2);
        int validationTarget = Math.max(1, (int) Math.round(size * validationFraction));
        validationTarget = Math.min(validationTarget, size - trainTarget - 1);
        int testTarget = size - trainTarget - validationTarget;
        return new int[] {trainTarget, validationTarget, testTarget};
    }

    private static void requireFolds(int folds, int sampleCount) {
        if (folds < 2) {
            throw new IllegalArgumentException("folds must be at least 2, got: " + folds);
        }
        if (sampleCount < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to dataset size, got folds="
                            + folds + ", size=" + sampleCount);
        }
    }

    private static List<List<Integer>> emptyFoldBuckets(int folds) {
        List<List<Integer>> validationFolds = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            validationFolds.add(new ArrayList<>());
        }
        return validationFolds;
    }

    private static void shuffleValidationFolds(List<List<Integer>> validationFolds, long seed) {
        for (int fold = 0; fold < validationFolds.size(); fold++) {
            Collections.shuffle(validationFolds.get(fold), new Random(seed ^ (FOLD_SHUFFLE_SEED * (fold + 1L))));
        }
    }

    private static int smallestFold(int[] foldSizes) {
        int best = 0;
        for (int fold = 1; fold < foldSizes.length; fold++) {
            if (foldSizes[fold] < foldSizes[best]) {
                best = fold;
            }
        }
        return best;
    }

    private static List<Map<String, Integer>> emptySignatureBuckets(int folds) {
        List<Map<String, Integer>> buckets = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            buckets.add(new LinkedHashMap<>());
        }
        return buckets;
    }

    private static int firstEmptyFold(int[] foldGroupCounts) {
        for (int fold = 0; fold < foldGroupCounts.length; fold++) {
            if (foldGroupCounts[fold] == 0) {
                return fold;
            }
        }
        return -1;
    }

    private static GroupProfile profile(Dataset<? extends List<MultimodalContent>> dataset, Group group) {
        Map<String, Integer> signatureCounts = new LinkedHashMap<>();
        for (int index : group.indices()) {
            signatureCounts.merge(signature(dataset.get(index)), 1, Integer::sum);
        }
        return new GroupProfile(group, signatureCounts);
    }

    private static int bestStratifiedGroupedFold(
            GroupProfile profile,
            List<Map<String, Integer>> foldSignatureCounts,
            int[] foldSizes,
            Map<String, Integer> totalSignatureCounts,
            int totalSamples) {
        int best = 0;
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (int fold = 0; fold < foldSizes.length; fold++) {
            double penalty = stratifiedGroupedPenalty(
                    profile,
                    foldSignatureCounts.get(fold),
                    foldSizes[fold],
                    totalSignatureCounts,
                    totalSamples,
                    foldSizes.length);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && foldSizes[fold] < foldSizes[best])) {
                bestPenalty = penalty;
                best = fold;
            }
        }
        return best;
    }

    private static double stratifiedGroupedPenalty(
            GroupProfile profile,
            Map<String, Integer> foldSignatureCounts,
            int foldSize,
            Map<String, Integer> totalSignatureCounts,
            int totalSamples,
            int folds) {
        double penalty = 0.0;
        for (Map.Entry<String, Integer> entry : totalSignatureCounts.entrySet()) {
            double target = (double) entry.getValue() / folds;
            int after = foldSignatureCounts.getOrDefault(entry.getKey(), 0)
                    + profile.signatureCounts().getOrDefault(entry.getKey(), 0);
            double diff = after - target;
            penalty += (diff * diff) / Math.max(1.0, target);
        }
        double targetSize = (double) totalSamples / folds;
        double sizeDiff = foldSize + profile.size() - targetSize;
        penalty += (sizeDiff * sizeDiff) / Math.max(1.0, targetSize);
        return penalty;
    }

    private static SplitSide bestStratifiedGroupedSplitSide(
            GroupProfile profile,
            Map<String, Integer> trainSignatureCounts,
            Map<String, Integer> validationSignatureCounts,
            int trainSize,
            int validationSize,
            Map<String, Integer> totalSignatureCounts,
            int totalSamples,
            int trainTarget,
            double trainFraction) {
        double trainPenalty = stratifiedGroupedSplitPenalty(
                profile,
                SplitSide.TRAIN,
                trainSignatureCounts,
                validationSignatureCounts,
                trainSize,
                validationSize,
                totalSignatureCounts,
                totalSamples,
                trainTarget,
                trainFraction);
        double validationPenalty = stratifiedGroupedSplitPenalty(
                profile,
                SplitSide.VALIDATION,
                trainSignatureCounts,
                validationSignatureCounts,
                trainSize,
                validationSize,
                totalSignatureCounts,
                totalSamples,
                trainTarget,
                trainFraction);
        if (trainPenalty + 1e-9 < validationPenalty) {
            return SplitSide.TRAIN;
        }
        if (validationPenalty + 1e-9 < trainPenalty) {
            return SplitSide.VALIDATION;
        }
        return trainSize < trainTarget ? SplitSide.TRAIN : SplitSide.VALIDATION;
    }

    private static double stratifiedGroupedSplitPenalty(
            GroupProfile profile,
            SplitSide side,
            Map<String, Integer> trainSignatureCounts,
            Map<String, Integer> validationSignatureCounts,
            int trainSize,
            int validationSize,
            Map<String, Integer> totalSignatureCounts,
            int totalSamples,
            int trainTarget,
            double trainFraction) {
        int validationTarget = totalSamples - trainTarget;
        int trainAfter = trainSize + (side == SplitSide.TRAIN ? profile.size() : 0);
        int validationAfter = validationSize + (side == SplitSide.VALIDATION ? profile.size() : 0);
        double trainSizeDiff = trainAfter - trainTarget;
        double validationSizeDiff = validationAfter - validationTarget;
        double penalty = (trainSizeDiff * trainSizeDiff) / Math.max(1.0, trainTarget)
                + (validationSizeDiff * validationSizeDiff) / Math.max(1.0, validationTarget);

        for (Map.Entry<String, Integer> entry : totalSignatureCounts.entrySet()) {
            String signature = entry.getKey();
            int profileCount = profile.signatureCounts().getOrDefault(signature, 0);
            int trainSignatureAfter = trainSignatureCounts.getOrDefault(signature, 0)
                    + (side == SplitSide.TRAIN ? profileCount : 0);
            int validationSignatureAfter = validationSignatureCounts.getOrDefault(signature, 0)
                    + (side == SplitSide.VALIDATION ? profileCount : 0);
            double trainSignatureTarget = entry.getValue() * trainFraction;
            double validationSignatureTarget = entry.getValue() - trainSignatureTarget;
            double trainSignatureDiff = trainSignatureAfter - trainSignatureTarget;
            double validationSignatureDiff = validationSignatureAfter - validationSignatureTarget;
            penalty += (trainSignatureDiff * trainSignatureDiff) / Math.max(1.0, trainSignatureTarget);
            penalty += (validationSignatureDiff * validationSignatureDiff) / Math.max(1.0, validationSignatureTarget);
        }
        return penalty;
    }

    private static void addSignatureCounts(Map<String, Integer> target, Map<String, Integer> source) {
        source.forEach((signature, count) -> target.merge(signature, count, Integer::sum));
    }

    private static int emptyPartitionCount(List<SplitPartition> partitions) {
        int empty = 0;
        for (SplitPartition partition : partitions) {
            if (partition.groupCount() == 0) {
                empty++;
            }
        }
        return empty;
    }

    private static SplitPartition firstEmptyPartition(List<SplitPartition> partitions) {
        for (SplitPartition partition : partitions) {
            if (partition.groupCount() == 0) {
                return partition;
            }
        }
        throw new IllegalStateException("no empty partition available");
    }

    private static SplitPartition bestThreeWayPartition(
            GroupProfile profile,
            List<SplitPartition> partitions,
            Map<String, Integer> totalSignatureCounts) {
        SplitPartition best = partitions.get(0);
        double bestPenalty = Double.POSITIVE_INFINITY;
        for (SplitPartition candidate : partitions) {
            double penalty = threeWayPenalty(profile, candidate, partitions, totalSignatureCounts);
            if (penalty + 1e-9 < bestPenalty
                    || (Math.abs(penalty - bestPenalty) <= 1e-9 && candidate.size() < best.size())) {
                bestPenalty = penalty;
                best = candidate;
            }
        }
        return best;
    }

    private static double threeWayPenalty(
            GroupProfile profile,
            SplitPartition candidate,
            List<SplitPartition> partitions,
            Map<String, Integer> totalSignatureCounts) {
        double penalty = 0.0;
        for (SplitPartition partition : partitions) {
            int sizeAfter = partition.size() + (partition == candidate ? profile.size() : 0);
            double sizeDiff = sizeAfter - partition.targetSamples();
            penalty += (sizeDiff * sizeDiff) / Math.max(1.0, partition.targetSamples());

            for (Map.Entry<String, Integer> entry : totalSignatureCounts.entrySet()) {
                String signature = entry.getKey();
                int profileCount = profile.signatureCounts().getOrDefault(signature, 0);
                int after = partition.signatureCounts().getOrDefault(signature, 0)
                        + (partition == candidate ? profileCount : 0);
                double target = entry.getValue() * partition.targetFraction();
                double diff = after - target;
                penalty += (diff * diff) / Math.max(1.0, target);
            }
        }
        return penalty;
    }

    private static List<Dataset.Fold<List<MultimodalContent>>> foldsByValidationIndices(
            Dataset<? extends List<MultimodalContent>> dataset,
            List<List<Integer>> validationFolds) {
        List<Dataset.Fold<List<MultimodalContent>>> result = new ArrayList<>(validationFolds.size());
        for (int fold = 0; fold < validationFolds.size(); fold++) {
            List<Integer> validationIndices = validationFolds.get(fold);
            if (validationIndices.isEmpty()) {
                throw new IllegalArgumentException("fold " + fold + " has no validation samples");
            }
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
            if (trainIndices.isEmpty()) {
                throw new IllegalArgumentException("fold " + fold + " has no training samples");
            }
            result.add(new Dataset.Fold<>(
                    fold,
                    validationFolds.size(),
                    new IndexedMultimodalDataset(dataset, trainIndices),
                    new IndexedMultimodalDataset(dataset, validationIndices)));
        }
        return List.copyOf(result);
    }

    private static long mixSeed(long seed, int ordinal, Object key) {
        long mixed = seed ^ (0x9E3779B97F4A7C15L * (ordinal + 1L));
        return mixed ^ (key == null ? 0L : Integer.toUnsignedLong(key.hashCode()));
    }

    private record Group(List<Integer> indices) {
        private Group {
            indices = List.copyOf(indices);
        }

        private int size() {
            return indices.size();
        }
    }

    private record GroupProfile(Group group, Map<String, Integer> signatureCounts) {
        private GroupProfile {
            group = Objects.requireNonNull(group, "group must not be null");
            signatureCounts = Collections.unmodifiableMap(new LinkedHashMap<>(
                    Objects.requireNonNull(signatureCounts, "signatureCounts must not be null")));
        }

        private int size() {
            return group.size();
        }
    }

    private enum SplitSide {
        TRAIN,
        VALIDATION
    }

    private enum SplitPart {
        TRAIN,
        VALIDATION,
        TEST
    }

    private static final class SplitPartition {
        private final SplitPart part;
        private final int targetSamples;
        private final double targetFraction;
        private final List<Integer> indices = new ArrayList<>();
        private final Map<String, Integer> signatureCounts = new LinkedHashMap<>();
        private int groupCount;

        private SplitPartition(SplitPart part, int targetSamples, double targetFraction) {
            this.part = Objects.requireNonNull(part, "part must not be null");
            this.targetSamples = targetSamples;
            this.targetFraction = targetFraction;
        }

        private void add(GroupProfile profile) {
            indices.addAll(profile.group().indices());
            addSignatureCounts(signatureCounts, profile.signatureCounts());
            groupCount++;
        }

        private SplitPart part() {
            return part;
        }

        private int targetSamples() {
            return targetSamples;
        }

        private double targetFraction() {
            return targetFraction;
        }

        private List<Integer> indices() {
            return indices;
        }

        private Map<String, Integer> signatureCounts() {
            return signatureCounts;
        }

        private int groupCount() {
            return groupCount;
        }

        private int size() {
            return indices.size();
        }
    }

    private static final class IndexedMultimodalDataset implements Dataset<List<MultimodalContent>> {
        private final Dataset<? extends List<MultimodalContent>> source;
        private final List<Integer> indices;

        private IndexedMultimodalDataset(
                Dataset<? extends List<MultimodalContent>> source,
                List<Integer> indices) {
            this.source = Objects.requireNonNull(source, "source must not be null");
            this.indices = List.copyOf(Objects.requireNonNull(indices, "indices must not be null"));
        }

        @Override
        public int size() {
            return indices.size();
        }

        @Override
        public List<MultimodalContent> get(int index) {
            return source.get(indices.get(index));
        }

        private List<Integer> sourceIndices() {
            return indices;
        }
    }
}
