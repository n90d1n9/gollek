package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

final class TensorDatasetSplits {

    private TensorDatasetSplits() {
    }

    static DataLoader.TensorDatasetSplit randomSplit(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            long seed) {
        requireTrainFraction(trainFraction);
        List<Integer> indices = new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList());
        Collections.shuffle(indices, new Random(seed));
        int trainSize = (int) Math.round(dataset.size() * trainFraction);
        trainSize = Math.max(1, Math.min(dataset.size() - 1, trainSize));
        return splitByIndices(
                dataset,
                indices.subList(0, trainSize),
                indices.subList(trainSize, indices.size()));
    }

    static DataLoader.TensorDatasetThreeWaySplit randomSplit(
            DataLoader.TensorDataset dataset,
            double trainFraction,
            double validationFraction,
            long seed) {
        SplitCounts counts = threeWayCounts(dataset.size(), trainFraction, validationFraction);
        List<Integer> indices = new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList());
        Collections.shuffle(indices, new Random(seed));
        List<Integer> train = indices.subList(0, counts.train());
        List<Integer> validation = indices.subList(counts.train(), counts.train() + counts.validation());
        List<Integer> test = indices.subList(counts.train() + counts.validation(), indices.size());
        return splitByIndices(dataset, train, validation, test);
    }

    static DataLoader.TensorDatasetSplit splitByIndices(
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
        return new DataLoader.TensorDatasetSplit(
                new DataLoader.TensorDataset(train),
                new DataLoader.TensorDataset(validation));
    }

    static DataLoader.TensorDatasetThreeWaySplit splitByIndices(
            DataLoader.TensorDataset dataset,
            List<Integer> trainIndices,
            List<Integer> validationIndices,
            List<Integer> testIndices) {
        List<GradTensor[]> train = new ArrayList<>(trainIndices.size());
        List<GradTensor[]> validation = new ArrayList<>(validationIndices.size());
        List<GradTensor[]> test = new ArrayList<>(testIndices.size());
        for (int index : trainIndices) {
            train.add(dataset.get(index));
        }
        for (int index : validationIndices) {
            validation.add(dataset.get(index));
        }
        for (int index : testIndices) {
            test.add(dataset.get(index));
        }
        return new DataLoader.TensorDatasetThreeWaySplit(
                new DataLoader.TensorDataset(train),
                new DataLoader.TensorDataset(validation),
                new DataLoader.TensorDataset(test));
    }

    static List<DataLoader.TensorDatasetFold> kFold(
            DataLoader.TensorDataset dataset,
            int folds,
            long seed) {
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

    static List<DataLoader.TensorDatasetFold> repeatedKFoldBySeed(
            DataLoader.TensorDataset dataset,
            int folds,
            int repeats,
            long seed) {
        requireKFold(folds, dataset.size());
        requireRepeats(repeats);
        return repeatedFolds(folds, repeats, repeat -> kFold(dataset, folds, repeatSeed(seed, repeat)));
    }

    static List<DataLoader.TensorDatasetFold> timeSeriesFolds(
            DataLoader.TensorDataset dataset,
            int splits,
            int validationSize,
            int gap,
            int maxTrainSize) {
        requireTimeSeriesSplit(splits, dataset.size(), validationSize, gap, maxTrainSize);

        int firstValidationStart = dataset.size() - splits * validationSize;
        List<DataLoader.TensorDatasetFold> result = new ArrayList<>(splits);
        for (int fold = 0; fold < splits; fold++) {
            int validationStart = firstValidationStart + fold * validationSize;
            int validationEnd = validationStart + validationSize;
            int trainEnd = validationStart - gap;
            int trainStart = maxTrainSize > 0 ? Math.max(0, trainEnd - maxTrainSize) : 0;

            List<Integer> trainIndices = IntStream.range(trainStart, trainEnd).boxed().toList();
            List<Integer> validationIndices = IntStream.range(validationStart, validationEnd).boxed().toList();
            DataLoader.TensorDatasetSplit split = splitByIndices(dataset, trainIndices, validationIndices);
            result.add(new DataLoader.TensorDatasetFold(fold, splits, split.train(), split.validation()));
        }
        return List.copyOf(result);
    }

    static int defaultTimeSeriesValidationSize(int sampleCount, int splits) {
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

    static List<DataLoader.TensorDatasetFold> foldsByValidationIndices(
            DataLoader.TensorDataset dataset,
            List<List<Integer>> validationFolds) {
        List<DataLoader.TensorDatasetFold> result = new ArrayList<>(validationFolds.size());
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

            DataLoader.TensorDatasetSplit split = splitByIndices(dataset, trainIndices, validationIndices);
            result.add(new DataLoader.TensorDatasetFold(
                    fold,
                    validationFolds.size(),
                    split.train(),
                    split.validation()));
        }
        return List.copyOf(result);
    }

    static List<DataLoader.TensorDatasetFold> repeatedFolds(
            int folds,
            int repeats,
            IntFunction<List<DataLoader.TensorDatasetFold>> foldFactory) {
        int totalFoldCount = totalRepeatedFoldCount(folds, repeats);
        List<DataLoader.TensorDatasetFold> result = new ArrayList<>(totalFoldCount);
        for (int repeat = 0; repeat < repeats; repeat++) {
            for (DataLoader.TensorDatasetFold fold : foldFactory.apply(repeat)) {
                result.add(new DataLoader.TensorDatasetFold(
                        repeat * folds + fold.foldIndex(),
                        totalFoldCount,
                        fold.train(),
                        fold.validation()));
            }
        }
        return List.copyOf(result);
    }

    static void requireTrainFraction(double trainFraction) {
        if (trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be between 0 and 1");
        }
    }

    static void requireKFold(int folds, int sampleCount) {
        if (folds < 2) {
            throw new IllegalArgumentException("folds must be at least 2, got: " + folds);
        }
        if (sampleCount < folds) {
            throw new IllegalArgumentException(
                    "folds must be less than or equal to dataset size, got folds="
                            + folds + ", size=" + sampleCount);
        }
    }

    static void requireRepeats(int repeats) {
        if (repeats < 1) {
            throw new IllegalArgumentException("repeats must be at least 1, got: " + repeats);
        }
    }

    static int totalRepeatedFoldCount(int folds, int repeats) {
        try {
            return Math.multiplyExact(folds, repeats);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException(
                    "folds * repeats is too large, got folds=" + folds + ", repeats=" + repeats,
                    error);
        }
    }

    static long repeatSeed(long seed, int repeat) {
        return seed ^ (0x9E3779B97F4A7C15L * (repeat + 1L));
    }

    static SplitCounts threeWayCounts(int size, double trainFraction, double validationFraction) {
        requireTrainValidationFractions(trainFraction, validationFraction);
        if (size < 3) {
            throw new IllegalArgumentException(
                    "dataset must contain at least three samples for train/validation/test split");
        }
        int trainCount = (int) Math.round(size * trainFraction);
        int validationCount = (int) Math.round(size * validationFraction);
        trainCount = Math.max(1, Math.min(size - 2, trainCount));
        validationCount = Math.max(1, Math.min(size - trainCount - 1, validationCount));
        int testCount = size - trainCount - validationCount;
        return new SplitCounts(trainCount, validationCount, testCount);
    }

    static void requireTrainValidationFractions(double trainFraction, double validationFraction) {
        if (!Double.isFinite(trainFraction) || !Double.isFinite(validationFraction)) {
            throw new IllegalArgumentException("split fractions must be finite");
        }
        if (trainFraction <= 0.0 || validationFraction <= 0.0 || trainFraction + validationFraction >= 1.0) {
            throw new IllegalArgumentException(
                    "trainFraction and validationFraction must be positive and leave a positive test fraction");
        }
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
}
