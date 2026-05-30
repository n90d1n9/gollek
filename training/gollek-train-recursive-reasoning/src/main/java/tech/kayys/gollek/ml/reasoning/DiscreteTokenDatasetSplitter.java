package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Deterministic split helpers for generic token datasets.
 */
public final class DiscreteTokenDatasetSplitter {
    private DiscreteTokenDatasetSplitter() {}

    public static DiscreteTokenDatasetSplit sequentialByCounts(
            List<DiscreteTokenDatasetExample> examples,
            int validationCount,
            int testCount) {
        return splitByCounts(examples, validationCount, testCount, false, 0L);
    }

    public static DiscreteTokenDatasetSplit shuffledByCounts(
            List<DiscreteTokenDatasetExample> examples,
            int validationCount,
            int testCount,
            long seed) {
        return splitByCounts(examples, validationCount, testCount, true, seed);
    }

    public static DiscreteTokenDatasetSplit sequentialByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction) {
        return splitByFractions(examples, validationFraction, testFraction, false, 0L);
    }

    public static DiscreteTokenDatasetSplit shuffledByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction,
            long seed) {
        return splitByFractions(examples, validationFraction, testFraction, true, seed);
    }

    public static DiscreteTokenDatasetSplit stratifiedSequentialByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction) {
        return splitStratifiedByFractions(examples, validationFraction, testFraction, false, 0L);
    }

    public static DiscreteTokenDatasetSplit stratifiedShuffledByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction,
            long seed) {
        return splitStratifiedByFractions(examples, validationFraction, testFraction, true, seed);
    }

    private static DiscreteTokenDatasetSplit splitByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction,
            boolean shuffled,
            long seed) {
        Objects.requireNonNull(examples, "examples must not be null");
        validateFraction(validationFraction, "validationFraction");
        validateFraction(testFraction, "testFraction");
        if (validationFraction + testFraction > 1.0d) {
            throw new IllegalArgumentException("validationFraction + testFraction must be <= 1.0");
        }
        int validationCount = (int) Math.floor(examples.size() * validationFraction);
        int testCount = (int) Math.floor(examples.size() * testFraction);
        return splitByCounts(examples, validationCount, testCount, shuffled, seed);
    }

    private static DiscreteTokenDatasetSplit splitByCounts(
            List<DiscreteTokenDatasetExample> examples,
            int validationCount,
            int testCount,
            boolean shuffled,
            long seed) {
        List<DiscreteTokenDatasetExample> ordered = copyExamples(examples);
        if (validationCount < 0) {
            throw new IllegalArgumentException("validationCount must be >= 0 but was " + validationCount);
        }
        if (testCount < 0) {
            throw new IllegalArgumentException("testCount must be >= 0 but was " + testCount);
        }
        if (validationCount + testCount > ordered.size()) {
            throw new IllegalArgumentException(
                    "validationCount + testCount must be <= example count " + ordered.size());
        }
        if (shuffled) {
            Collections.shuffle(ordered, new Random(seed));
        }

        int trainCount = ordered.size() - validationCount - testCount;
        int validationEnd = trainCount + validationCount;
        return new DiscreteTokenDatasetSplit(
                ordered.subList(0, trainCount),
                ordered.subList(trainCount, validationEnd),
                ordered.subList(validationEnd, ordered.size()),
                shuffled,
                seed);
    }

    private static DiscreteTokenDatasetSplit splitStratifiedByFractions(
            List<DiscreteTokenDatasetExample> examples,
            double validationFraction,
            double testFraction,
            boolean shuffled,
            long seed) {
        List<DiscreteTokenDatasetExample> ordered = copyExamples(examples);
        validateFraction(validationFraction, "validationFraction");
        validateFraction(testFraction, "testFraction");
        if (validationFraction + testFraction > 1.0d) {
            throw new IllegalArgumentException("validationFraction + testFraction must be <= 1.0");
        }

        Map<String, List<DiscreteTokenDatasetExample>> byTask = new LinkedHashMap<>();
        for (DiscreteTokenDatasetExample example : ordered) {
            byTask.computeIfAbsent(example.taskId(), ignored -> new ArrayList<>()).add(example);
        }

        List<DiscreteTokenDatasetExample> train = new ArrayList<>();
        List<DiscreteTokenDatasetExample> validation = new ArrayList<>();
        List<DiscreteTokenDatasetExample> test = new ArrayList<>();
        Random random = new Random(seed);
        for (List<DiscreteTokenDatasetExample> taskExamples : byTask.values()) {
            List<DiscreteTokenDatasetExample> group = new ArrayList<>(taskExamples);
            if (shuffled) {
                Collections.shuffle(group, random);
            }

            int validationCount = (int) Math.floor(group.size() * validationFraction);
            int testCount = (int) Math.floor(group.size() * testFraction);
            int trainCount = group.size() - validationCount - testCount;
            int validationEnd = trainCount + validationCount;
            train.addAll(group.subList(0, trainCount));
            validation.addAll(group.subList(trainCount, validationEnd));
            test.addAll(group.subList(validationEnd, group.size()));
        }

        return new DiscreteTokenDatasetSplit(train, validation, test, shuffled, seed);
    }

    private static List<DiscreteTokenDatasetExample> copyExamples(List<DiscreteTokenDatasetExample> examples) {
        Objects.requireNonNull(examples, "examples must not be null");
        List<DiscreteTokenDatasetExample> copy = new ArrayList<>(examples.size());
        for (int index = 0; index < examples.size(); index++) {
            copy.add(Objects.requireNonNull(examples.get(index), "examples[" + index + "] must not be null"));
        }
        return copy;
    }

    private static void validateFraction(double fraction, String name) {
        if (!Double.isFinite(fraction) || fraction < 0.0d || fraction > 1.0d) {
            throw new IllegalArgumentException(name + " must be finite and between 0.0 and 1.0 but was " + fraction);
        }
    }
}
