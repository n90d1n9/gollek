package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Builds repeatable mini-batch epochs from generic token examples.
 */
public final class DiscreteTokenDatasetEpochBatcher {
    private DiscreteTokenDatasetEpochBatcher() {}

    public static DiscreteTokenDatasetEpoch sequential(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int padToken) {
        return sequential(examples, batchSize, padToken, padToken, false);
    }

    public static DiscreteTokenDatasetEpoch sequential(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean dropLast) {
        return build(examples, batchSize, inputPadToken, targetPadToken, false, 0L, dropLast);
    }

    public static DiscreteTokenDatasetEpoch shuffled(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int padToken,
            long seed,
            boolean dropLast) {
        return shuffled(examples, batchSize, padToken, padToken, seed, dropLast);
    }

    public static DiscreteTokenDatasetEpoch shuffled(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            long seed,
            boolean dropLast) {
        return build(examples, batchSize, inputPadToken, targetPadToken, true, seed, dropLast);
    }

    public static DiscreteTokenDatasetEpoch lengthSorted(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int padToken,
            boolean dropLast) {
        return lengthSorted(examples, batchSize, padToken, padToken, dropLast);
    }

    public static DiscreteTokenDatasetEpoch lengthSorted(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean dropLast) {
        List<DiscreteTokenDatasetExample> ordered = copyExamples(examples);
        ordered.sort(Comparator.comparingInt(DiscreteTokenDatasetEpochBatcher::combinedTokenCount)
                .thenComparing(DiscreteTokenDatasetExample::taskId)
                .thenComparingInt(DiscreteTokenDatasetExample::exampleIndex));
        return buildOrdered(ordered, batchSize, inputPadToken, targetPadToken, false, 0L, dropLast);
    }

    private static DiscreteTokenDatasetEpoch build(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean shuffled,
            long seed,
            boolean dropLast) {
        List<DiscreteTokenDatasetExample> ordered = copyExamples(examples);
        if (shuffled) {
            Collections.shuffle(ordered, new Random(seed));
        }
        return buildOrdered(ordered, batchSize, inputPadToken, targetPadToken, shuffled, seed, dropLast);
    }

    private static DiscreteTokenDatasetEpoch buildOrdered(
            List<DiscreteTokenDatasetExample> ordered,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean shuffled,
            long seed,
            boolean dropLast) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0 but was " + batchSize);
        }

        List<DiscreteTokenDatasetBatch> batches = new ArrayList<>();
        int emitted = 0;
        int dropped = 0;
        for (int start = 0; start < ordered.size(); start += batchSize) {
            int end = Math.min(start + batchSize, ordered.size());
            int remaining = end - start;
            if (dropLast && remaining < batchSize) {
                dropped += remaining;
                break;
            }
            batches.add(DiscreteTokenDatasetBatcher.batch(
                    ordered.subList(start, end),
                    inputPadToken,
                    targetPadToken));
            emitted += remaining;
        }

        return new DiscreteTokenDatasetEpoch(
                batches,
                ordered.size(),
                emitted,
                dropped,
                batchSize,
                shuffled,
                dropLast,
                seed,
                inputPadToken,
                targetPadToken);
    }

    private static List<DiscreteTokenDatasetExample> copyExamples(List<DiscreteTokenDatasetExample> examples) {
        Objects.requireNonNull(examples, "examples must not be null");
        List<DiscreteTokenDatasetExample> ordered = new ArrayList<>(examples.size());
        for (int index = 0; index < examples.size(); index++) {
            ordered.add(Objects.requireNonNull(examples.get(index), "examples[" + index + "] must not be null"));
        }
        return ordered;
    }

    private static int combinedTokenCount(DiscreteTokenDatasetExample example) {
        return example.inputTokenCount() + example.targetTokenCount();
    }
}
