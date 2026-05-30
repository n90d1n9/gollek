package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Immutable train/validation/test split for generic token examples.
 */
public record DiscreteTokenDatasetSplit(
        List<DiscreteTokenDatasetExample> trainExamples,
        List<DiscreteTokenDatasetExample> validationExamples,
        List<DiscreteTokenDatasetExample> testExamples,
        boolean shuffled,
        long seed) {

    public DiscreteTokenDatasetSplit {
        trainExamples = copyExamples(trainExamples, "trainExamples");
        validationExamples = copyExamples(validationExamples, "validationExamples");
        testExamples = copyExamples(testExamples, "testExamples");
    }

    public int trainCount() {
        return trainExamples.size();
    }

    public int validationCount() {
        return validationExamples.size();
    }

    public int testCount() {
        return testExamples.size();
    }

    public int exampleCount() {
        return trainCount() + validationCount() + testCount();
    }

    public boolean hasValidationExamples() {
        return !validationExamples.isEmpty();
    }

    public boolean hasTestExamples() {
        return !testExamples.isEmpty();
    }

    public DiscreteTokenDatasetProfile profile() {
        java.util.ArrayList<DiscreteTokenDatasetExample> examples =
                new java.util.ArrayList<>(exampleCount());
        examples.addAll(trainExamples);
        examples.addAll(validationExamples);
        examples.addAll(testExamples);
        return DiscreteTokenDatasetProfiler.profile(examples);
    }

    public DiscreteTokenDatasetProfile trainProfile() {
        return DiscreteTokenDatasetProfiler.profile(trainExamples);
    }

    public DiscreteTokenDatasetProfile validationProfile() {
        return DiscreteTokenDatasetProfiler.profile(validationExamples);
    }

    public DiscreteTokenDatasetProfile testProfile() {
        return DiscreteTokenDatasetProfiler.profile(testExamples);
    }

    public DiscreteTokenDatasetEpoch trainEpoch(int batchSize, int padToken, boolean shuffle, long epochSeed, boolean dropLast) {
        return epoch(trainExamples, batchSize, padToken, padToken, shuffle, epochSeed, dropLast);
    }

    public DiscreteTokenDatasetEpoch trainEpoch(
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean shuffle,
            long epochSeed,
            boolean dropLast) {
        return epoch(trainExamples, batchSize, inputPadToken, targetPadToken, shuffle, epochSeed, dropLast);
    }

    public DiscreteTokenDatasetEpoch validationEpoch(int batchSize, int padToken) {
        return validationEpoch(batchSize, padToken, padToken);
    }

    public DiscreteTokenDatasetEpoch validationEpoch(int batchSize, int inputPadToken, int targetPadToken) {
        return DiscreteTokenDatasetEpochBatcher.sequential(
                validationExamples,
                batchSize,
                inputPadToken,
                targetPadToken,
                false);
    }

    public DiscreteTokenDatasetEpoch testEpoch(int batchSize, int padToken) {
        return testEpoch(batchSize, padToken, padToken);
    }

    public DiscreteTokenDatasetEpoch testEpoch(int batchSize, int inputPadToken, int targetPadToken) {
        return DiscreteTokenDatasetEpochBatcher.sequential(
                testExamples,
                batchSize,
                inputPadToken,
                targetPadToken,
                false);
    }

    private static DiscreteTokenDatasetEpoch epoch(
            List<DiscreteTokenDatasetExample> examples,
            int batchSize,
            int inputPadToken,
            int targetPadToken,
            boolean shuffle,
            long epochSeed,
            boolean dropLast) {
        if (shuffle) {
            return DiscreteTokenDatasetEpochBatcher.shuffled(
                    examples,
                    batchSize,
                    inputPadToken,
                    targetPadToken,
                    epochSeed,
                    dropLast);
        }
        return DiscreteTokenDatasetEpochBatcher.sequential(
                examples,
                batchSize,
                inputPadToken,
                targetPadToken,
                dropLast);
    }

    private static List<DiscreteTokenDatasetExample> copyExamples(
            List<DiscreteTokenDatasetExample> examples,
            String name) {
        Objects.requireNonNull(examples, name + " must not be null");
        for (int index = 0; index < examples.size(); index++) {
            Objects.requireNonNull(examples.get(index), name + "[" + index + "] must not be null");
        }
        return List.copyOf(examples);
    }
}
