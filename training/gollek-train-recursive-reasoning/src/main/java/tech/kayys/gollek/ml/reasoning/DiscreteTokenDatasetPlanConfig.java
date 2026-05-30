package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;

/**
 * Configuration for planning a trainer-ready token dataset flow.
 */
public record DiscreteTokenDatasetPlanConfig(
        double validationFraction,
        double testFraction,
        DiscreteTokenDatasetSplitMode splitMode,
        long splitSeed,
        int trainBatchSize,
        int evaluationBatchSize,
        int inputPadToken,
        int targetPadToken,
        DiscreteTokenDatasetTrainEpochMode trainEpochMode,
        long trainEpochSeed,
        boolean dropLastTrain) {

    public DiscreteTokenDatasetPlanConfig {
        validateFraction(validationFraction, "validationFraction");
        validateFraction(testFraction, "testFraction");
        if (validationFraction + testFraction > 1.0d) {
            throw new IllegalArgumentException("validationFraction + testFraction must be <= 1.0");
        }
        splitMode = Objects.requireNonNull(splitMode, "splitMode must not be null");
        if (trainBatchSize <= 0) {
            throw new IllegalArgumentException("trainBatchSize must be > 0 but was " + trainBatchSize);
        }
        if (evaluationBatchSize <= 0) {
            throw new IllegalArgumentException("evaluationBatchSize must be > 0 but was " + evaluationBatchSize);
        }
        trainEpochMode = Objects.requireNonNull(trainEpochMode, "trainEpochMode must not be null");
    }

    public static DiscreteTokenDatasetPlanConfig stratifiedLengthSorted(
            double validationFraction,
            double testFraction,
            long splitSeed,
            int trainBatchSize,
            int evaluationBatchSize,
            int padToken) {
        return new DiscreteTokenDatasetPlanConfig(
                validationFraction,
                testFraction,
                DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                splitSeed,
                trainBatchSize,
                evaluationBatchSize,
                padToken,
                padToken,
                DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                0L,
                false);
    }

    public boolean usesStratifiedSplit() {
        return splitMode == DiscreteTokenDatasetSplitMode.STRATIFIED_SEQUENTIAL_FRACTIONS
                || splitMode == DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS;
    }

    public boolean usesShuffledSplit() {
        return splitMode == DiscreteTokenDatasetSplitMode.SHUFFLED_FRACTIONS
                || splitMode == DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS;
    }

    public boolean usesLengthSortedTraining() {
        return trainEpochMode == DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED;
    }

    private static void validateFraction(double fraction, String name) {
        if (!Double.isFinite(fraction) || fraction < 0.0d || fraction > 1.0d) {
            throw new IllegalArgumentException(name + " must be finite and between 0.0 and 1.0 but was " + fraction);
        }
    }
}
