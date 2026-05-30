package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Composes profiling, splitting, and epoch construction for token datasets.
 */
public final class DiscreteTokenDatasetPlanner {
    private DiscreteTokenDatasetPlanner() {}

    public static DiscreteTokenDatasetPlan plan(
            List<DiscreteTokenDatasetExample> examples,
            DiscreteTokenDatasetPlanConfig config) {
        Objects.requireNonNull(examples, "examples must not be null");
        Objects.requireNonNull(config, "config must not be null");

        DiscreteTokenDatasetProfile profile = DiscreteTokenDatasetProfiler.profile(examples);
        DiscreteTokenDatasetSplit split = split(examples, config);
        DiscreteTokenDatasetEpoch trainEpoch = trainEpoch(split.trainExamples(), config);
        DiscreteTokenDatasetEpoch validationEpoch = DiscreteTokenDatasetEpochBatcher.sequential(
                split.validationExamples(),
                config.evaluationBatchSize(),
                config.inputPadToken(),
                config.targetPadToken(),
                false);
        DiscreteTokenDatasetEpoch testEpoch = DiscreteTokenDatasetEpochBatcher.sequential(
                split.testExamples(),
                config.evaluationBatchSize(),
                config.inputPadToken(),
                config.targetPadToken(),
                false);

        return new DiscreteTokenDatasetPlan(config, profile, split, trainEpoch, validationEpoch, testEpoch);
    }

    private static DiscreteTokenDatasetSplit split(
            List<DiscreteTokenDatasetExample> examples,
            DiscreteTokenDatasetPlanConfig config) {
        return switch (config.splitMode()) {
            case SEQUENTIAL_FRACTIONS -> DiscreteTokenDatasetSplitter.sequentialByFractions(
                    examples,
                    config.validationFraction(),
                    config.testFraction());
            case SHUFFLED_FRACTIONS -> DiscreteTokenDatasetSplitter.shuffledByFractions(
                    examples,
                    config.validationFraction(),
                    config.testFraction(),
                    config.splitSeed());
            case STRATIFIED_SEQUENTIAL_FRACTIONS -> DiscreteTokenDatasetSplitter.stratifiedSequentialByFractions(
                    examples,
                    config.validationFraction(),
                    config.testFraction());
            case STRATIFIED_SHUFFLED_FRACTIONS -> DiscreteTokenDatasetSplitter.stratifiedShuffledByFractions(
                    examples,
                    config.validationFraction(),
                    config.testFraction(),
                    config.splitSeed());
        };
    }

    private static DiscreteTokenDatasetEpoch trainEpoch(
            List<DiscreteTokenDatasetExample> examples,
            DiscreteTokenDatasetPlanConfig config) {
        return switch (config.trainEpochMode()) {
            case SEQUENTIAL -> DiscreteTokenDatasetEpochBatcher.sequential(
                    examples,
                    config.trainBatchSize(),
                    config.inputPadToken(),
                    config.targetPadToken(),
                    config.dropLastTrain());
            case SHUFFLED -> DiscreteTokenDatasetEpochBatcher.shuffled(
                    examples,
                    config.trainBatchSize(),
                    config.inputPadToken(),
                    config.targetPadToken(),
                    config.trainEpochSeed(),
                    config.dropLastTrain());
            case LENGTH_SORTED -> DiscreteTokenDatasetEpochBatcher.lengthSorted(
                    examples,
                    config.trainBatchSize(),
                    config.inputPadToken(),
                    config.targetPadToken(),
                    config.dropLastTrain());
        };
    }
}
