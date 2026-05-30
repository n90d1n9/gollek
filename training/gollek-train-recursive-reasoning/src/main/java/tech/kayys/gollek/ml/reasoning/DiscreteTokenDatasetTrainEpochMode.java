package tech.kayys.gollek.ml.reasoning;

/**
 * Training epoch ordering policy for token dataset plans.
 */
public enum DiscreteTokenDatasetTrainEpochMode {
    SEQUENTIAL,
    SHUFFLED,
    LENGTH_SORTED
}
