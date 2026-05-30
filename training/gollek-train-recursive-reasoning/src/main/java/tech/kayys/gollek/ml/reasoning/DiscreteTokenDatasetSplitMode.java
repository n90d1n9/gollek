package tech.kayys.gollek.ml.reasoning;

/**
 * Split policy for trainer-facing token dataset plans.
 */
public enum DiscreteTokenDatasetSplitMode {
    SEQUENTIAL_FRACTIONS,
    SHUFFLED_FRACTIONS,
    STRATIFIED_SEQUENTIAL_FRACTIONS,
    STRATIFIED_SHUFFLED_FRACTIONS
}
