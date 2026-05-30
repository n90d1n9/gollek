package tech.kayys.gollek.ml.reasoning;

/**
 * Reduction used when aggregating deep-supervision losses across GRAM steps.
 */
public enum GramObjectiveReduction {
    SUM,
    MEAN
}
