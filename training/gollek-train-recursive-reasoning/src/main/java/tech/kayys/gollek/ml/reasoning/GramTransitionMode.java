package tech.kayys.gollek.ml.reasoning;

/**
 * Selects whether a GRAM transition samples from the learned prior or target-conditioned posterior.
 */
public enum GramTransitionMode {
    PRIOR,
    POSTERIOR
}
