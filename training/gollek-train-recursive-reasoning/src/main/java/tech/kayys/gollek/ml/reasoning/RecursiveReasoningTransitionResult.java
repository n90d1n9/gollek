package tech.kayys.gollek.ml.reasoning;

/**
 * Result of one stochastic latent transition in a recursive reasoning rollout.
 */
public record RecursiveReasoningTransitionResult(
        RecursiveReasoningState nextState,
        double logProbability,
        Double rewardScore) {
}
