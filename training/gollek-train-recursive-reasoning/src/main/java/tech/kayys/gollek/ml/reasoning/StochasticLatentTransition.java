package tech.kayys.gollek.ml.reasoning;

/**
 * Shared contract for GRAM-style stochastic latent transitions.
 */
@FunctionalInterface
public interface StochasticLatentTransition {

    RecursiveReasoningTransitionResult sample(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context);
}
