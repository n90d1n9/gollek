package tech.kayys.gollek.ml.reasoning;

/**
 * Converts a sampled GRAM latent payload into the next recursive rollout state.
 */
@FunctionalInterface
public interface GramNextStateFactory {
    RecursiveReasoningState nextState(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            GramTransitionSample sample);
}
