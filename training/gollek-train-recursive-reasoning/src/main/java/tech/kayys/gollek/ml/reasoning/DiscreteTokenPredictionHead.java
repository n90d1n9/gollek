package tech.kayys.gollek.ml.reasoning;

/**
 * Converts a GRAM transition sample into generic discrete prediction tokens.
 */
@FunctionalInterface
public interface DiscreteTokenPredictionHead {
    DiscreteTokenProjection predictTokens(
            RecursiveReasoningState proposedState,
            RecursiveReasoningContext context,
            GramTransitionSample sample);
}
