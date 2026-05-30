package tech.kayys.gollek.ml.reasoning;

/**
 * Converts a GRAM transition sample into N-Queens prediction tokens.
 */
@FunctionalInterface
public interface NQueensTokenPredictionHead {
    NQueensTokenProjectionResult predictTokens(
            RecursiveReasoningState proposedState,
            RecursiveReasoningContext context,
            GramTransitionSample sample,
            NQueensProblem problem);
}
