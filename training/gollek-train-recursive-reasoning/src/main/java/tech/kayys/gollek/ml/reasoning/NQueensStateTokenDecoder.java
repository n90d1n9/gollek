package tech.kayys.gollek.ml.reasoning;

/**
 * Extracts N-Queens board tokens from a generic recursive-reasoning state.
 */
@FunctionalInterface
public interface NQueensStateTokenDecoder {
    int[] decodeTokens(RecursiveReasoningState state, NQueensProblem problem);

    static NQueensStateTokenDecoder fromMetadata(String metadataKey) {
        DiscreteStateTokenDecoder decoder = DiscreteStateTokenDecoder.fromMetadata(metadataKey);
        return (state, problem) -> {
            if (problem == null) {
                throw new NullPointerException("problem must not be null");
            }
            return decoder.decodeTokens(state);
        };
    }
}
