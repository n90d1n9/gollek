package tech.kayys.gollek.ml.reasoning;

/**
 * Decorates GRAM next-state creation with N-Queens token predictions.
 */
public final class NQueensTokenProjectingNextStateFactory implements GramNextStateFactory {
    private final DiscreteTokenProjectingNextStateFactory delegate;

    public NQueensTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            NQueensProblem problem,
            NQueensTokenPredictionHead predictionHead) {
        this(delegate, problem, predictionHead, NQueensStateTokens.DEFAULT_METADATA_KEY);
    }

    public NQueensTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            NQueensProblem problem,
            NQueensTokenPredictionHead predictionHead,
            String metadataKey) {
        if (problem == null) {
            throw new NullPointerException("problem must not be null");
        }
        if (predictionHead == null) {
            throw new NullPointerException("predictionHead must not be null");
        }
        this.delegate = new DiscreteTokenProjectingNextStateFactory(
                delegate,
                (proposedState, context, sample) -> {
                    NQueensTokenProjectionResult projection = predictionHead.predictTokens(
                            proposedState,
                            context,
                            sample,
                            problem);
                    if (projection == null) {
                        return null;
                    }
                    return new DiscreteTokenProjection(
                            projection.tokens(),
                            projection.mode().name(),
                            projection.metadata());
                },
                metadataKey,
                NQueensStateTokens.PROJECTION_MODE_METADATA_KEY,
                NQueensStateTokens.PROJECTION_METADATA_KEY);
    }

    @Override
    public RecursiveReasoningState nextState(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            GramTransitionSample sample) {
        return delegate.nextState(previousState, context, sample);
    }
}
