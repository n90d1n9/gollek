package tech.kayys.gollek.ml.reasoning;

/**
 * Decorates GRAM next-state creation with Graph Coloring token predictions.
 */
public final class GraphColoringTokenProjectingNextStateFactory implements GramNextStateFactory {
    private final DiscreteTokenProjectingNextStateFactory delegate;

    public GraphColoringTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            GraphColoringProblem problem,
            GraphColoringTokenPredictionHead predictionHead) {
        this(delegate, problem, predictionHead, GraphColoringStateTokens.DEFAULT_METADATA_KEY);
    }

    public GraphColoringTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            GraphColoringProblem problem,
            GraphColoringTokenPredictionHead predictionHead,
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
                    GraphColoringTokenProjectionResult projection = predictionHead.predictTokens(
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
                GraphColoringStateTokens.PROJECTION_MODE_METADATA_KEY,
                GraphColoringStateTokens.PROJECTION_METADATA_KEY);
    }

    @Override
    public RecursiveReasoningState nextState(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            GramTransitionSample sample) {
        return delegate.nextState(previousState, context, sample);
    }
}
