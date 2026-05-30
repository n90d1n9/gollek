package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;

/**
 * Decorates GRAM next-state creation with projected discrete token metadata.
 */
public final class DiscreteTokenProjectingNextStateFactory implements GramNextStateFactory {
    private final GramNextStateFactory delegate;
    private final DiscreteTokenPredictionHead predictionHead;
    private final String tokenMetadataKey;
    private final String projectionModeMetadataKey;
    private final String projectionMetadataKey;

    public DiscreteTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            DiscreteTokenPredictionHead predictionHead) {
        this(
                delegate,
                predictionHead,
                DiscreteStateTokens.DEFAULT_METADATA_KEY,
                DiscreteStateTokens.PROJECTION_MODE_METADATA_KEY,
                DiscreteStateTokens.PROJECTION_METADATA_KEY);
    }

    public DiscreteTokenProjectingNextStateFactory(
            GramNextStateFactory delegate,
            DiscreteTokenPredictionHead predictionHead,
            String tokenMetadataKey,
            String projectionModeMetadataKey,
            String projectionMetadataKey) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.predictionHead = Objects.requireNonNull(predictionHead, "predictionHead must not be null");
        this.tokenMetadataKey = Objects.requireNonNull(tokenMetadataKey, "tokenMetadataKey must not be null");
        this.projectionModeMetadataKey = Objects.requireNonNull(
                projectionModeMetadataKey,
                "projectionModeMetadataKey must not be null");
        this.projectionMetadataKey = Objects.requireNonNull(
                projectionMetadataKey,
                "projectionMetadataKey must not be null");
    }

    @Override
    public RecursiveReasoningState nextState(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            GramTransitionSample sample) {
        RecursiveReasoningState proposed = Objects.requireNonNull(
                delegate.nextState(previousState, context, sample),
                "delegate returned null nextState");
        DiscreteTokenProjection projection = Objects.requireNonNull(
                predictionHead.predictTokens(proposed, context, sample),
                "predictionHead returned null projection");
        return DiscreteStateTokens.attachProjection(
                proposed,
                tokenMetadataKey,
                projection.tokens(),
                projectionModeMetadataKey,
                projection.projectionMode(),
                projectionMetadataKey,
                projection.metadata());
    }
}
