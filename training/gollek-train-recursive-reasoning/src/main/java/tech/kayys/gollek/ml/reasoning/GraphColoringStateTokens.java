package tech.kayys.gollek.ml.reasoning;

/**
 * Shared metadata keys and helpers for carrying Graph Coloring predictions in states.
 */
public final class GraphColoringStateTokens {
    public static final String DEFAULT_METADATA_KEY = "graphColoringTokens";
    public static final String PROJECTION_MODE_METADATA_KEY = "graphColoringProjectionMode";
    public static final String PROJECTION_METADATA_KEY = "graphColoringProjectionMetadata";

    private GraphColoringStateTokens() {
    }

    public static RecursiveReasoningState attach(
            RecursiveReasoningState state,
            int[] tokens) {
        return attach(state, DEFAULT_METADATA_KEY, tokens);
    }

    public static RecursiveReasoningState attach(
            RecursiveReasoningState state,
            String metadataKey,
            int[] tokens) {
        return DiscreteStateTokens.attach(state, metadataKey, tokens);
    }

    public static RecursiveReasoningState attachProjection(
            RecursiveReasoningState state,
            GraphColoringTokenProjectionResult projection) {
        return attachProjection(state, DEFAULT_METADATA_KEY, projection);
    }

    public static RecursiveReasoningState attachProjection(
            RecursiveReasoningState state,
            String metadataKey,
            GraphColoringTokenProjectionResult projection) {
        if (projection == null) {
            throw new NullPointerException("projection must not be null");
        }
        return DiscreteStateTokens.attachProjection(
                state,
                metadataKey,
                projection.tokens(),
                PROJECTION_MODE_METADATA_KEY,
                projection.mode().name(),
                PROJECTION_METADATA_KEY,
                projection.metadata());
    }

    public static DiscreteStateTokenDecoder defaultDecoder() {
        return DiscreteStateTokenDecoder.fromMetadata(DEFAULT_METADATA_KEY);
    }
}
