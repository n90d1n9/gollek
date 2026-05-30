package tech.kayys.gollek.ml.reasoning;

/**
 * Shared metadata keys and helpers for carrying N-Queens predictions in states.
 */
public final class NQueensStateTokens {
    public static final String DEFAULT_METADATA_KEY = "nQueensTokens";
    public static final String PROJECTION_MODE_METADATA_KEY = "nQueensProjectionMode";
    public static final String PROJECTION_METADATA_KEY = "nQueensProjectionMetadata";

    private NQueensStateTokens() {
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
            NQueensTokenProjectionResult projection) {
        return attachProjection(state, DEFAULT_METADATA_KEY, projection);
    }

    public static RecursiveReasoningState attachProjection(
            RecursiveReasoningState state,
            String metadataKey,
            NQueensTokenProjectionResult projection) {
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

    public static NQueensStateTokenDecoder defaultDecoder() {
        return NQueensStateTokenDecoder.fromMetadata(DEFAULT_METADATA_KEY);
    }
}
