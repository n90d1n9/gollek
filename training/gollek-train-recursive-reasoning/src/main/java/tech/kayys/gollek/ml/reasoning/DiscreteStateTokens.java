package tech.kayys.gollek.ml.reasoning;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Shared state-metadata helpers for discrete token predictions.
 */
public final class DiscreteStateTokens {
    public static final String DEFAULT_METADATA_KEY = "discreteTokens";
    public static final String PROJECTION_MODE_METADATA_KEY = "discreteProjectionMode";
    public static final String PROJECTION_METADATA_KEY = "discreteProjectionMetadata";

    private DiscreteStateTokens() {
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
        Objects.requireNonNull(state, "state must not be null");
        Objects.requireNonNull(metadataKey, "metadataKey must not be null");
        Objects.requireNonNull(tokens, "tokens must not be null");
        Map<String, Object> metadata = new HashMap<>(state.metadata());
        metadata.put(metadataKey, tokens.clone());
        return copyWithMetadata(state, metadata);
    }

    public static RecursiveReasoningState attachProjection(
            RecursiveReasoningState state,
            int[] tokens,
            String projectionMode,
            Map<String, Object> projectionMetadata) {
        return attachProjection(
                state,
                DEFAULT_METADATA_KEY,
                tokens,
                PROJECTION_MODE_METADATA_KEY,
                projectionMode,
                PROJECTION_METADATA_KEY,
                projectionMetadata);
    }

    public static RecursiveReasoningState attachProjection(
            RecursiveReasoningState state,
            String tokenMetadataKey,
            int[] tokens,
            String projectionModeMetadataKey,
            String projectionMode,
            String projectionMetadataKey,
            Map<String, Object> projectionMetadata) {
        Objects.requireNonNull(state, "state must not be null");
        Objects.requireNonNull(tokenMetadataKey, "tokenMetadataKey must not be null");
        Objects.requireNonNull(tokens, "tokens must not be null");
        Objects.requireNonNull(projectionModeMetadataKey, "projectionModeMetadataKey must not be null");
        Objects.requireNonNull(projectionMode, "projectionMode must not be null");
        Objects.requireNonNull(projectionMetadataKey, "projectionMetadataKey must not be null");
        Map<String, Object> metadata = new HashMap<>(state.metadata());
        metadata.put(tokenMetadataKey, tokens.clone());
        metadata.put(projectionModeMetadataKey, projectionMode);
        metadata.put(projectionMetadataKey, projectionMetadata == null ? Map.of() : Map.copyOf(projectionMetadata));
        return copyWithMetadata(state, metadata);
    }

    public static DiscreteStateTokenDecoder defaultDecoder() {
        return DiscreteStateTokenDecoder.fromMetadata(DEFAULT_METADATA_KEY);
    }

    public static RecursiveReasoningState copyWithMetadata(
            RecursiveReasoningState state,
            Map<String, Object> metadata) {
        Objects.requireNonNull(state, "state must not be null");
        return new RecursiveReasoningState(
                state.stateId(),
                state.supervisionStep(),
                state.transitionIndex(),
                state.sampleIndex(),
                state.latentState(),
                metadata);
    }
}
