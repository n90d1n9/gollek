package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Immutable latent-state snapshot within a recursive reasoning rollout.
 */
public record RecursiveReasoningState(
        String stateId,
        int supervisionStep,
        int transitionIndex,
        int sampleIndex,
        Tensor latentState,
        Map<String, Object> metadata) {

    public RecursiveReasoningState {
        stateId = Objects.requireNonNull(stateId, "stateId must not be null");
        if (supervisionStep < 0) {
            throw new IllegalArgumentException("supervisionStep must be >= 0 but was " + supervisionStep);
        }
        if (transitionIndex < 0) {
            throw new IllegalArgumentException("transitionIndex must be >= 0 but was " + transitionIndex);
        }
        if (sampleIndex < 0) {
            throw new IllegalArgumentException("sampleIndex must be >= 0 but was " + sampleIndex);
        }
        latentState = Objects.requireNonNull(latentState, "latentState must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
