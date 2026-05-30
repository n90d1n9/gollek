package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * One sampled latent reasoning path through a recursive rollout.
 */
public record RecursiveReasoningTrajectory(
        int sampleIndex,
        List<RecursiveReasoningState> states,
        double cumulativeLogProbability,
        Double terminalRewardScore) {

    public RecursiveReasoningTrajectory {
        if (sampleIndex < 0) {
            throw new IllegalArgumentException("sampleIndex must be >= 0 but was " + sampleIndex);
        }
        states = List.copyOf(Objects.requireNonNull(states, "states must not be null"));
        if (states.isEmpty()) {
            throw new IllegalArgumentException("states must not be empty");
        }
    }

    public RecursiveReasoningState initialState() {
        return states.getFirst();
    }

    public RecursiveReasoningState finalState() {
        return states.getLast();
    }
}
