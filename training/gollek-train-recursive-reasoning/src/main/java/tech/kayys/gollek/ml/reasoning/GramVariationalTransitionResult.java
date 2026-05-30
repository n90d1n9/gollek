package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;

/**
 * Full variational transition output plus a bridge to the generic rollout API.
 */
public record GramVariationalTransitionResult(
        GramTransitionSample sample,
        RecursiveReasoningState nextState,
        double logProbability,
        Double rewardScore) {

    public GramVariationalTransitionResult {
        sample = Objects.requireNonNull(sample, "sample must not be null");
        nextState = Objects.requireNonNull(nextState, "nextState must not be null");
        if (!Double.isFinite(logProbability)) {
            throw new IllegalArgumentException("logProbability must be finite but was " + logProbability);
        }
    }

    public RecursiveReasoningTransitionResult toRolloutTransitionResult() {
        return new RecursiveReasoningTransitionResult(nextState, logProbability, rewardScore);
    }
}
