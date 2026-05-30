package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Small reference executor for GRAM-style recursive rollouts.
 */
public final class ReferenceRecursiveReasoningRolloutExecutor {
    public RecursiveReasoningRolloutResult execute(
            Tensor initialLatentState,
            RecursiveReasoningContext context,
            StochasticLatentTransition transition) {
        Objects.requireNonNull(initialLatentState, "initialLatentState must not be null");
        Objects.requireNonNull(context, "context must not be null");
        Objects.requireNonNull(transition, "transition must not be null");

        RecursiveReasoningConfig config = context.config();
        List<RecursiveReasoningTrajectory> trajectories = new ArrayList<>(config.parallelSamples());
        int transitionsPerStep = config.transitionsPerSupervisionStep();
        int supervisionSteps = config.supervisionSteps();

        for (int sampleIndex = 0; sampleIndex < config.parallelSamples(); sampleIndex++) {
            List<RecursiveReasoningState> states = new ArrayList<>();
            RecursiveReasoningState current = new RecursiveReasoningState(
                    "root-s" + sampleIndex,
                    0,
                    0,
                    sampleIndex,
                    initialLatentState,
                    Map.of("initial", true));
            states.add(current);
            double cumulativeLogProbability = 0.0;
            Double terminalReward = null;

            for (int supervisionStep = 0; supervisionStep < supervisionSteps; supervisionStep++) {
                for (int transitionIndex = 1; transitionIndex <= transitionsPerStep; transitionIndex++) {
                    RecursiveReasoningTransitionResult sampled = transition.sample(current, context);
                    RecursiveReasoningState sampledState =
                            Objects.requireNonNull(sampled.nextState(), "transition returned null nextState");
                    Map<String, Object> metadata = new HashMap<>(sampledState.metadata());
                    metadata.put("sampleIndex", sampleIndex);
                    metadata.put("supervisionStep", supervisionStep);
                    metadata.put("transitionIndex", transitionIndex);

                    current = new RecursiveReasoningState(
                            sampledState.stateId(),
                            supervisionStep,
                            transitionIndex,
                            sampleIndex,
                            sampledState.latentState(),
                            metadata);
                    states.add(current);
                    cumulativeLogProbability += sampled.logProbability();
                    terminalReward = sampled.rewardScore();
                }
            }

            trajectories.add(new RecursiveReasoningTrajectory(
                    sampleIndex,
                    states,
                    cumulativeLogProbability,
                    terminalReward));
        }

        int selectedTrajectoryIndex = selectBestTrajectoryIndex(trajectories);
        RecursiveReasoningTrajectory selected = trajectories.get(selectedTrajectoryIndex);
        RecursiveReasoningRolloutSummary summary = new RecursiveReasoningRolloutSummary(
                trajectories.size(),
                trajectories.size(),
                selected.finalState().stateId(),
                Map.of(
                        "selector", selected.terminalRewardScore() != null ? "reward-then-logprob" : "logprob",
                        "selectedSampleIndex", selected.sampleIndex(),
                        "supervisionSteps", supervisionSteps,
                        "transitionsPerSupervisionStep", transitionsPerStep));
        return new RecursiveReasoningRolloutResult(trajectories, selectedTrajectoryIndex, summary);
    }

    private int selectBestTrajectoryIndex(List<RecursiveReasoningTrajectory> trajectories) {
        int bestIndex = 0;
        for (int i = 1; i < trajectories.size(); i++) {
            RecursiveReasoningTrajectory candidate = trajectories.get(i);
            RecursiveReasoningTrajectory best = trajectories.get(bestIndex);
            int rewardComparison = compareNullable(candidate.terminalRewardScore(), best.terminalRewardScore());
            if (rewardComparison > 0
                    || (rewardComparison == 0
                            && candidate.cumulativeLogProbability() > best.cumulativeLogProbability())) {
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private int compareNullable(Double left, Double right) {
        if (left == null && right == null) {
            return 0;
        }
        if (left == null) {
            return -1;
        }
        if (right == null) {
            return 1;
        }
        return Double.compare(left, right);
    }
}
