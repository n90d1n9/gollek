package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Extracts final-state discrete token predictions from recursive rollouts.
 */
public final class DiscreteRolloutTokenCollector {
    private DiscreteRolloutTokenCollector() {
    }

    public static DiscreteRolloutTokenReport collectFinalStateTokens(
            RecursiveReasoningRolloutResult rollout,
            DiscreteStateTokenDecoder decoder) {
        Objects.requireNonNull(rollout, "rollout must not be null");
        Objects.requireNonNull(decoder, "decoder must not be null");

        List<DiscreteTrajectoryTokenPrediction> predictions = new ArrayList<>();
        for (int trajectoryIndex = 0; trajectoryIndex < rollout.trajectories().size(); trajectoryIndex++) {
            RecursiveReasoningTrajectory trajectory = rollout.trajectories().get(trajectoryIndex);
            RecursiveReasoningState finalState = trajectory.finalState();
            int[] tokens = Objects.requireNonNull(
                    decoder.decodeTokens(finalState),
                    "decoder returned null tokens for state " + finalState.stateId());
            predictions.add(new DiscreteTrajectoryTokenPrediction(
                    trajectoryIndex,
                    trajectory.sampleIndex(),
                    finalState.stateId(),
                    tokens,
                    trajectoryMetadata(trajectory, finalState)));
        }

        return new DiscreteRolloutTokenReport(
                predictions,
                rollout.selectedTrajectoryIndex(),
                reportMetadata(rollout));
    }

    private static Map<String, Object> trajectoryMetadata(
            RecursiveReasoningTrajectory trajectory,
            RecursiveReasoningState finalState) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("stateSupervisionStep", finalState.supervisionStep());
        metadata.put("stateTransitionIndex", finalState.transitionIndex());
        metadata.put("stateSampleIndex", finalState.sampleIndex());
        metadata.put("cumulativeLogProbability", trajectory.cumulativeLogProbability());
        if (trajectory.terminalRewardScore() != null) {
            metadata.put("terminalRewardScore", trajectory.terminalRewardScore());
        }
        return metadata;
    }

    private static Map<String, Object> reportMetadata(RecursiveReasoningRolloutResult rollout) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("exploredTrajectoryCount", rollout.summary().exploredTrajectoryCount());
        metadata.put("completedTrajectoryCount", rollout.summary().completedTrajectoryCount());
        metadata.put("selectedSampleIndex", rollout.selectedTrajectory().sampleIndex());
        if (rollout.summary().selectedStateId() != null) {
            metadata.put("selectedStateId", rollout.summary().selectedStateId());
        }
        if (!rollout.summary().metadata().isEmpty()) {
            metadata.put("rolloutSummary", rollout.summary().metadata());
        }
        return metadata;
    }
}
