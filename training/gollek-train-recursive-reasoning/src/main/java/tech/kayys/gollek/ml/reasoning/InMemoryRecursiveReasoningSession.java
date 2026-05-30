package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Tiny in-memory session for recursive reasoning rollouts and demos.
 */
public final class InMemoryRecursiveReasoningSession implements RecursiveReasoningSession {
    private final RecursiveReasoningConfig config;
    private final RecursiveReasoningContext context;
    private final Tensor initialLatentState;
    private final StochasticLatentTransition transition;
    private final ReferenceRecursiveReasoningRolloutExecutor executor;

    private boolean stopped;
    private RecursiveReasoningRolloutResult lastResult;
    private TrainingSummary lastSummary;

    public InMemoryRecursiveReasoningSession(
            RecursiveReasoningContext context,
            Tensor initialLatentState,
            StochasticLatentTransition transition) {
        this(context, initialLatentState, transition, new ReferenceRecursiveReasoningRolloutExecutor());
    }

    public InMemoryRecursiveReasoningSession(
            RecursiveReasoningContext context,
            Tensor initialLatentState,
            StochasticLatentTransition transition,
            ReferenceRecursiveReasoningRolloutExecutor executor) {
        this.context = Objects.requireNonNull(context, "context must not be null");
        this.config = context.config();
        this.initialLatentState = Objects.requireNonNull(initialLatentState, "initialLatentState must not be null");
        this.transition = Objects.requireNonNull(transition, "transition must not be null");
        this.executor = Objects.requireNonNull(executor, "executor must not be null");
    }

    @Override
    public RecursiveReasoningConfig config() {
        return config;
    }

    @Override
    public int currentSupervisionStep() {
        if (lastResult == null) {
            return 0;
        }
        return lastResult.selectedTrajectory().finalState().supervisionStep();
    }

    @Override
    public int currentTransitionIndex() {
        if (lastResult == null) {
            return 0;
        }
        return lastResult.selectedTrajectory().finalState().transitionIndex();
    }

    @Override
    public int activeSampleCount() {
        if (lastResult == null) {
            return config.parallelSamples();
        }
        return lastResult.trajectories().size();
    }

    @Override
    public TrainingSummary fit() {
        if (stopped) {
            throw new IllegalStateException("session is stopped");
        }
        lastResult = executor.execute(initialLatentState, context, transition);
        RecursiveReasoningTrajectory selected = lastResult.selectedTrajectory();
        lastSummary = new TrainingSummary(
                config.supervisionSteps(),
                selected.terminalRewardScore() == null ? Double.NaN : -selected.terminalRewardScore(),
                selected.finalState().supervisionStep(),
                selected.terminalRewardScore(),
                selected.terminalRewardScore(),
                0L,
                Map.of(
                        "familyId", RecursiveReasoningModelFamily.FAMILY_ID,
                        "selectedStateId", selected.finalState().stateId(),
                        "selectedSampleIndex", selected.sampleIndex(),
                        "trajectoryCount", lastResult.trajectories().size()));
        return lastSummary;
    }

    @Override
    public RecursiveReasoningRolloutSummary rolloutSummary() {
        if (lastResult == null) {
            return new RecursiveReasoningRolloutSummary(
                    0,
                    0,
                    "not-run",
                    Map.of("status", "idle"));
        }
        return lastResult.summary();
    }

    public RecursiveReasoningRolloutResult rolloutResult() {
        return lastResult;
    }

    public RecursiveReasoningReport report() {
        if (lastResult == null) {
            throw new IllegalStateException("fit() must run before report()");
        }
        RecursiveReasoningTrajectory selected = lastResult.selectedTrajectory();
        return new RecursiveReasoningReport(
                "1",
                RecursiveReasoningModelFamily.FAMILY_ID,
                context.taskId(),
                config,
                lastResult.summary(),
                selected.finalState().stateId(),
                lastResult.selectedTrajectoryIndex(),
                selected.cumulativeLogProbability(),
                selected.terminalRewardScore(),
                Map.of(
                        "trajectoryCount", lastResult.trajectories().size(),
                        "selectedSampleIndex", selected.sampleIndex()));
    }

    @Override
    public boolean isStopped() {
        return stopped;
    }

    @Override
    public void stop() {
        stopped = true;
    }

    public TrainingSummary summary() {
        return lastSummary;
    }
}
