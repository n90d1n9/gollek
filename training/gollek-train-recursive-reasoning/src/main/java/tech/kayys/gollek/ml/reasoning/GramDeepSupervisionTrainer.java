package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Reference GRAM deep-supervision adapter for posterior rollouts and objective aggregation.
 */
public final class GramDeepSupervisionTrainer {
    private final GramVariationalTransition transition;
    private final GramTerminalLossEvaluator lossEvaluator;
    private final GramObjectiveConfig objectiveConfig;

    public GramDeepSupervisionTrainer(
            GramVariationalTransition transition,
            GramTerminalLossEvaluator lossEvaluator,
            GramObjectiveConfig objectiveConfig) {
        this.transition = Objects.requireNonNull(transition, "transition must not be null");
        this.lossEvaluator = Objects.requireNonNull(lossEvaluator, "lossEvaluator must not be null");
        this.objectiveConfig = Objects.requireNonNull(objectiveConfig, "objectiveConfig must not be null");
    }

    public GramDeepSupervisionTrainingResult fit(
            RecursiveReasoningState initialState,
            RecursiveReasoningContext context,
            GramTrainingTarget target) {
        Objects.requireNonNull(initialState, "initialState must not be null");
        Objects.requireNonNull(context, "context must not be null");
        Objects.requireNonNull(target, "target must not be null");

        RecursiveReasoningConfig config = context.config();
        List<RecursiveReasoningState> states = new ArrayList<>();
        List<GramVariationalTransitionResult> allTransitions = new ArrayList<>();
        List<GramSupervisionStepLoss> supervisionLosses = new ArrayList<>();
        states.add(initialState);
        RecursiveReasoningState current = initialState;

        for (int supervisionStep = 0; supervisionStep < config.supervisionSteps(); supervisionStep++) {
            List<GramVariationalTransitionResult> stepTransitions = new ArrayList<>();
            for (int transitionIndex = 1;
                    transitionIndex <= config.transitionsPerSupervisionStep();
                    transitionIndex++) {
                GramVariationalTransitionResult raw =
                        transition.samplePosterior(current, context, target.targetEmbedding());
                GramVariationalTransitionResult normalized =
                        normalizeTransition(raw, supervisionStep, transitionIndex);
                stepTransitions.add(normalized);
                allTransitions.add(normalized);
                current = normalized.nextState();
                states.add(current);
            }

            GramTerminalLossInput lossInput = new GramTerminalLossInput(
                    supervisionStep,
                    current,
                    context,
                    target,
                    stepTransitions);
            GramTerminalLossTerms terms = Objects.requireNonNull(
                    lossEvaluator.evaluate(lossInput),
                    "lossEvaluator returned null");
            supervisionLosses.add(new GramSupervisionStepLoss(
                    supervisionStep,
                    terms.reconstructionNll(),
                    klForStep(stepTransitions),
                    terms.latentProcessRewardLoss(),
                    terms.adaptiveComputationLoss(),
                    stepMetadata(terms, stepTransitions)));
        }

        GramObjectiveBreakdown objective = GramObjective.evaluate(supervisionLosses, objectiveConfig);
        RecursiveReasoningTrajectory trajectory = new RecursiveReasoningTrajectory(
                initialState.sampleIndex(),
                states,
                allTransitions.stream().mapToDouble(GramVariationalTransitionResult::logProbability).sum(),
                allTransitions.isEmpty() ? null : allTransitions.getLast().rewardScore());
        TrainingSummary summary = new TrainingSummary(
                config.supervisionSteps(),
                objective.totalLoss(),
                bestStepIndex(supervisionLosses),
                objective.totalLoss(),
                objective.totalLoss(),
                0L,
                Map.of(
                        "familyId", RecursiveReasoningModelFamily.FAMILY_ID,
                        "objective", objective.metadata().get("objective"),
                        "transitionCount", allTransitions.size(),
                        "supervisionSteps", supervisionLosses.size(),
                        "truncatedSurrogate", objectiveConfig.truncatedSurrogate()));
        return new GramDeepSupervisionTrainingResult(
                trajectory,
                allTransitions,
                objective,
                summary,
                Map.of(
                        "taskId", context.taskId(),
                        "targetMetadata", target.metadata()));
    }

    private GramVariationalTransitionResult normalizeTransition(
            GramVariationalTransitionResult raw,
            int supervisionStep,
            int transitionIndex) {
        RecursiveReasoningState rawState = raw.nextState();
        Map<String, Object> metadata = new HashMap<>(rawState.metadata());
        metadata.put("supervisionStep", supervisionStep);
        metadata.put("transitionIndex", transitionIndex);
        metadata.put("normalizedBy", "gram-deep-supervision-trainer");
        RecursiveReasoningState normalizedState = new RecursiveReasoningState(
                rawState.stateId(),
                supervisionStep,
                transitionIndex,
                rawState.sampleIndex(),
                rawState.latentState(),
                metadata);
        return new GramVariationalTransitionResult(
                raw.sample(),
                normalizedState,
                raw.logProbability(),
                raw.rewardScore());
    }

    private double klForStep(List<GramVariationalTransitionResult> stepTransitions) {
        if (objectiveConfig.truncatedSurrogate()) {
            return klValue(stepTransitions.getLast());
        }
        return stepTransitions.stream().mapToDouble(this::klValue).sum();
    }

    private double klValue(GramVariationalTransitionResult transitionResult) {
        if (transitionResult.sample().klDivergence() == null) {
            return 0.0;
        }
        double value = transitionResult.sample().klDivergence().item();
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException("KL divergence must be finite and >= 0 but was " + value);
        }
        return value;
    }

    private Map<String, Object> stepMetadata(
            GramTerminalLossTerms terms,
            List<GramVariationalTransitionResult> stepTransitions) {
        Map<String, Object> metadata = new HashMap<>(terms.metadata());
        metadata.put("transitionCount", stepTransitions.size());
        metadata.put("klAggregation", objectiveConfig.truncatedSurrogate() ? "final-transition" : "all-transitions");
        metadata.put("terminalStateId", stepTransitions.getLast().nextState().stateId());
        return metadata;
    }

    private int bestStepIndex(List<GramSupervisionStepLoss> supervisionLosses) {
        int best = 0;
        double bestLoss = supervisionLosses.getFirst().weightedLoss(objectiveConfig);
        for (int i = 1; i < supervisionLosses.size(); i++) {
            double candidate = supervisionLosses.get(i).weightedLoss(objectiveConfig);
            if (candidate < bestLoss) {
                best = i;
                bestLoss = candidate;
            }
        }
        return best;
    }
}
