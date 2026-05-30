package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Evaluation context for one GRAM deep-supervision terminal prediction.
 */
public record GramTerminalLossInput(
        int supervisionStep,
        RecursiveReasoningState terminalState,
        RecursiveReasoningContext context,
        GramTrainingTarget target,
        List<GramVariationalTransitionResult> transitionsInStep) {

    public GramTerminalLossInput {
        if (supervisionStep < 0) {
            throw new IllegalArgumentException("supervisionStep must be >= 0 but was " + supervisionStep);
        }
        terminalState = Objects.requireNonNull(terminalState, "terminalState must not be null");
        context = Objects.requireNonNull(context, "context must not be null");
        target = Objects.requireNonNull(target, "target must not be null");
        transitionsInStep = List.copyOf(
                Objects.requireNonNull(transitionsInStep, "transitionsInStep must not be null"));
        if (transitionsInStep.isEmpty()) {
            throw new IllegalArgumentException("transitionsInStep must not be empty");
        }
    }

    public GramVariationalTransitionResult finalTransition() {
        return transitionsInStep.getLast();
    }
}
