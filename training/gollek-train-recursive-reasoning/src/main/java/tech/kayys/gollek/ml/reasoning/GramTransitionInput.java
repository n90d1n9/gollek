package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Input bundle shared by GRAM prior, posterior, and sampler adapters.
 */
public record GramTransitionInput(
        RecursiveReasoningState previousState,
        RecursiveReasoningContext context,
        Tensor deterministicProposal,
        Tensor targetEmbedding,
        GramTransitionMode mode) {

    public GramTransitionInput {
        previousState = Objects.requireNonNull(previousState, "previousState must not be null");
        context = Objects.requireNonNull(context, "context must not be null");
        deterministicProposal = Objects.requireNonNull(deterministicProposal, "deterministicProposal must not be null");
        mode = Objects.requireNonNull(mode, "mode must not be null");
        if (mode == GramTransitionMode.POSTERIOR && targetEmbedding == null) {
            throw new IllegalArgumentException("targetEmbedding must not be null for posterior transitions");
        }
    }

    public static GramTransitionInput prior(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            Tensor deterministicProposal) {
        return new GramTransitionInput(
                previousState,
                context,
                deterministicProposal,
                null,
                GramTransitionMode.PRIOR);
    }

    public static GramTransitionInput posterior(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            Tensor deterministicProposal,
            Tensor targetEmbedding) {
        return new GramTransitionInput(
                previousState,
                context,
                deterministicProposal,
                Objects.requireNonNull(targetEmbedding, "targetEmbedding must not be null"),
                GramTransitionMode.POSTERIOR);
    }
}
