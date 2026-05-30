package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Backend-neutral orchestration for GRAM prior/posterior stochastic transitions.
 */
public final class GramVariationalTransition {
    private final GramDeterministicTransition deterministicTransition;
    private final GramTransitionDistributionHead priorHead;
    private final GramTransitionDistributionHead posteriorHead;
    private final GramNoiseSampler noiseSampler;
    private final GramNextStateFactory nextStateFactory;

    public GramVariationalTransition(
            GramDeterministicTransition deterministicTransition,
            GramTransitionDistributionHead priorHead,
            GramTransitionDistributionHead posteriorHead,
            GramNoiseSampler noiseSampler,
            GramNextStateFactory nextStateFactory) {
        this.deterministicTransition =
                Objects.requireNonNull(deterministicTransition, "deterministicTransition must not be null");
        this.priorHead = Objects.requireNonNull(priorHead, "priorHead must not be null");
        this.posteriorHead = posteriorHead;
        this.noiseSampler = Objects.requireNonNull(noiseSampler, "noiseSampler must not be null");
        this.nextStateFactory = Objects.requireNonNull(nextStateFactory, "nextStateFactory must not be null");
    }

    public GramVariationalTransitionResult samplePrior(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context) {
        Tensor deterministicProposal = deterministicTransition.propose(previousState, context);
        return sample(GramTransitionInput.prior(previousState, context, deterministicProposal));
    }

    public GramVariationalTransitionResult samplePosterior(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context,
            Tensor targetEmbedding) {
        if (posteriorHead == null) {
            throw new IllegalStateException("posteriorHead is required for posterior transitions");
        }
        Tensor deterministicProposal = deterministicTransition.propose(previousState, context);
        return sample(GramTransitionInput.posterior(previousState, context, deterministicProposal, targetEmbedding));
    }

    public StochasticLatentTransition asPriorTransition() {
        return (previousState, context) -> samplePrior(previousState, context).toRolloutTransitionResult();
    }

    private GramVariationalTransitionResult sample(GramTransitionInput input) {
        GramLatentGaussian prior = Objects.requireNonNull(priorHead.distribution(input), "priorHead returned null");
        GramLatentGaussian posterior = null;
        GramLatentGaussian sampledDistribution = prior;
        Tensor klDivergence = null;

        if (input.mode() == GramTransitionMode.POSTERIOR) {
            posterior = Objects.requireNonNull(posteriorHead.distribution(input), "posteriorHead returned null");
            sampledDistribution = posterior;
            klDivergence = GramGaussianKl.meanPosteriorToPrior(posterior, prior);
        }

        Tensor epsilon = Objects.requireNonNull(
                noiseSampler.sampleEpsilon(sampledDistribution, input),
                "noiseSampler returned null");
        Tensor latentState = GramReparameterization.sample(sampledDistribution, epsilon);
        GramTransitionSample sample = new GramTransitionSample(
                input.mode(),
                prior,
                posterior,
                sampledDistribution,
                epsilon,
                latentState,
                klDivergence,
                Map.of(
                        "mode", input.mode().name(),
                        "taskId", input.context().taskId()));
        RecursiveReasoningState nextState = Objects.requireNonNull(
                nextStateFactory.nextState(input.previousState(), input.context(), sample),
                "nextStateFactory returned null");
        return new GramVariationalTransitionResult(sample, nextState, 0.0, null);
    }
}
