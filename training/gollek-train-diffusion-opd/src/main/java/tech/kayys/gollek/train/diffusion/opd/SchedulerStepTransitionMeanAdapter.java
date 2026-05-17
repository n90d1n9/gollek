package tech.kayys.gollek.train.diffusion.opd;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;

/**
 * Adapts a scheduler step into the transition-mean view used by OPD.
 */
public final class SchedulerStepTransitionMeanAdapter implements TransitionMeanAdapter {
    private final DiffusionScheduler scheduler;

    public SchedulerStepTransitionMeanAdapter(DiffusionScheduler scheduler) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler must not be null");
    }

    @Override
    public Tensor transitionMean(Tensor xT, Tensor modelPrediction, int timestepIndex) {
        return scheduler.step(xT, modelPrediction, timestepIndex);
    }

    @Override
    public float stepVariance(int timestepIndex) {
        return 1.0f;
    }
}
