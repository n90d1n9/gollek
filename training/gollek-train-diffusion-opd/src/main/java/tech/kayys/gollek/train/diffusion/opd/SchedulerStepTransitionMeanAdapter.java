package tech.kayys.gollek.train.diffusion.opd;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;

/**
 * Adapts a scheduler step into the transition-mean view used by OPD.
 *
 * <p>This is the default bridge when a {@link DiffusionScheduler} already
 * exposes the executable transition step directly and no extra backend
 * translation layer is needed.
 */
public final class SchedulerStepTransitionMeanAdapter implements TransitionMeanAdapter {
    private final DiffusionScheduler scheduler;

    public SchedulerStepTransitionMeanAdapter(DiffusionScheduler scheduler) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler must not be null");
    }

    /**
     * Delegates directly to the scheduler step so OPD can supervise the scheduler-native
     * transition mean without another translation layer.
     */
    @Override
    public Tensor transitionMean(Tensor xT, Tensor modelPrediction, int timestepIndex) {
        return scheduler.step(xT, modelPrediction, timestepIndex);
    }

    /**
     * Uses unit variance for scheduler-step adapters that only expose the mean path directly.
     */
    @Override
    public float stepVariance(int timestepIndex) {
        return 1.0f;
    }
}
