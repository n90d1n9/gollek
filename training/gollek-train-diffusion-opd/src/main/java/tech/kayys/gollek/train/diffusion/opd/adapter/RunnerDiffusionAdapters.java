package tech.kayys.gollek.train.diffusion.opd.adapter;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.model.UNetModel;
import tech.kayys.gollek.diffusion.scheduler.Scheduler;
import tech.kayys.gollek.train.diffusion.api.DiffusionDenoiser;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;

/**
 * Adapters from the Java diffusion runner contracts to the Java diffusion OPD
 * training contracts.
 *
 * <p>This keeps DiffusionOPD training Java-native while reusing existing
 * Gollek diffusion runner surfaces instead of introducing a separate bridge
 * layer for the algorithm itself. Use this adapter path when the source model
 * and scheduler already speak core {@link Tensor}; the safetensor-native
 * bridge lives in {@link StableDiffusionRunnerAdapters}.
 */
public final class RunnerDiffusionAdapters {

    private RunnerDiffusionAdapters() {
    }

    public static DiffusionDenoiser denoiser(UNetModel model) {
        Objects.requireNonNull(model, "model must not be null");
        return new DiffusionDenoiser() {
            @Override
            public Tensor predict(Tensor latents, Tensor conditioning, int timestep) {
                return model.predict(latents, conditioning, timestep);
            }
        };
    }

    public static DiffusionScheduler scheduler(Scheduler scheduler) {
        Objects.requireNonNull(scheduler, "scheduler must not be null");
        return new DiffusionScheduler() {
            @Override
            public Tensor step(Tensor xT, Tensor modelPrediction, int timestepIndex) {
                return scheduler.step(xT, modelPrediction, timestepIndex);
            }

            @Override
            public int[] timesteps() {
                return scheduler.timesteps();
            }
        };
    }
}
