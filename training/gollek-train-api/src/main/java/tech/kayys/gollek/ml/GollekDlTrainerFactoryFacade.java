package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.Acceleration;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Trainer and runtime construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlTrainerFactoryFacade extends GollekDlLossFactoryFacade {
    protected GollekDlTrainerFactoryFacade() {
    }

    public static CanonicalTrainer.Builder trainer() {
        return CanonicalTrainer.builder();
    }

    public static tech.kayys.gollek.train.diffusion.opd.DefaultDiffusionOpdTrainer.Builder diffusionOpdTrainer() {
        return tech.kayys.gollek.train.diffusion.opd.DefaultDiffusionOpdTrainer.builder();
    }

    public static Acceleration.BackendStatus accelerationStatus() {
        return Acceleration.status();
    }

    public static Acceleration.BackendStatus accelerationStatus(String deviceId) {
        return Acceleration.status(deviceId);
    }
}
