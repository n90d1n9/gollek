package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.layer.Sequential;

/**
 * Tensor and module construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlTensorFactoryFacade extends GollekDlDataFacade {
    protected GollekDlTensorFactoryFacade() {
    }

    public static GradTensor tensor(float[] data, long... shape) {
        return Gollek.tensor(data, shape);
    }

    public static NNModule sequential(NNModule... layers) {
        return new Sequential(layers);
    }
}
