package tech.kayys.gollek.ml;

import java.util.List;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.SGD;

/**
 * Core optimizer construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlCoreOptimizerFactoryFacade extends GollekDlTensorFactoryFacade {
    protected GollekDlCoreOptimizerFactoryFacade() {
    }

    public static Optimizer sgd(List<Parameter> params, float lr) {
        return SGD.builder(params, lr).build();
    }

    public static Optimizer sgd(List<Parameter> params, float lr, float momentum) {
        return SGD.builder(params, lr)
                .momentum(momentum)
                .build();
    }

    public static Optimizer sgd(
            List<Parameter> params,
            float lr,
            float momentum,
            float weightDecay,
            boolean nesterov) {
        return SGD.builder(params, lr)
                .momentum(momentum)
                .weightDecay(weightDecay)
                .nesterov(nesterov)
                .build();
    }
}
