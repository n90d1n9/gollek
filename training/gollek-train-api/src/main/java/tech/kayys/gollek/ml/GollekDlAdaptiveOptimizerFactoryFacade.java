package tech.kayys.gollek.ml;

import java.util.List;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.Adam;
import tech.kayys.gollek.ml.optim.AdamW;
import tech.kayys.gollek.ml.optim.NAdam;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.RMSprop;

/**
 * Adaptive optimizer construction helpers inherited by {@link Gollek.DL}.
 */
public class GollekDlAdaptiveOptimizerFactoryFacade extends GollekDlCoreOptimizerFactoryFacade {
    protected GollekDlAdaptiveOptimizerFactoryFacade() {
    }

    public static Optimizer adam(List<Parameter> params, float lr) {
        return Adam.builder(params, lr).build();
    }

    public static Optimizer adam(List<Parameter> params, float lr, float weightDecay) {
        return Adam.builder(params, lr)
                .weightDecay(weightDecay)
                .build();
    }

    public static Optimizer adam(
            List<Parameter> params,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            boolean amsgrad) {
        return Adam.builder(params, lr)
                .betas(beta1, beta2)
                .eps(eps)
                .weightDecay(weightDecay)
                .amsgrad(amsgrad)
                .build();
    }

    public static Optimizer adamW(List<Parameter> params, float lr) {
        return AdamW.builder(params, lr).build();
    }

    public static Optimizer adamW(List<Parameter> params, float lr, float weightDecay) {
        return AdamW.builder(params, lr)
                .weightDecay(weightDecay)
                .build();
    }

    public static Optimizer adamW(
            List<Parameter> params,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            boolean amsgrad) {
        return AdamW.builder(params, lr)
                .betas(beta1, beta2)
                .eps(eps)
                .weightDecay(weightDecay)
                .amsgrad(amsgrad)
                .build();
    }

    public static Optimizer rmsprop(List<Parameter> params, float lr) {
        return RMSprop.builder(params, lr).build();
    }

    public static Optimizer rmsprop(
            List<Parameter> params,
            float lr,
            float alpha,
            float eps,
            float weightDecay,
            float momentum) {
        return RMSprop.builder(params, lr)
                .alpha(alpha)
                .eps(eps)
                .weightDecay(weightDecay)
                .momentum(momentum)
                .build();
    }

    public static Optimizer nadam(List<Parameter> params, float lr) {
        return NAdam.builder(params, lr).build();
    }

    public static Optimizer nadam(List<Parameter> params, float lr, float weightDecay) {
        return NAdam.builder(params, lr).weightDecay(weightDecay).build();
    }

    public static Optimizer nadamW(List<Parameter> params, float lr, float weightDecay) {
        return NAdam.builder(params, lr)
                .weightDecay(weightDecay)
                .decoupledWeightDecay(true)
                .build();
    }
}
