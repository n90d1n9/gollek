package tech.kayys.gollek.train;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

public final class AdamW implements Optimizer {
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final OptimizerState state = new OptimizerState();

    public AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step(Map<String, Tensor> params,
            Map<String, Tensor> grads) {
        state.step++;
        for (String k : params.keySet()) {
            Tensor p = params.get(k);
            Tensor g = grads.get(k);
            if (g == null)
                continue;
            // init moments
            state.m.putIfAbsent(k, g.zerosLike());
            state.v.putIfAbsent(k, g.zerosLike());
            Tensor m = state.m.get(k);
            Tensor v = state.v.get(k);
            // m = beta1 * m + (1 - beta1) * g
            m = m.mul(beta1).add(g.mul(1 - beta1));
            // v = beta2 * v + (1 - beta2) * g^2
            v = v.mul(beta2).add(g.mul(g).mul(1 - beta2));
            state.m.put(k, m);
            state.v.put(k, v);
            // bias correction
            float bias1 = 1 - (float) Math.pow(beta1, state.step);
            float bias2 = 1 - (float) Math.pow(beta2, state.step);
            Tensor mHat = m.div(bias1);
            Tensor vHat = v.div(bias2);
            // Adam update
            Tensor update = mHat.div(vHat.sqrt().add(eps));
            // decoupled weight decay
            p = p.sub(p.mul(lr * weightDecay));
            // final update
            p = p.sub(update.mul(lr));
            params.put(k, p);
        }
    }
}