package tech.kayys.gollek.train;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

public final class SGD implements Optimizer {
    private final float lr;

    public SGD(float lr) {
        this.lr = lr;
    }

    @Override
    public void step(Map<String, Tensor> params,
            Map<String, Tensor> grads) {
        for (String k : params.keySet()) {
            Tensor p = params.get(k);
            Tensor g = grads.get(k);
            if (g == null)
                continue;
            params.put(k, p.sub(g.mul(lr)));
        }
    }
}