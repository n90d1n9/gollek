package tech.kayys.gollek.train;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

public interface Optimizer {
    void step(Map<String, Tensor> params,
            Map<String, Tensor> grads);
}