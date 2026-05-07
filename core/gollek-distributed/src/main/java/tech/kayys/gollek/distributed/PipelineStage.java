package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;

public interface PipelineStage {
    Tensor forward(Tensor input);

    Tensor backward(Tensor grad);
}