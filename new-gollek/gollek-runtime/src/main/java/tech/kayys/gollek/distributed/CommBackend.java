package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;

public interface CommBackend {
    Tensor allReduce(Tensor t);

    Tensor broadcast(Tensor t, int src);

    Tensor allGather(Tensor t);

    void barrier();
}