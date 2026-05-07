package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;

public final class TensorParallelLinear {
    private final CommBackend comm;

    public TensorParallelLinear(CommBackend comm) {
        this.comm = comm;
    }

    public Tensor forward(Tensor x, Tensor localWeight) {
        // partial output
        Tensor out = x.matmul(localWeight);
        // gather all parts
        return comm.allGather(out);
    }
}