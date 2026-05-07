package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

// DDP (DATA PARALLEL)
public final class DDPWrapper {
    private final CommBackend comm;
    private final DistributedContext ctx;

    public DDPWrapper(CommBackend comm, DistributedContext ctx) {
        this.comm = comm;
        this.ctx = ctx;
    }

    public Map<String, Tensor> syncGradients(Map<String, Tensor> grads) {
        for (String k : grads.keySet()) {
            Tensor g = grads.get(k);
            // sum across all nodes
            Tensor reduced = comm.allReduce(g);
            // average
            grads.put(k, reduced.div(ctx.worldSize()));
        }
        return grads;
    }
}