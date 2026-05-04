package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

public final class ZeROOptimizer {
    private final CommBackend comm;
    private final DistributedContext ctx;

    public ZeROOptimizer(CommBackend comm, DistributedContext ctx) {
        this.comm = comm;
        this.ctx = ctx;
    }

public void step(Map<String, Tensor> params,
Map<String, Tensor> grads) {
for (String k : params.keySet()) {
    if (!ownsParam(k)) continue;
    Tensor g = grads.get(k);
    // gather full grad
    Tensor fullGrad = comm.allReduce(g);
    // update shard only
    Tensor p = params.get(k);
    p = p.sub(fullGrad.mul(0.001f)); // simplified
    params.put(k, p);
}

private boolean ownsParam(String key) {
return Math.abs(key.hashCode()) % ctx.worldSize() == ctx.rank();
}
}
}