package tech.kayys.gollek.distributed;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.ir.*;
import java.util.Map;

/**
 * Distributed trainer - requires gollek-train module at runtime.
 * Stubbed for compilation.
 */
public final class DistributedTrainer {
    private final Object base; // Trainer - from training module
    private final DDPWrapper ddp;

    public DistributedTrainer(Object base, DDPWrapper ddp) {
        this.base = base;
        this.ddp = ddp;
    }

    public void trainStep(GGraph model,
            GValueId loss,
            Map<String, Tensor> params,
            Map<String, Tensor> inputs) {
        throw new UnsupportedOperationException(
                "DistributedTrainer requires gollek-train module");
    }
}