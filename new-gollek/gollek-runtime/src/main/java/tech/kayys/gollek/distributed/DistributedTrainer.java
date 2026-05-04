package tech.kayys.gollek.distributed;

package tech.kayys.gollek.distributed;Diffuser Improvement Suggestions

import tech.kayys.gollek.train.*;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.ir.*;
import java.util.Map;

public final class DistributedTrainer {
    private final Trainer base;
    private final DDPWrapper ddp;

    public DistributedTrainer(Trainer base,
            DDPWrapper ddp) {
        this.base = base;
        this.ddp = ddp;
    }

    public void trainStep(GGraph model,
            GValueId loss,
            Map<String, Tensor> params,
            Map<String, Tensor> inputs) {
        // forward + backward
        Map<String, Tensor> grads = baseStep(model, loss, params, inputs);
        // sync gradients
        grads = ddp.syncGradients(grads);
        // optimizer step
        base.optimizer().step(params, grads);
    }

    private Map<String, Tensor> baseStep(
            GGraph model,
            GValueId loss,
            Map<String, Tensor> params,
            Map<String, Tensor> inputs) {
        // call into existing trainer but expose grads
        // (you’ll refactor Trainer slightly to return grads)
        return base.computeGradients(model, loss, params, inputs);
    }
}