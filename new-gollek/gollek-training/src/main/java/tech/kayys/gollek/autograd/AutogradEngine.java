package tech.kayys.gollek.autograd;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class AutogradEngine {
    private final GradRegistry registry;

    public AutogradEngine(GradRegistry registry) {
        this.registry = registry;
    }

    public GGraph buildBackward(GGraph forward, GValueId loss) {
        GradContext ctx = new GradContext();
        // seed: dLoss/dLoss = 1
        GValueId lossGrad = new GValueId(loss.id() + "_grad");
        ctx.accumulate(loss, lossGrad);
        Map<GValueId, Integer> useCount = GradGraphAnalyzer.buildUseCount(forward);
        Map<GValueId, GOp> producer = GradGraphAnalyzer.buildProducerMap(forward);
        Deque<GValueId> queue = new ArrayDeque<>();
        queue.add(loss);
        while (!queue.isEmpty()) {
            GValueId value = queue.poll();
            if (!ctx.hasGrad(value))
                continue;
            GOp op = producer.get(value);
            if (op == null)
                continue;
            // skip if no grad needed
            if (!requiresGrad(op))
                continue;
            GradFn fn = registry.get(op.opType());
            if (fn == null)
                continue;
            GValueId gradOut = ctx.getGrad(value);
            Map<GValueId, GValueId> grads = fn.backward(op, gradOut, ctx);
            for (Map.Entry<GValueId, GValueId> e : grads.entrySet()) {
                GValueId input = e.getKey();
                GValueId grad = e.getValue();
                ctx.accumulate(input, grad);
                int remaining = useCount.merge(input,
                        -1, Integer::sum);
                if (remaining == 0) {
                    queue.add(input);
                }
            }
        }
        return new GGraph(
                ctx.ops(),
                forward.outputs(),
                forward.inputs());
    }

    private boolean requiresGrad(GOp op) {
        // simple version: assume true
        return true;
    }
}