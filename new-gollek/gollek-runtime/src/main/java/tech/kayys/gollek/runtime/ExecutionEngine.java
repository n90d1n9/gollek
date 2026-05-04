package tech.kayys.gollek.runtime;

import tech.kayys.gollek.runtime.plan.*;
import tech.kayys.gollek.runtime.weight.WeightStore;
import tech.kayys.gollek.runtime.control.*;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.ir.GGraph;
import tech.kayys.gollek.ir.GOp;
import tech.kayys.gollek.ir.GValueId;
import tech.kayys.gollek.ir.GValueRef;
import tech.kayys.gollek.ir.OpKernel;
import tech.kayys.gollek.ir.OpRegistry;

import java.util.*;

/**
 * 
 * ExecutionPlan (static)
 * ↓
 * ExecutionEngine
 * ↓
 * ExecutionSession
 * ↓
 * ExecutionController
 * ↓
 * Pause / Resume / Cancel / Shutdown
 * 
 */
public final class ExecutionEngine {
    private final OpRegistry registry;

    public ExecutionEngine(OpRegistry registry) {
        this.registry = registry;
    }

    private final OpRegistry registry;

    public Map<String, Tensor> run(
            GGraph graph,
            Map<String, Tensor> inputs,
            WeightStore weights) {

        Map<String, Tensor> values = new HashMap<>(inputs);
        Map<String, Integer> refCounts = computeRefCounts(graph);

        for (GOp op : graph.ops()) {
            // Resolve inputs
            Tensor[] in = op.inputs().stream()
                    .map(ref -> {
                        Tensor t = values.get(ref.id().id());
                        if (t != null)
                            return t;
                        return weights.get(ref.id().id());
                    })
                    .toArray(Tensor[]::new);

            // Execute
            OpKernel kernel = registry.get(op.opType());
            Tensor[] out = kernel.compute(in, op.attrs());

            // Store outputs and decrement ref counts
            for (int i = 0; i < op.outputs().size(); i++) {
                String outId = op.outputs().get(i).id();
                values.put(outId, out[i]);
            }

            // Clean up inputs that are no longer needed
            for (GValueRef ref : op.inputs()) {
                String inId = ref.id().id();
                int newCount = refCounts.merge(inId, -1, Integer::sum);
                if (newCount == 0 && !inputs.containsKey(inId)) {
                    Tensor removed = values.remove(inId);
                    if (removed != null && removed.buffer() != null) {
                        removed.buffer().release(); // Free memory
                    }
                }
            }
        }

        // Return only outputs
        Map<String, Tensor> outputs = new HashMap<>();
        for (GValueId outId : graph.outputs()) {
            outputs.put(outId.id(), values.get(outId.id()));
        }
        return outputs;
    }

    private Map<String, Integer> computeRefCounts(GGraph graph) {
        Map<String, Integer> counts = new HashMap<>();
        for (GOp op : graph.ops()) {
            for (GValueRef ref : op.inputs()) {
                counts.merge(ref.id().id(), 1, Integer::sum);
            }
        }
        return counts;
    }

    private void handleControl(ExecutionSession session) {
        ExecutionController ctrl = session.controller;

        if (ctrl.isCancelled()) {
            throw new RuntimeException("Execution cancelled");
        }
        if (ctrl.isShutdown()) {
            throw new RuntimeException("Execution shutdown");
        }
        // 🔥 cooperative pause
        while (ctrl.isPaused()) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException ignored) {
            }
        }
    }

    public Map<String, Tensor> run(
            GGraph graph,
            Map<String, Tensor> inputs,
            WeightStore weights) {
        Map<String, Tensor> values = new HashMap<>(inputs);
        for (GOp op : graph.ops()) {
            Tensor[] in = op.inputs.stream()
                    .map(ref -> {
                        Tensor t = values.get(ref.name);
                        if (t != null)
                            return t;
                        // 🔥 resolve constant
                        return weights.get(ref.name);
                    })
                    .toArray(Tensor[]::new);
            OpKernel kernel = registry.get(op.opType);
            Tensor[] out = kernel.compute(in, op.attrs);
            for (int i = 0; i < op.outputs.size(); i++) {
                values.put(op.outputs.get(i), out[i]);
            }
        }
        return values;
    }
}