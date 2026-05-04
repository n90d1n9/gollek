package tech.kayys.gollek.runtime.optimizer;

public final class ExecutionEngine {
    private final KernelExecutor executor;

    public Map<String, Tensor> run(
            ExecutablePlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session) {
        Map<String, Tensor> values = new HashMap<>(inputs);
        for (PlannedStep step : plan.steps) {
            handleControl(session);
            Tensor[] in = step.op.inputs().stream()
                    .map(ref -> values.get(ref.id().id()))
                    .toArray(Tensor[]::new);
            Tensor[] out = executor.execute(
                    step.kernelId,
                    step.device,
                    in,
                    step.op.attrs());
            for (int i = 0; i < step.op.outputs().size(); i++) {
                values.put(step.op.outputs().get(i).id(), out[i]);
            }
        }
        return values;
    }
}