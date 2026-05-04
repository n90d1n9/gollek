package tech.kayys.gollek.runtime.distributed;

public final class LocalPartitionExecutor implements PartitionExecutor {
    private final ExecutionEngine engine;

    public LocalPartitionExecutor(ExecutionEngine engine) {
        this.engine = engine;
    }

    @Override
    public Map<String, Tensor> execute(
            Partition partition,
            Map<String, Tensor> inputs) {
        return engine.runPartition(partition, inputs);
    }
}