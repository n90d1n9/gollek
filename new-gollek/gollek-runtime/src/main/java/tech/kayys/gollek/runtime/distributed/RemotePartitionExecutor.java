package tech.kayys.gollek.runtime.distributed;

public final class RemotePartitionExecutor implements PartitionExecutor {
    private final RemoteClient client;

    public RemotePartitionExecutor(RemoteClient client) {
        this.client = client;
    }

    @Override
    public Map<String, Tensor> execute(
            Partition partition,
            Map<String, Tensor> inputs) {
        return client.executePartition(partition, inputs);
    }
}