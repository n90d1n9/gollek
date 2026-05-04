package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.core.tensor.Device;

import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.*;

public final class AsyncPipelineExecutor {
    private final Map<Device, PartitionExecutor> executors;
    private final ExecutorService pool = Executors.newCachedThreadPool();

    public AsyncPipelineExecutor(Map<Device, PartitionExecutor> executors) {
        this.executors = executors;
    }

    public CompletableFuture<Map<String, Tensor>> execute(PartitionedPlan plan, Map<String, Tensor> initialInputs) {
        CompletableFuture<Map<String, Tensor>> future = CompletableFuture.completedFuture(initialInputs);

        for (Partition partition : plan.partitions()) {
            PartitionExecutor executor = executors.get(partition.device());
            if (executor == null) {
                throw new RuntimeException("No executor found for device: " + partition.device());
            }

            // Chain the partitions asynchronously
            future = future.thenComposeAsync(inputs -> {
                try {
                    return CompletableFuture.completedFuture(executor.execute(partition, inputs));
                } catch (Exception e) {
                    throw new CompletionException(e);
                }
            }, pool);
        }

        return future;
    }
    
    public void shutdown() {
        pool.shutdown();
    }
}
