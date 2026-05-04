package tech.kayys.gollek.runtime.pipeline;

import tech.kayys.gollek.runtime.distributed.*;
import tech.kayys.gollek.core.tensor.Tensor;
import java.util.*;
import java.util.concurrent.*;

public final class AsyncPipelineExecutor {
    private final Map<Device, PartitionExecutor> executors;
    private final ExecutorService pool = Executors.newCachedThreadPool();

    public AsyncPipelineExecutor(Map<Device, PartitionExecutor> executors) {
        this.executors = executors;
    }

    public Map<String, Tensor> run(PartitionedPlan plan,
            Map<String, Tensor> inputs) {
        List<Partition> parts = plan.partitions;
        List<Future<Map<String, Tensor>>> futures = new ArrayList<>();
        Map<String, Tensor> current = inputs;
        for (Partition p : parts) {
            Map<String, Tensor> inputSnapshot = new HashMap<>(current);
            Future<Map<String, Tensor>> f = pool.submit(() -> {
                PartitionExecutor exec = executors.get(p.device);
                return exec.execute(p, inputSnapshot);
            });
            futures.add(f);
            try {
                current = f.get(); // simple version (sync boundary)
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        return current;
    }
}