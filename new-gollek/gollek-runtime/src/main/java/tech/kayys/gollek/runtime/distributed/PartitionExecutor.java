package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.runtime.execution.ExecutionContext;

public interface PartitionExecutor {
    void execute(
            Partition partition,
            ExecutionContext context);
}