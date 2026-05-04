package tech.kayys.gollek.multitenancy;

public final class MultiTenantScheduler {
    private final ExecutorService pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public Future<Map<String, Tensor>> submit(
            ExecutionOrchestrator orchestrator,
            ExecutionRequest req) {
        return pool.submit(() -> orchestrator.run(req.plan, req.inputs, req.session));
    }
}