package tech.kayys.gollek.engine.inference;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.KVCacheState;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.plugin.optimization.DefaultExecutionContext;
import tech.kayys.gollek.plugin.optimization.OptimizationPluginManager;
import tech.kayys.gollek.plugin.optimization.ExecutionContext;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Default scheduler that supports continuous batching.
 * Incorporates VRAM memory management to admit or delay sequences.
 */
public class DefaultBatchScheduler {

    private final Queue<InferenceRequest> pendingRequests = new ConcurrentLinkedQueue<>();
    private final RunnerSession runnerSession;
    private static final double MAX_VRAM_UTILIZATION = 0.95; // Hard limit before rejecting/delaying

    public DefaultBatchScheduler(RunnerSession runnerSession) {
        this.runnerSession = runnerSession;
    }

    /**
     * Submit a new sequence for inference.
     */
    public void submit(InferenceRequest request) {
        pendingRequests.offer(request);
    }

    /**
     * Continuous execution loop logic.
     * Checks VRAM usage before admitting new sequences to the batch.
     */
    public void runContinuous() {
        while (!pendingRequests.isEmpty()) {
            // Apply optimization plugins (e.g., KV cache offloading) before checking VRAM
            ExecutionContext optContext = new DefaultExecutionContext(runnerSession);
            OptimizationPluginManager.getInstance().applyOptimizations(optContext);
            KVCacheState cacheState = runnerSession.getKVCacheState();
            if (cacheState != null && cacheState.getVramUtilization() >= MAX_VRAM_UTILIZATION) {
                // VRAM is full, delay admission of new sequences
                // Wait for ongoing sequences to finish and free KV cache
                System.out.println("VRAM limit reached, delaying sequence admission...");
                break; 
            }

            InferenceRequest nextRequest = pendingRequests.poll();
            if (nextRequest != null) {
                // Admit sequence to active batch
                admitSequence(nextRequest);
            }
        }
    }

    private void admitSequence(InferenceRequest request) {
        // Logic to dispatch the request to the active decoding batch
        // (Implementation details omitted for brevity)
    }
}
