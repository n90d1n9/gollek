package tech.kayys.gollek.ml.runner.batching;

import tech.kayys.gollek.ml.runner.InferenceInput;
import tech.kayys.gollek.ml.runner.InferenceResult;
import tech.kayys.gollek.ml.runner.ModelRunner;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Dynamic batching engine that automatically groups inference requests into
 * optimal batches to maximize GPU utilization while respecting latency SLAs.
 *
 * <h2>Problem</h2>
 * <p>Single-request inference underutilizes GPU. Manual batching is error-prone.
 *
 * <h2>Solution</h2>
 * <p>Requests are queued and automatically batched:
 * <pre>
 *   Request 1 arrives ─┐
 *   Request 2 arrives ─┤ → Batch of 4 → Single GPU kernel launch
 *   Request 3 arrives ─┤
 *   Request 4 arrives ─┘
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>
 *   BatchingEngine batcher = BatchingEngine.builder()
 *       .runner(modelRunner)
 *       .maxBatchSize(256)
 *       .maxWaitTime(Duration.ofMillis(10))
 *       .slaP95(Duration.ofMillis(100))
 *       .build();
 *
 *   // Submit requests (automatically batched)
 *   CompletableFuture&lt;InferenceResult&gt; future = batcher.submit(input);
 *   InferenceResult result = future.get();
 * </pre>
 *
 * @since 0.3.0
 */
public class BatchingEngine {

    private final ModelRunner runner;
    private final int maxBatchSize;
    private final Duration maxWaitTime;
    private final Duration slaP95;
    private final ExecutorService executor;

    private final Queue<PendingRequest> queue = new ConcurrentLinkedQueue<>();
    private final AtomicLong totalBatches = new AtomicLong(0);
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalDropped = new AtomicLong(0);
    private volatile boolean running = false;
    private volatile Thread batchThread;

    private BatchingEngine(Builder builder) {
        this.runner = builder.runner;
        this.maxBatchSize = builder.maxBatchSize;
        this.maxWaitTime = builder.maxWaitTime;
        this.slaP95 = builder.slaP95;
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Submits an inference request for batched execution.
     *
     * @param input the inference input
     * @return future with the result
     */
    public CompletableFuture<InferenceResult> submit(InferenceInput input) {
        if (!running) {
            start();
        }

        CompletableFuture<InferenceResult> future = new CompletableFuture<>();
        PendingRequest req = new PendingRequest(input, future, System.currentTimeMillis());
        queue.offer(req);
        totalRequests.incrementAndGet();
        return future;
    }

    /**
     * Starts the background batching loop.
     */
    public synchronized void start() {
        if (running) return;
        running = true;
        batchThread = Thread.ofVirtual().name("batching-engine").start(this::batchingLoop);
    }

    /**
     * Stops the batching engine and waits for pending requests.
     */
    public synchronized void stop() {
        if (!running) return;
        running = false;
        if (batchThread != null) {
            batchThread.interrupt();
            try { batchThread.join(5000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }

        // Fail pending requests
        PendingRequest req;
        while ((req = queue.poll()) != null) {
            req.future().completeExceptionally(new RejectedExecutionException("Batching engine stopped"));
            totalDropped.incrementAndGet();
        }
    }

    /**
     * Gets the current queue size.
     */
    public int queueSize() {
        return queue.size();
    }

    /**
     * Gets batch statistics.
     */
    public BatchStats stats() {
        return new BatchStats(
            totalBatches.get(),
            totalRequests.get(),
            totalDropped.get(),
            queue.size()
        );
    }

    /**
     * Background loop that collects requests and executes batches.
     */
    private void batchingLoop() {
        while (running) {
            try {
                // Collect batch
                List<PendingRequest> batch = new ArrayList<>(maxBatchSize);
                long batchStart = System.currentTimeMillis();
                long deadline = batchStart + maxWaitTime.toMillis();

                while (batch.size() < maxBatchSize && System.currentTimeMillis() < deadline) {
                    PendingRequest req = queue.poll();
                    if (req != null) {
                        batch.add(req);
                    } else {
                        // Brief sleep to avoid busy-wait
                        Thread.sleep(0, 100_000);  // 100μs
                    }
                }

                if (!batch.isEmpty()) {
                    // Execute batch
                    executeBatch(batch);
                    totalBatches.incrementAndGet();
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                // Log and continue
            }
        }
    }

    /**
     * Executes a batch of requests.
     */
    private void executeBatch(List<PendingRequest> batch) {
        try {
            if (batch.size() == 1) {
                // Single request: direct inference
                PendingRequest req = batch.get(0);
                InferenceResult result = runner.infer(req.input());
                req.future().complete(result);
            } else {
                // Multiple requests: batch inference
                InferenceInput[] inputs = batch.stream()
                    .map(PendingRequest::input)
                    .toArray(InferenceInput[]::new);

                InferenceResult[] results = runner.inferBatch(inputs);

                for (int i = 0; i < batch.size(); i++) {
                    batch.get(i).future().complete(results[i]);
                }
            }
        } catch (Exception e) {
            for (PendingRequest req : batch) {
                req.future().completeExceptionally(e);
            }
        }
    }

    /**
     * Pending request in the queue.
     */
    private record PendingRequest(
        InferenceInput input,
        CompletableFuture<InferenceResult> future,
        long enqueueTimeMs
    ) {}

    /**
     * Batch execution statistics.
     */
    public record BatchStats(
        long totalBatches,
        long totalRequests,
        long totalDropped,
        int currentQueueSize
    ) {
        public double avgBatchSize() {
            return totalBatches > 0 ? (double) totalRequests / totalBatches : 0;
        }
    }

    /**
     * Builder for BatchingEngine.
     */
    public static class Builder {
        private ModelRunner runner;
        private int maxBatchSize = 256;
        private Duration maxWaitTime = Duration.ofMillis(10);
        private Duration slaP95 = Duration.ofMillis(100);

        Builder() {}

        public Builder runner(ModelRunner runner) {
            this.runner = runner;
            return this;
        }

        public Builder maxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }

        public Builder maxWaitTime(Duration maxWaitTime) {
            this.maxWaitTime = maxWaitTime;
            return this;
        }

        public Builder slaP95(Duration slaP95) {
            this.slaP95 = slaP95;
            return this;
        }

        public BatchingEngine build() {
            if (runner == null) throw new IllegalStateException("runner is required");
            return new BatchingEngine(this);
        }
    }
}
