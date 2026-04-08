package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.UnaryOperator;

/**
 * Multi-threaded DataLoader with prefetching — uses JDK 25 virtual threads.
 *
 * <p>Equivalent to {@code torch.utils.data.DataLoader}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loader = DataLoader.builder(dataset)
 *     .batchSize(32)
 *     .shuffle(true)
 *     .numWorkers(4)
 *     .transform(Transforms.compose(
 *         Transforms.randomHorizontalFlip(0.5f),
 *         Transforms.normalize(mean, std)
 *     ))
 *     .build();
 *
 * for (Batch batch : loader) {
 *     GradTensor x = batch.inputs();
 *     GradTensor y = batch.labels();
 * }
 * }</pre>
 */
public class DataLoader implements Iterable<DataLoader.Batch> {

    private final Dataset dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final int numWorkers;
    private final int prefetchFactor;
    private final UnaryOperator<GradTensor> transform;

    private DataLoader(Builder b) {
        this.dataset       = b.dataset;
        this.batchSize     = b.batchSize;
        this.shuffle       = b.shuffle;
        this.numWorkers    = b.numWorkers;
        this.prefetchFactor = b.prefetchFactor;
        this.transform     = b.transform;
    }

    public static Builder builder(Dataset dataset) { return new Builder(dataset); }

    public int numBatches() { return (int) Math.ceil((double) dataset.size() / batchSize); }

    @Override
    public Iterator<Batch> iterator() {
        int[] indices = buildIndices();
        // Prefetch queue filled by virtual-thread workers
        BlockingQueue<Batch> queue = new LinkedBlockingQueue<>(prefetchFactor * numWorkers);

        // Producer: virtual threads load batches into queue
        Thread.ofVirtual().start(() -> {
            try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
                for (int b = 0; b < numBatches(); b++) {
                    final int batchIdx = b;
                    exec.submit(() -> {
                        Batch batch = loadBatch(indices, batchIdx);
                        try { queue.put(batch); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                    });
                }
                exec.shutdown();
                exec.awaitTermination(1, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            // Poison pill
            try { queue.put(Batch.END); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        });

        return new Iterator<>() {
            Batch next = null;
            @Override public boolean hasNext() {
                if (next == null) next = poll();
                return next != Batch.END;
            }
            @Override public Batch next() {
                Batch b = next; next = null; return b;
            }
            private Batch poll() {
                try { return queue.take(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); return Batch.END; }
            }
        };
    }

    private Batch loadBatch(int[] indices, int batchIdx) {
        int start = batchIdx * batchSize;
        int end   = Math.min(start + batchSize, dataset.size());
        int actual = end - start;

        List<GradTensor> inputs = new ArrayList<>(actual);
        List<GradTensor> labels = new ArrayList<>(actual);

        for (int i = start; i < end; i++) {
            Dataset.Sample s = dataset.get(indices[i]);
            GradTensor x = transform != null ? transform.apply(s.input()) : s.input();
            inputs.add(x);
            labels.add(s.label());
        }
        return new Batch(collate(inputs), collate(labels));
    }

    /** Stack list of tensors along new batch dim 0. */
    private static GradTensor collate(List<GradTensor> tensors) {
        if (tensors.isEmpty()) return GradTensor.zeros(0);
        long[] shape = tensors.get(0).shape();
        int N = tensors.size();
        int elemSize = (int) GradTensor.numelFor(shape);
        float[] out = new float[N * elemSize];
        for (int i = 0; i < N; i++)
            System.arraycopy(tensors.get(i).data(), 0, out, i * elemSize, elemSize);
        long[] batchShape = new long[shape.length + 1];
        batchShape[0] = N;
        System.arraycopy(shape, 0, batchShape, 1, shape.length);
        return GradTensor.of(out, batchShape);
    }

    private int[] buildIndices() {
        int[] idx = new int[dataset.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        if (shuffle) {
            Random rng = new Random();
            for (int i = idx.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1); int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
        return idx;
    }

    // ── Batch ─────────────────────────────────────────────────────────────

    public record Batch(GradTensor inputs, GradTensor labels) {
        static final Batch END = new Batch(null, null);
    }

    // ── Builder ───────────────────────────────────────────────────────────

    public static class Builder {
        private final Dataset dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private int numWorkers = 2;
        private int prefetchFactor = 2;
        private UnaryOperator<GradTensor> transform;

        Builder(Dataset dataset) { this.dataset = dataset; }
        public Builder batchSize(int n)          { this.batchSize = n; return this; }
        public Builder shuffle(boolean s)         { this.shuffle = s; return this; }
        public Builder numWorkers(int n)          { this.numWorkers = n; return this; }
        public Builder prefetchFactor(int n)      { this.prefetchFactor = n; return this; }
        public Builder transform(UnaryOperator<GradTensor> t) { this.transform = t; return this; }
        public DataLoader build()                 { return new DataLoader(this); }
    }
}
