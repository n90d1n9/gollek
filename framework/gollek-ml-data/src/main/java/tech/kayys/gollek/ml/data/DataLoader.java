package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * DataLoader — batches, shuffles, and yields data in mini-batches.
 *
 * <p>Supports:
 * <ul>
 *   <li>Batching with configurable batch size</li>
 *   <li>Shuffling (on/off)</li>
 *   <li>Drop last incomplete batch</li>
 *   <li>Custom collate function</li>
 *   <li>Element mapping via {@link #map(Function)}</li>
 * </ul>
 *
 * <h3>Generic usage</h3>
 * <pre>{@code
 * Dataset<Integer> dataset = ...;
 * var loader = new DataLoader<>(dataset, 32);
 * for (List<Integer> batch : loader) {
 *     // process batch
 * }
 * }</pre>
 *
 * <h3>Tensor (training) usage</h3>
 * <pre>{@code
 * var tensorDataset = new TensorDataset(inputs, targets);
 * var loader = DataLoader.tensorBuilder(tensorDataset)
 *     .batchSize(32)
 *     .shuffle(true)
 *     .dropLast(true)
 *     .build();
 * for (List<GradTensor> batch : loader) {
 *     var input = batch.get(0);
 *     var target = batch.get(1);
 * }
 * }</pre>
 *
 * @param <T> the type of elements produced by this data loader
 * @author Gollek Team
 * @version 0.2.0
 */
public class DataLoader<T> implements Iterable<List<T>> {

    private final Dataset<? extends T> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;

    // ── Constructors ─────────────────────────────────────────────────────

    /**
     * Create a DataLoader with the given batch size (no shuffle, no drop-last).
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize) {
        this(dataset, batchSize, false, false);
    }

    /**
     * Create a DataLoader with full configuration.
     *
     * @param dataset   the dataset to load from
     * @param batchSize number of samples per batch
     * @param shuffle   whether to shuffle indices each epoch
     * @param dropLast  whether to drop the last incomplete batch
     */
    public DataLoader(Dataset<? extends T> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
    }

    // ── Iteration ────────────────────────────────────────────────────────

    @Override
    public Iterator<List<T>> iterator() {
        int n = dataset.size();
        // Create a mutable copy — toList() returns an unmodifiable list
        List<Integer> indices = new ArrayList<>(IntStream.range(0, n).boxed().toList());

        if (shuffle) {
            Collections.shuffle(indices, ThreadLocalRandom.current());
        }

        int numBatches = numBatches();

        return new Iterator<>() {
            private int current = 0;

            @Override
            public boolean hasNext() {
                return current < numBatches;
            }

            @Override
            public List<T> next() {
                if (current >= numBatches) throw new NoSuchElementException();

                int start = current * batchSize;
                int end = Math.min(start + batchSize, n);

                List<T> batch = new ArrayList<>(end - start);
                for (int i = start; i < end; i++) {
                    batch.add(dataset.get(indices.get(i)));
                }

                current++;
                return batch;
            }
        };
    }

    /** Get number of batches per epoch. */
    public int numBatches() {
        int n = dataset.size();
        return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
    }

    // ── Mapping ──────────────────────────────────────────────────────────

    /**
     * Create a new DataLoader that applies a mapping function to each element.
     *
     * <pre>{@code
     * DataLoader<String> mapped = new DataLoader<>(intDataset, 5)
     *     .map(i -> "val_" + i);
     * }</pre>
     *
     * @param mapper function to apply to each element
     * @param <R>    the output type
     * @return a new DataLoader producing mapped elements
     */
    public <R> DataLoader<R> map(Function<? super T, ? extends R> mapper) {
        Dataset<? extends T> srcDataset = this.dataset;
        Dataset<R> mappedDataset = new Dataset<>() {
            @Override
            public R get(int index) {
                return mapper.apply(srcDataset.get(index));
            }

            @Override
            public int size() {
                return srcDataset.size();
            }
        };
        return new DataLoader<>(mappedDataset, this.batchSize, this.shuffle, this.dropLast);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TensorDataLoader — GradTensor-specific builder with collation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a builder for a tensor-specific DataLoader that supports
     * custom collation and stacking of {@link GradTensor} batches.
     *
     * @param dataset the tensor dataset
     * @return a new builder
     */
    public static TensorBuilder tensorBuilder(TensorDatasetAdapter dataset) {
        return new TensorBuilder(dataset);
    }

    /**
     * Adapter interface for tensor datasets that return arrays of GradTensors.
     * This is the specialized interface for training workflows where each
     * sample is a tuple of tensors (e.g., [input, target]).
     */
    public interface TensorDatasetAdapter {
        /** Total number of samples. */
        int size();

        /** Get number of samples for given indices. */
        default int size(List<Integer> indices) {
            return indices.size();
        }

        /** Get sample at given index. Returns array of tensors (e.g., [input, target]). */
        GradTensor[] get(int index);
    }

    /**
     * Simple dataset from arrays of GradTensor.
     */
    public static class TensorDataset implements TensorDatasetAdapter {
        private final List<GradTensor[]> samples;

        public TensorDataset(GradTensor[]... samples) {
            this.samples = Arrays.asList(samples);
        }

        /**
         * Create from separate input and target tensors.
         * Each input[i] and target[i] form a sample pair.
         */
        public TensorDataset(GradTensor inputs, GradTensor targets) {
            if (inputs.shape()[0] != targets.shape()[0]) {
                throw new IllegalArgumentException("inputs and targets must have same batch dimension");
            }
            int n = (int) inputs.shape()[0];
            long[] inputShape = inputs.shape();
            long[] targetShape = targets.shape();
            samples = new ArrayList<>(n);

            for (int i = 0; i < n; i++) {
                // Extract individual samples
                int inputStride = 1;
                for (int d = 1; d < inputShape.length; d++) inputStride *= inputShape[d];
                int targetStride = 1;
                for (int d = 1; d < targetShape.length; d++) targetStride *= targetShape[d];

                float[] inputData = inputs.data();
                float[] targetData = targets.data();
                int inputOffset = i * inputStride;
                int targetOffset = i * targetStride;

                long[] inputSampleShape = Arrays.copyOfRange(inputShape, 1, inputShape.length);
                long[] targetSampleShape = Arrays.copyOfRange(targetShape, 1, targetShape.length);

                float[] inputSample = Arrays.copyOfRange(inputData, inputOffset, inputOffset + inputStride);
                float[] targetSample = Arrays.copyOfRange(targetData, targetOffset, targetOffset + targetStride);

                samples.add(new GradTensor[]{
                    GradTensor.of(inputSample, inputSampleShape),
                    GradTensor.of(targetSample, targetSampleShape)
                });
            }
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public GradTensor[] get(int index) {
            return samples.get(index);
        }
    }

    /**
     * Builder for tensor-specific DataLoader with collation support.
     */
    public static class TensorBuilder {
        private final TensorDatasetAdapter dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private boolean dropLast = false;
        private CollateFn collateFn;

        TensorBuilder(TensorDatasetAdapter dataset) {
            this.dataset = dataset;
        }

        public TensorBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public TensorBuilder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }

        public TensorBuilder dropLast(boolean dropLast) {
            this.dropLast = dropLast;
            return this;
        }

        public TensorBuilder collateFn(CollateFn collateFn) {
            this.collateFn = collateFn;
            return this;
        }

        public TensorDataLoader build() {
            return new TensorDataLoader(this);
        }
    }

    /**
     * Tensor-specific DataLoader that handles collation and stacking
     * of {@link GradTensor} batches.
     */
    public static class TensorDataLoader implements Iterable<Batch> {
        private final TensorDatasetAdapter dataset;
        private final int batchSize;
        private final boolean shuffle;
        private final boolean dropLast;
        private final CollateFn collateFn;

        private TensorDataLoader(TensorBuilder builder) {
            this.dataset = Objects.requireNonNull(builder.dataset, "dataset must not be null");
            this.batchSize = builder.batchSize;
            this.shuffle = builder.shuffle;
            this.dropLast = builder.dropLast;
            this.collateFn = builder.collateFn != null ? builder.collateFn : defaultCollate();
        }

        @Override
        public Iterator<Batch> iterator() {
            int n = dataset.size();
            List<Integer> indices = new ArrayList<>(IntStream.range(0, n).boxed().toList());

            if (shuffle) {
                Collections.shuffle(indices, ThreadLocalRandom.current());
            }

            int numBatches = numBatches();

            return new Iterator<>() {
                private int current = 0;

                @Override
                public boolean hasNext() {
                    return current < numBatches;
                }

                @Override
                public Batch next() {
                    if (current >= numBatches) throw new NoSuchElementException();

                    int start = current * batchSize;
                    int end = Math.min(start + batchSize, n);
                    List<Integer> batchIndices = indices.subList(start, end);

                    current++;
                    return collateFn.collate(batchIndices, dataset);
                }
            };
        }

        /** Get number of batches per epoch. */
        public int numBatches() {
            int n = dataset.size();
            return dropLast ? n / batchSize : (int) Math.ceil((double) n / batchSize);
        }

        /** Default collate function: stacks tensors into batched tensors. */
        private static CollateFn defaultCollate() {
            return (indices, ds) -> {
                int numSamples = ds.size(indices);
                int numTensors = ds.get(0).length;
                if (numTensors < 2) {
                    throw new IllegalStateException("TensorDataset must have at least 2 tensors (input and label)");
                }

                GradTensor[] inputSamples = new GradTensor[numSamples];
                GradTensor[] labelSamples = new GradTensor[numSamples];

                for (int i = 0; i < numSamples; i++) {
                    GradTensor[] sample = ds.get(indices.get(i));
                    inputSamples[i] = sample[0];
                    labelSamples[i] = sample[1];
                }

                return new Batch(
                    GradTensor.stack(0, inputSamples),
                    GradTensor.stack(0, labelSamples)
                );
            };
        }
    }

    /**
     * Collate function interface for tensor datasets.
     */
    @FunctionalInterface
    public interface CollateFn {
        Batch collate(List<Integer> indices, TensorDatasetAdapter dataset);
    }

    /**
     * A record representing a mini-batch of data.
     *
     * @param inputs input tensor (usually [N, ...])
     * @param labels target/label tensor (usually [N, ...])
     */
    public record Batch(GradTensor inputs, GradTensor labels) {
        /** @return inputs tensor */
        @Override public GradTensor inputs() { return inputs; }
        /** @return labels tensor */
        @Override public GradTensor labels() { return labels; }
    }
}
