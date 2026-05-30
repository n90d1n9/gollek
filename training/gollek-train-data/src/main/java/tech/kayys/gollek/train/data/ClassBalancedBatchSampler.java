package tech.kayys.gollek.train.data;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Random;
import java.util.TreeMap;

/**
 * Produces an index stream arranged as class-balanced mini-batches.
 *
 * <p>The sampler returns a flat index list because {@link DataLoader.TensorDataLoader}
 * chunks sampler output by batch size. Each contiguous chunk produced here is
 * balanced across the classes present in the label vector. Replacement mode is
 * useful for imbalanced datasets because minority classes can be oversampled
 * without changing the underlying dataset.</p>
 */
public final class ClassBalancedBatchSampler implements IndexSampler {

    private final int[] labels;
    private final int batchSize;
    private final int batchesPerEpoch;
    private final boolean replacement;
    private final long seed;
    private final List<Integer> classes;

    public ClassBalancedBatchSampler(int[] labels, int batchSize, int batchesPerEpoch, long seed) {
        this(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public ClassBalancedBatchSampler(
            int[] labels,
            int batchSize,
            int batchesPerEpoch,
            boolean replacement,
            long seed) {
        this.labels = validateLabels(labels);
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        if (batchesPerEpoch <= 0) {
            throw new IllegalArgumentException("batchesPerEpoch must be positive, got: " + batchesPerEpoch);
        }
        int requestedSamples = Math.multiplyExact(batchSize, batchesPerEpoch);
        if (!replacement && requestedSamples > labels.length) {
            throw new IllegalArgumentException(
                    "requested samples cannot exceed label count without replacement, got: "
                            + requestedSamples + " > " + labels.length);
        }
        this.batchSize = batchSize;
        this.batchesPerEpoch = batchesPerEpoch;
        this.replacement = replacement;
        this.seed = seed;
        this.classes = classOrder(this.labels);
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireDatasetSize(datasetSize);
        Map<Integer, List<Integer>> byClass = groupedIndices();
        if (replacement) {
            return sampleWithReplacement(byClass);
        }
        return sampleWithoutReplacement(byClass);
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        return Math.multiplyExact(batchSize, batchesPerEpoch);
    }

    public int[] labels() {
        return Arrays.copyOf(labels, labels.length);
    }

    public int batchSize() {
        return batchSize;
    }

    public int batchesPerEpoch() {
        return batchesPerEpoch;
    }

    public boolean replacement() {
        return replacement;
    }

    public long seed() {
        return seed;
    }

    public List<Integer> classes() {
        return List.copyOf(classes);
    }

    private List<Integer> sampleWithReplacement(Map<Integer, List<Integer>> byClass) {
        Map<Integer, Integer> cursors = new TreeMap<>();
        List<Integer> selected = new ArrayList<>(sampleCount(labels.length));
        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            List<Integer> batchClasses = batchClasses(batch);
            List<Integer> quotas = quotas(batchClasses.size());
            List<Integer> batchIndices = new ArrayList<>(batchSize);
            for (int i = 0; i < batchClasses.size(); i++) {
                int label = batchClasses.get(i);
                List<Integer> indices = byClass.get(label);
                int cursor = cursors.getOrDefault(label, 0);
                for (int draw = 0; draw < quotas.get(i); draw++) {
                    if (cursor >= indices.size()) {
                        Collections.shuffle(indices, new Random(mix(seed, label, batch, draw)));
                        cursor = 0;
                    }
                    batchIndices.add(indices.get(cursor));
                    cursor++;
                }
                cursors.put(label, cursor);
            }
            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0xCBF29CE484222325L)));
            selected.addAll(batchIndices);
        }
        return selected;
    }

    private List<Integer> sampleWithoutReplacement(Map<Integer, List<Integer>> byClass) {
        Map<Integer, Queue<Integer>> queues = new TreeMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : byClass.entrySet()) {
            queues.put(entry.getKey(), new ArrayDeque<>(entry.getValue()));
        }

        List<Integer> selected = new ArrayList<>(sampleCount(labels.length));
        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            List<Integer> availableClasses = new ArrayList<>();
            for (int label : classes) {
                Queue<Integer> queue = queues.get(label);
                if (queue != null && !queue.isEmpty()) {
                    availableClasses.add(label);
                }
            }
            if (availableClasses.isEmpty()) {
                break;
            }
            Collections.shuffle(availableClasses, new Random(mix(seed, batch, 0x9E3779B97F4A7C15L, 0)));

            List<Integer> batchIndices = new ArrayList<>(batchSize);
            int cursor = 0;
            while (batchIndices.size() < batchSize && !availableClasses.isEmpty()) {
                int label = availableClasses.get(cursor % availableClasses.size());
                Queue<Integer> queue = queues.get(label);
                batchIndices.add(queue.remove());
                if (queue.isEmpty()) {
                    availableClasses.remove(Integer.valueOf(label));
                    if (availableClasses.isEmpty()) {
                        break;
                    }
                    cursor = cursor % availableClasses.size();
                } else {
                    cursor++;
                }
            }
            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0x100000001B3L)));
            selected.addAll(batchIndices);
        }
        if (selected.size() != sampleCount(labels.length)) {
            throw new IllegalStateException(
                    "could not produce requested samples without replacement, got: "
                            + selected.size() + " vs " + sampleCount(labels.length));
        }
        return selected;
    }

    private List<Integer> batchClasses(int batch) {
        List<Integer> batchClasses = new ArrayList<>(classes);
        Collections.shuffle(batchClasses, new Random(mix(seed, batch, classes.size(), 0xA0761D6478BD642FL)));
        if (batchSize < batchClasses.size()) {
            return new ArrayList<>(batchClasses.subList(0, batchSize));
        }
        return batchClasses;
    }

    private List<Integer> quotas(int classCount) {
        int base = batchSize / classCount;
        int remainder = batchSize % classCount;
        List<Integer> quotas = new ArrayList<>(classCount);
        for (int i = 0; i < classCount; i++) {
            quotas.add(base + (i < remainder ? 1 : 0));
        }
        return quotas;
    }

    private Map<Integer, List<Integer>> groupedIndices() {
        Map<Integer, List<Integer>> byClass = new TreeMap<>();
        for (int i = 0; i < labels.length; i++) {
            byClass.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        }
        for (Map.Entry<Integer, List<Integer>> entry : byClass.entrySet()) {
            Collections.shuffle(entry.getValue(), new Random(mix(seed, entry.getKey(), labels.length, 0)));
        }
        return byClass;
    }

    private void requireDatasetSize(int datasetSize) {
        if (datasetSize != labels.length) {
            throw new IllegalArgumentException(
                    "labels length must match dataset size, got: " + labels.length + " vs " + datasetSize);
        }
    }

    private static int[] validateLabels(int[] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int[] copy = Arrays.copyOf(labels, labels.length);
        for (int label : copy) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
        }
        return copy;
    }

    private static List<Integer> classOrder(int[] labels) {
        List<Integer> order = new ArrayList<>();
        for (int label : labels) {
            if (!order.contains(label)) {
                order.add(label);
            }
        }
        Collections.sort(order);
        return order;
    }

    private static long mix(long seed, long a, long b, long c) {
        long value = seed ^ 0x9E3779B97F4A7C15L;
        value = (value ^ a) * 0xBF58476D1CE4E5B9L;
        value = (value ^ b) * 0x94D049BB133111EBL;
        return value ^ c ^ (value >>> 31);
    }
}
