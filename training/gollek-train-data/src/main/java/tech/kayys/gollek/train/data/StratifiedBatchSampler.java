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
 * Produces mini-batches that preserve the original class distribution.
 *
 * <p>This sampler is intentionally different from {@link ClassBalancedBatchSampler}:
 * balanced batches push every class toward equal representation, while stratified
 * batches keep the dataset's natural proportions as closely as possible.</p>
 */
public final class StratifiedBatchSampler implements IndexSampler {
    private final int[] labels;
    private final int batchSize;
    private final int batchesPerEpoch;
    private final boolean replacement;
    private final long seed;
    private final List<Integer> classes;
    private final int[] classCounts;

    public StratifiedBatchSampler(int[] labels, int batchSize, int batchesPerEpoch, long seed) {
        this(labels, batchSize, batchesPerEpoch, true, seed);
    }

    public StratifiedBatchSampler(
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
        this.classCounts = classCounts(this.labels, this.classes);
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

    public int[] classCounts() {
        return Arrays.copyOf(classCounts, classCounts.length);
    }

    private List<Integer> sampleWithReplacement(Map<Integer, List<Integer>> byClass) {
        int[] assigned = new int[classes.size()];
        int[] cursors = new int[classes.size()];
        List<Integer> selected = new ArrayList<>(sampleCount(labels.length));
        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            int[] quotas = proportionalQuotas(batch, assigned, false);
            List<Integer> batchIndices = new ArrayList<>(batchSize);
            for (int position = 0; position < classes.size(); position++) {
                int label = classes.get(position);
                List<Integer> indices = byClass.get(label);
                for (int draw = 0; draw < quotas[position]; draw++) {
                    if (cursors[position] >= indices.size()) {
                        Collections.shuffle(indices, new Random(mix(seed, label, batch, draw)));
                        cursors[position] = 0;
                    }
                    batchIndices.add(indices.get(cursors[position]));
                    cursors[position]++;
                    assigned[position]++;
                }
            }
            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0xC2B2AE3D27D4EB4FL)));
            selected.addAll(batchIndices);
        }
        return selected;
    }

    private List<Integer> sampleWithoutReplacement(Map<Integer, List<Integer>> byClass) {
        Map<Integer, Queue<Integer>> queues = new TreeMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : byClass.entrySet()) {
            queues.put(entry.getKey(), new ArrayDeque<>(entry.getValue()));
        }

        int[] assigned = new int[classes.size()];
        List<Integer> selected = new ArrayList<>(sampleCount(labels.length));
        for (int batch = 0; batch < batchesPerEpoch; batch++) {
            int[] quotas = proportionalQuotas(batch, assigned, true);
            List<Integer> batchIndices = new ArrayList<>(batchSize);
            for (int position = 0; position < classes.size(); position++) {
                Queue<Integer> queue = queues.get(classes.get(position));
                for (int draw = 0; draw < quotas[position]; draw++) {
                    if (queue == null || queue.isEmpty()) {
                        break;
                    }
                    batchIndices.add(queue.remove());
                    assigned[position]++;
                }
            }
            fillRemainingWithoutReplacement(batch, assigned, queues, batchIndices);
            Collections.shuffle(batchIndices, new Random(mix(seed, batch, batchSize, 0x9E3779B97F4A7C15L)));
            selected.addAll(batchIndices);
        }
        if (selected.size() != sampleCount(labels.length)) {
            throw new IllegalStateException(
                    "could not produce requested samples without replacement, got: "
                            + selected.size() + " vs " + sampleCount(labels.length));
        }
        return selected;
    }

    private void fillRemainingWithoutReplacement(
            int batch,
            int[] assigned,
            Map<Integer, Queue<Integer>> queues,
            List<Integer> batchIndices) {
        while (batchIndices.size() < batchSize) {
            int position = nextQuotaPosition((batch + 1) * batchSize, assigned, new int[classes.size()], true);
            if (position < 0) {
                throw new IllegalStateException("no class has remaining samples for stratified batch");
            }
            Queue<Integer> queue = queues.get(classes.get(position));
            if (queue == null || queue.isEmpty()) {
                assigned[position] = classCounts[position];
                continue;
            }
            batchIndices.add(queue.remove());
            assigned[position]++;
        }
    }

    private int[] proportionalQuotas(int batch, int[] assigned, boolean capAtClassCounts) {
        int targetTotal = Math.multiplyExact(batchSize, batch + 1);
        int[] quotas = new int[classes.size()];
        int sum = 0;
        for (int position = 0; position < classes.size(); position++) {
            int maxAdditional = maxAdditional(position, assigned, capAtClassCounts);
            int quota = (int) Math.floor(expectedCumulative(position, targetTotal)) - assigned[position];
            quota = Math.max(0, Math.min(quota, maxAdditional));
            quotas[position] = quota;
            sum += quota;
        }
        while (sum < batchSize) {
            int position = nextQuotaPosition(targetTotal, assigned, quotas, capAtClassCounts);
            if (position < 0) {
                break;
            }
            quotas[position]++;
            sum++;
        }
        if (sum != batchSize) {
            throw new IllegalStateException(
                    "could not allocate a full stratified batch, got: " + sum + " vs " + batchSize);
        }
        return quotas;
    }

    private int nextQuotaPosition(int targetTotal, int[] assigned, int[] quotas, boolean capAtClassCounts) {
        int bestPosition = -1;
        double bestDeficit = -Double.MAX_VALUE;
        long bestTie = Long.MAX_VALUE;
        for (int position = 0; position < classes.size(); position++) {
            if (quotas[position] >= maxAdditional(position, assigned, capAtClassCounts)) {
                continue;
            }
            double deficit = expectedCumulative(position, targetTotal) - assigned[position] - quotas[position];
            long tie = mix(seed, targetTotal, classes.get(position), position);
            if (deficit > bestDeficit + 1e-12 || (Math.abs(deficit - bestDeficit) <= 1e-12 && tie < bestTie)) {
                bestDeficit = deficit;
                bestTie = tie;
                bestPosition = position;
            }
        }
        return bestPosition;
    }

    private int maxAdditional(int position, int[] assigned, boolean capAtClassCounts) {
        if (!capAtClassCounts) {
            return batchSize;
        }
        return Math.max(0, classCounts[position] - assigned[position]);
    }

    private double expectedCumulative(int position, int targetTotal) {
        return targetTotal * (double) classCounts[position] / labels.length;
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
        Map<Integer, Boolean> present = new TreeMap<>();
        for (int label : labels) {
            present.put(label, Boolean.TRUE);
        }
        return List.copyOf(present.keySet());
    }

    private static int[] classCounts(int[] labels, List<Integer> classes) {
        Map<Integer, Integer> positions = new TreeMap<>();
        for (int i = 0; i < classes.size(); i++) {
            positions.put(classes.get(i), i);
        }
        int[] counts = new int[classes.size()];
        for (int label : labels) {
            counts[positions.get(label)]++;
        }
        return counts;
    }

    private static long mix(long seed, long a, long b, long c) {
        long x = seed ^ 0x9E3779B97F4A7C15L;
        x ^= a + 0xBF58476D1CE4E5B9L + (x << 6) + (x >>> 2);
        x ^= b + 0x94D049BB133111EBL + (x << 6) + (x >>> 2);
        x ^= c + 0xD1B54A32D192ED03L + (x << 6) + (x >>> 2);
        return x;
    }
}
