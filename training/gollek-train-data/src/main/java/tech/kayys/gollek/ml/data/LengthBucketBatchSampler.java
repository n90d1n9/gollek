package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Orders variable-length samples so each DataLoader batch contains similar sequence lengths.
 */
public final class LengthBucketBatchSampler implements IndexSampler {
    public static final int DEFAULT_BUCKET_SIZE_MULTIPLIER = 50;

    private final int[] lengths;
    private final int batchSize;
    private final int bucketSizeMultiplier;
    private final boolean shuffleBatches;
    private final boolean shuffleWithinBuckets;
    private final boolean dropLast;
    private final long seed;

    public LengthBucketBatchSampler(int[] lengths, int batchSize, long seed) {
        this(lengths, batchSize, DEFAULT_BUCKET_SIZE_MULTIPLIER, true, false, false, seed);
    }

    public LengthBucketBatchSampler(int[] lengths, int batchSize, boolean dropLast, long seed) {
        this(lengths, batchSize, DEFAULT_BUCKET_SIZE_MULTIPLIER, true, false, dropLast, seed);
    }

    public LengthBucketBatchSampler(
            int[] lengths,
            int batchSize,
            int bucketSizeMultiplier,
            boolean shuffleBatches,
            boolean shuffleWithinBuckets,
            boolean dropLast,
            long seed) {
        this.lengths = validateLengths(lengths);
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        if (bucketSizeMultiplier <= 0) {
            throw new IllegalArgumentException(
                    "bucketSizeMultiplier must be positive, got: " + bucketSizeMultiplier);
        }
        this.batchSize = batchSize;
        this.bucketSizeMultiplier = bucketSizeMultiplier;
        this.shuffleBatches = shuffleBatches;
        this.shuffleWithinBuckets = shuffleWithinBuckets;
        this.dropLast = dropLast;
        this.seed = seed;
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireDatasetSize(datasetSize);
        int count = sampleCount(datasetSize);
        if (count == 0) {
            return List.of();
        }

        List<Integer> sorted = sortedByLength();
        int bucketSize = bucketSize();
        Random random = new Random(seed);
        List<List<Integer>> fullBatches = new ArrayList<>(count / batchSize);
        List<Integer> leftovers = new ArrayList<>();

        for (int start = 0; start < sorted.size(); start += bucketSize) {
            int end = Math.min(start + bucketSize, sorted.size());
            List<Integer> bucket = new ArrayList<>(sorted.subList(start, end));
            if (shuffleWithinBuckets) {
                Collections.shuffle(bucket, random);
            }
            appendFullBatches(bucket, fullBatches, leftovers);
        }

        List<Integer> partial = List.of();
        if (!dropLast && !leftovers.isEmpty()) {
            leftovers.sort(this::compareByLength);
            int fullLimit = (leftovers.size() / batchSize) * batchSize;
            for (int start = 0; start < fullLimit; start += batchSize) {
                fullBatches.add(new ArrayList<>(leftovers.subList(start, start + batchSize)));
            }
            if (fullLimit < leftovers.size()) {
                partial = new ArrayList<>(leftovers.subList(fullLimit, leftovers.size()));
            }
        }

        if (shuffleBatches) {
            Collections.shuffle(fullBatches, random);
        }

        List<Integer> result = new ArrayList<>(count);
        for (List<Integer> batch : fullBatches) {
            result.addAll(batch);
        }
        result.addAll(partial);
        return result;
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        return dropLast ? (lengths.length / batchSize) * batchSize : lengths.length;
    }

    public int[] lengths() {
        return Arrays.copyOf(lengths, lengths.length);
    }

    public int batchSize() {
        return batchSize;
    }

    public int bucketSizeMultiplier() {
        return bucketSizeMultiplier;
    }

    public boolean shuffleBatches() {
        return shuffleBatches;
    }

    public boolean shuffleWithinBuckets() {
        return shuffleWithinBuckets;
    }

    public boolean dropLast() {
        return dropLast;
    }

    public long seed() {
        return seed;
    }

    private void appendFullBatches(
            List<Integer> bucket,
            List<List<Integer>> fullBatches,
            List<Integer> leftovers) {
        int fullLimit = (bucket.size() / batchSize) * batchSize;
        for (int start = 0; start < fullLimit; start += batchSize) {
            List<Integer> batch = new ArrayList<>(bucket.subList(start, start + batchSize));
            if (!shuffleWithinBuckets) {
                batch.sort(this::compareByLength);
            }
            fullBatches.add(batch);
        }
        if (!dropLast && fullLimit < bucket.size()) {
            leftovers.addAll(bucket.subList(fullLimit, bucket.size()));
        }
    }

    private List<Integer> sortedByLength() {
        List<Integer> indices = new ArrayList<>(lengths.length);
        for (int i = 0; i < lengths.length; i++) {
            indices.add(i);
        }
        indices.sort(this::compareByLength);
        return indices;
    }

    private int compareByLength(int left, int right) {
        int comparison = Integer.compare(lengths[left], lengths[right]);
        return comparison != 0 ? comparison : Integer.compare(left, right);
    }

    private int bucketSize() {
        long bucketSize = (long) batchSize * bucketSizeMultiplier;
        if (bucketSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("bucket size is too large");
        }
        return (int) Math.max(batchSize, bucketSize);
    }

    private void requireDatasetSize(int datasetSize) {
        if (datasetSize != lengths.length) {
            throw new IllegalArgumentException(
                    "lengths must match dataset size, got " + lengths.length + " vs " + datasetSize);
        }
    }

    private static int[] validateLengths(int[] lengths) {
        Objects.requireNonNull(lengths, "lengths must not be null");
        int[] copy = Arrays.copyOf(lengths, lengths.length);
        for (int length : copy) {
            if (length < 0) {
                throw new IllegalArgumentException("lengths must be non-negative, got: " + length);
            }
        }
        return copy;
    }
}
