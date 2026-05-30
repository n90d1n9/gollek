package tech.kayys.gollek.ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Partitions dataset indices across multiple replicas for distributed training.
 */
public final class DistributedSampler implements IndexSampler {
    private final int numReplicas;
    private final int rank;
    private final boolean shuffle;
    private final boolean dropLast;
    private final long seed;
    private final long epoch;

    public DistributedSampler(int numReplicas, int rank) {
        this(numReplicas, rank, true, false, 0L, 0L);
    }

    public DistributedSampler(int numReplicas, int rank, boolean shuffle, boolean dropLast, long seed) {
        this(numReplicas, rank, shuffle, dropLast, seed, 0L);
    }

    public DistributedSampler(
            int numReplicas,
            int rank,
            boolean shuffle,
            boolean dropLast,
            long seed,
            long epoch) {
        if (numReplicas <= 0) {
            throw new IllegalArgumentException("numReplicas must be positive, got: " + numReplicas);
        }
        if (rank < 0 || rank >= numReplicas) {
            throw new IllegalArgumentException("rank must be in [0, " + (numReplicas - 1) + "], got: " + rank);
        }
        this.numReplicas = numReplicas;
        this.rank = rank;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.seed = seed;
        this.epoch = epoch;
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireDatasetSize(datasetSize);
        int perReplica = sampleCount(datasetSize);
        int totalSize = perReplica * numReplicas;
        if (totalSize == 0) {
            return List.of();
        }

        List<Integer> indices = new ArrayList<>(datasetSize);
        for (int i = 0; i < datasetSize; i++) {
            indices.add(i);
        }
        if (shuffle) {
            Collections.shuffle(indices, new Random(epochSeed()));
        }

        if (dropLast) {
            indices = new ArrayList<>(indices.subList(0, totalSize));
        } else {
            int sourceSize = indices.size();
            for (int i = 0; indices.size() < totalSize; i++) {
                indices.add(indices.get(i % sourceSize));
            }
        }

        List<Integer> shard = new ArrayList<>(perReplica);
        for (int i = rank; i < totalSize; i += numReplicas) {
            shard.add(indices.get(i));
        }
        return shard;
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        if (dropLast) {
            return datasetSize / numReplicas;
        }
        return (int) Math.ceil((double) datasetSize / numReplicas);
    }

    public DistributedSampler forEpoch(long epoch) {
        return new DistributedSampler(numReplicas, rank, shuffle, dropLast, seed, epoch);
    }

    public int numReplicas() {
        return numReplicas;
    }

    public int rank() {
        return rank;
    }

    public boolean shuffle() {
        return shuffle;
    }

    public boolean dropLast() {
        return dropLast;
    }

    public long seed() {
        return seed;
    }

    public long epoch() {
        return epoch;
    }

    private long epochSeed() {
        long z = seed + 0x9E3779B97F4A7C15L * (epoch + 1L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private static void requireDatasetSize(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("datasetSize must be non-negative, got: " + datasetSize);
        }
    }
}
