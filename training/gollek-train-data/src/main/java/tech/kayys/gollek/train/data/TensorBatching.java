package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

final class TensorBatching {
    private TensorBatching() {
    }

    static List<Integer> epochIndices(
            DataLoader.TensorDatasetAdapter dataset,
            IndexSampler sampler,
            boolean shuffle,
            Long shuffleSeed) {
        return epochIndices(dataset, sampler, shuffle, shuffleSeed, false, 0L);
    }

    static List<Integer> epochIndices(
            DataLoader.TensorDatasetAdapter dataset,
            IndexSampler sampler,
            boolean shuffle,
            Long shuffleSeed,
            boolean reshuffleEachEpoch,
            long epoch) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        List<Integer> indices = sampler == null
                ? new ArrayList<>(IntStream.range(0, dataset.size()).boxed().toList())
                : new ArrayList<>(sampler.sample(dataset.size()));

        if (sampler == null && shuffle) {
            Long epochSeed = DataLoaderShuffleSeeds.forEpoch(shuffleSeed, reshuffleEachEpoch, epoch);
            if (epochSeed == null) {
                Collections.shuffle(indices, ThreadLocalRandom.current());
            } else {
                Collections.shuffle(indices, new Random(epochSeed));
            }
        }
        return indices;
    }

    static int batchCount(int sampleCount, int batchSize, boolean dropLast) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        return dropLast ? sampleCount / batchSize : (int) Math.ceil((double) sampleCount / batchSize);
    }

    static List<List<Integer>> fixedBatches(List<Integer> indices, int batchSize, boolean dropLast) {
        Objects.requireNonNull(indices, "indices must not be null");
        int count = batchCount(indices.size(), batchSize, dropLast);
        List<List<Integer>> batches = new ArrayList<>(count);
        for (int current = 0; current < count; current++) {
            int start = current * batchSize;
            int end = Math.min(start + batchSize, indices.size());
            batches.add(indices.subList(start, end));
        }
        return batches;
    }
}
