package tech.kayys.gollek.train.data.internal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

public final class DataLoaderBatchingRules {
    private DataLoaderBatchingRules() {
    }

    public static List<Integer> epochIndices(
            int datasetSize,
            IntFunction<List<Integer>> sampler,
            boolean shuffle,
            Long shuffleSeed,
            boolean reshuffleEachEpoch,
            long epoch) {
        DataLoaderCountRules.requireDatasetSize(datasetSize);
        List<Integer> indices = sampler == null
                ? new ArrayList<>(IntStream.range(0, datasetSize).boxed().toList())
                : new ArrayList<>(sampler.apply(datasetSize));

        if (sampler == null && shuffle) {
            Long epochSeed = DataLoaderShuffleSeedRules.forEpoch(shuffleSeed, reshuffleEachEpoch, epoch);
            if (epochSeed == null) {
                Collections.shuffle(indices, ThreadLocalRandom.current());
            } else {
                Collections.shuffle(indices, new Random(epochSeed));
            }
        }
        return indices;
    }

    public static int batchCount(int sampleCount, int batchSize, boolean dropLast) {
        DataLoaderCountRules.requireSampleCount(sampleCount);
        DataLoaderBatchSizeRules.requirePositive(batchSize);
        return dropLast ? sampleCount / batchSize : (int) Math.ceil((double) sampleCount / batchSize);
    }

    public static List<List<Integer>> fixedBatches(List<Integer> indices, int batchSize, boolean dropLast) {
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
