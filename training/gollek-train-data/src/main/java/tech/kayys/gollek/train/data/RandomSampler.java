package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Uniformly samples dataset rows, optionally with replacement.
 */
public final class RandomSampler implements IndexSampler {
    private final Integer numSamples;
    private final boolean replacement;
    private final long seed;

    public RandomSampler(long seed) {
        this(null, false, seed);
    }

    public RandomSampler(int numSamples, boolean replacement, long seed) {
        this(Integer.valueOf(numSamples), replacement, seed);
    }

    private RandomSampler(Integer numSamples, boolean replacement, long seed) {
        if (numSamples != null && numSamples <= 0) {
            throw new IllegalArgumentException("numSamples must be positive, got: " + numSamples);
        }
        this.numSamples = numSamples;
        this.replacement = replacement;
        this.seed = seed;
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        int count = sampleCount(datasetSize);
        Random random = new Random(seed);
        if (replacement) {
            if (datasetSize == 0 && count > 0) {
                throw new IllegalArgumentException("cannot sample from an empty dataset with replacement");
            }
            List<Integer> indices = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                indices.add(random.nextInt(datasetSize));
            }
            return indices;
        }

        List<Integer> indices = new ArrayList<>(datasetSize);
        for (int i = 0; i < datasetSize; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        return count == indices.size() ? indices : new ArrayList<>(indices.subList(0, count));
    }

    @Override
    public int sampleCount(int datasetSize) {
        DataLoaderCounts.requireDatasetSize(datasetSize);
        int count = numSamples == null ? datasetSize : numSamples;
        if (!replacement && count > datasetSize) {
            throw new IllegalArgumentException(
                    "numSamples cannot exceed dataset size without replacement, got: "
                            + count + " > " + datasetSize);
        }
        return count;
    }

    public boolean replacement() {
        return replacement;
    }

    public long seed() {
        return seed;
    }

    public Integer numSamples() {
        return numSamples;
    }
}
