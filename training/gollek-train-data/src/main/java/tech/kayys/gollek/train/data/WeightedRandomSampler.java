package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Samples dataset rows with probability proportional to per-sample weights.
 *
 * <p>This mirrors the core behavior of PyTorch's weighted random sampler:
 * non-positive weights are never selected, replacement controls whether rows
 * can repeat, and a fixed seed makes epochs reproducible.</p>
 */
public final class WeightedRandomSampler implements IndexSampler {
    private final float[] sampleWeights;
    private final int numSamples;
    private final boolean replacement;
    private final long seed;
    private final int positiveWeightCount;

    public WeightedRandomSampler(float[] sampleWeights, int numSamples, boolean replacement, long seed) {
        this.sampleWeights = validateSampleWeights(sampleWeights);
        if (numSamples <= 0) {
            throw new IllegalArgumentException("numSamples must be positive, got: " + numSamples);
        }
        this.numSamples = numSamples;
        this.replacement = replacement;
        this.seed = seed;
        this.positiveWeightCount = positiveWeightCount(this.sampleWeights);
        if (!replacement && numSamples > positiveWeightCount) {
            throw new IllegalArgumentException(
                    "numSamples cannot exceed positive sample weight count without replacement, got: "
                            + numSamples + " > " + positiveWeightCount);
        }
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireSamplerDatasetSize(datasetSize);
        Random random = new Random(seed);
        float[] weights = replacement ? sampleWeights : Arrays.copyOf(sampleWeights, sampleWeights.length);
        List<Integer> indices = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            int index = drawWeightedIndex(weights, random);
            indices.add(index);
            if (!replacement) {
                weights[index] = 0.0f;
            }
        }
        return indices;
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireSamplerDatasetSize(datasetSize);
        return numSamples;
    }

    public float[] sampleWeights() {
        return Arrays.copyOf(sampleWeights, sampleWeights.length);
    }

    public int numSamples() {
        return numSamples;
    }

    public boolean replacement() {
        return replacement;
    }

    public long seed() {
        return seed;
    }

    private void requireSamplerDatasetSize(int datasetSize) {
        if (datasetSize != sampleWeights.length) {
            throw new IllegalArgumentException(
                    "sampleWeights length must match dataset size, got: "
                            + sampleWeights.length + " vs " + datasetSize);
        }
    }

    private static float[] validateSampleWeights(float[] sampleWeights) {
        Objects.requireNonNull(sampleWeights, "sampleWeights must not be null");
        if (sampleWeights.length == 0) {
            throw new IllegalArgumentException("sampleWeights must contain at least one value");
        }
        float[] copy = Arrays.copyOf(sampleWeights, sampleWeights.length);
        double total = 0.0;
        for (float weight : copy) {
            if (!Float.isFinite(weight) || weight < 0.0f) {
                throw new IllegalArgumentException(
                        "sampleWeights must be finite and non-negative, got: " + weight);
            }
            total += weight;
        }
        if (total <= 0.0) {
            throw new IllegalArgumentException("sampleWeights must contain at least one positive value");
        }
        return copy;
    }

    private static int positiveWeightCount(float[] weights) {
        int count = 0;
        for (float weight : weights) {
            if (weight > 0.0f) {
                count++;
            }
        }
        return count;
    }

    private static int drawWeightedIndex(float[] weights, Random random) {
        double total = 0.0;
        for (float weight : weights) {
            total += weight;
        }
        if (total <= 0.0) {
            throw new IllegalStateException("no positive sample weights remain");
        }
        double target = random.nextDouble() * total;
        double cumulative = 0.0;
        int lastPositive = -1;
        for (int i = 0; i < weights.length; i++) {
            float weight = weights[i];
            if (weight <= 0.0f) {
                continue;
            }
            lastPositive = i;
            cumulative += weight;
            if (target < cumulative) {
                return i;
            }
        }
        return lastPositive;
    }
}
