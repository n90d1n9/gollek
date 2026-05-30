package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Selects a fixed subset of dataset rows in the provided order.
 */
public final class SubsetSampler implements IndexSampler {
    private final int[] indices;

    public SubsetSampler(int... indices) {
        Objects.requireNonNull(indices, "indices must not be null");
        this.indices = Arrays.copyOf(indices, indices.length);
        for (int index : this.indices) {
            if (index < 0) {
                throw new IllegalArgumentException("subset indices must be non-negative, got: " + index);
            }
        }
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        requireValidFor(datasetSize);
        List<Integer> selected = new ArrayList<>(indices.length);
        for (int index : indices) {
            selected.add(index);
        }
        return selected;
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireValidFor(datasetSize);
        return indices.length;
    }

    public int[] indices() {
        return Arrays.copyOf(indices, indices.length);
    }

    private void requireValidFor(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("datasetSize must be non-negative, got: " + datasetSize);
        }
        for (int index : indices) {
            if (index >= datasetSize) {
                throw new IllegalArgumentException(
                        "subset index " + index + " out of range for dataset size " + datasetSize);
            }
        }
    }
}
