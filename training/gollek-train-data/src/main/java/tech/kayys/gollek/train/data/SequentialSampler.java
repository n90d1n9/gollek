package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.List;

/**
 * Selects every dataset row in natural order.
 */
public final class SequentialSampler implements IndexSampler {

    @Override
    public List<Integer> sample(int datasetSize) {
        requireDatasetSize(datasetSize);
        List<Integer> indices = new ArrayList<>(datasetSize);
        for (int i = 0; i < datasetSize; i++) {
            indices.add(i);
        }
        return indices;
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        return datasetSize;
    }

    private static void requireDatasetSize(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("datasetSize must be non-negative, got: " + datasetSize);
        }
    }
}
